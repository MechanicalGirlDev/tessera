//! GPU impulse solve for resident rigid spheres and a finite ground plane.
//!
//! Independent potential-contact islands run in parallel. Contacts within an
//! island remain sequential to preserve the impulse dependency order.

use core::mem::size_of;
use core::time::Duration;
use std::collections::BTreeMap;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_broad_phase::GpuPair;
use crate::gpu_lbvh::GpuLbvhResidentPairs;
use crate::gpu_rigid_sphere_contact::{GpuRigidCandidateContacts, GpuRigidSphereContacts};

const IMPULSE_SLOT_BYTES: usize = size_of::<[[f32; 4]; 4]>();

/// Accumulated velocity-solver impulse at one manifold point.
///
/// The world-space impulse acts on `body_b`; a pair applies its opposite to
/// `body_a`. Ground has no `body_a`. Sleeping support can retain cached values,
/// so this is solver history, not a measurement of net momentum change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidContactImpulse {
    /// First body, or `None` for the ground plane.
    pub body_a: Option<u32>,
    /// Body receiving the positive impulse.
    pub body_b: u32,
    /// Point index within the contact manifold.
    pub point_index: u32,
    /// Nonnegative normal impulse.
    pub normal_impulse: f32,
    /// World-space contact normal pointing toward `body_b`.
    pub normal: [f32; 3],
    /// World-space Coulomb friction impulse.
    pub tangent_impulse: [f32; 3],
}

impl GpuRigidContactImpulse {
    /// Return the complete world-space impulse on the second body.
    pub fn impulse_on_body_b(&self) -> [f32; 3] {
        core::array::from_fn(|axis| {
            self.normal[axis] * self.normal_impulse + self.tangent_impulse[axis]
        })
    }
}

/// Sparse impulse history from the most recent submitted contact solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuRigidContactImpulseReadback {
    /// Solver substep duration; absent after cache invalidation.
    pub dt: Option<f32>,
    /// Manifold points with positive normal impulse.
    pub contacts: Vec<GpuRigidContactImpulse>,
}

fn read_impulse_slots(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: Option<&wgpu::Buffer>,
    dt_bits: u32,
    slots: &[(u64, Option<u32>, u32, u32)],
) -> Result<GpuRigidContactImpulseReadback, GpuRigidSphereSolveError> {
    let Some(buffer) = buffer else {
        return Ok(GpuRigidContactImpulseReadback::default());
    };
    let mut result = GpuRigidContactImpulseReadback {
        dt: Some(f32::from_bits(dt_bits)),
        contacts: Vec::new(),
    };
    if slots.is_empty() {
        return Ok(result);
    }
    let bytes = slots.len() as u64 * IMPULSE_SLOT_BYTES as u64;
    if bytes > device.limits().max_buffer_size {
        return Err(GpuRigidSphereSolveError::Capacity);
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tessera contact impulse readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (index, &(slot, _, _, _)) in slots.iter().enumerate() {
        let offset = slot * IMPULSE_SLOT_BYTES as u64;
        if offset + IMPULSE_SLOT_BYTES as u64 > buffer.size() {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        encoder.copy_buffer_to_buffer(
            buffer,
            offset,
            &staging,
            index as u64 * IMPULSE_SLOT_BYTES as u64,
            IMPULSE_SLOT_BYTES as u64,
        );
    }
    let _submission = queue.submit(Some(encoder.finish()));
    let (sender, receiver) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |value| {
            let _ = sender.send(value);
        });
    let _status = device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .map_err(|e| GpuRigidSphereSolveError::Readback(e.to_string()))?;
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|e| GpuRigidSphereSolveError::Readback(e.to_string()))?
        .map_err(|e| GpuRigidSphereSolveError::Readback(e.to_string()))?;
    let view = staging.slice(..).get_mapped_range();
    for (index, &(_, body_a, body_b, point_index)) in slots.iter().enumerate() {
        let raw = &view[index * IMPULSE_SLOT_BYTES..(index + 1) * IMPULSE_SLOT_BYTES];
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect();
        if values[3] > 0.0 {
            result.contacts.push(GpuRigidContactImpulse {
                body_a,
                body_b,
                point_index,
                normal_impulse: values[3],
                normal: [values[5], values[6], values[7]],
                tangent_impulse: [values[0], values[1], values[2]],
            });
        }
    }
    drop(view);
    staging.unmap();
    Ok(result)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SolverParams {
    values: [f32; 4],
    counts: [u32; 4],
    flags: [u32; 4],
    temporal: [f32; 4],
    static_temporal: [f32; 4],
}

#[derive(Clone, Copy)]
struct SolveSettings {
    params: GpuRigidSphereSolveParams,
    temporal: [f32; 4],
    static_temporal: [f32; 4],
}
impl SolveSettings {
    fn pgs(params: GpuRigidSphereSolveParams) -> Self {
        Self {
            params,
            temporal: [0.0; 4],
            static_temporal: [0.0; 4],
        }
    }
}

/// GPU-resident impulse history for repeated sphere contacts.
#[derive(Debug, Default)]
pub struct GpuRigidSphereImpulseCache {
    buffer: Option<wgpu::Buffer>,
    pairs: Vec<GpuPair>,
    pair_stride: u32,
    body_count: usize,
    ground_enabled: bool,
    ground_stride: u32,
    dt_bits: u32,
}

impl GpuRigidSphereImpulseCache {
    /// Read accumulated solver impulses after submitting the solve encoder.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuRigidContactImpulseReadback, GpuRigidSphereSolveError> {
        let mut slots = Vec::new();
        for (index, pair) in self.pairs.iter().enumerate() {
            for point in 0..self.pair_stride {
                slots.push((
                    index as u64 * u64::from(self.pair_stride) + u64::from(point),
                    Some(pair.a),
                    pair.b,
                    point,
                ));
            }
        }
        if self.ground_enabled {
            let start = self.pairs.len() as u64 * u64::from(self.pair_stride);
            for body in 0..self.body_count as u32 {
                for point in 0..self.ground_stride {
                    slots.push((
                        start + u64::from(body) * u64::from(self.ground_stride) + u64::from(point),
                        None,
                        body,
                        point,
                    ));
                }
            }
        }
        read_impulse_slots(device, queue, self.buffer.as_ref(), self.dt_bits, &slots)
    }

    /// Discard impulses after a scene edit or complete reset.
    pub fn clear(&mut self) {
        self.buffer = None;
        self.pairs.clear();
        self.pair_stride = 0;
        self.body_count = 0;
        self.ground_enabled = false;
        self.ground_stride = 0;
        self.dt_bits = 0;
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        scratch_size: u64,
        dt: f32,
    ) -> bool {
        let same_pairs = self.pairs.len() == contacts.pairs().len()
            && self
                .pairs
                .iter()
                .zip(contacts.pairs())
                .all(|(a, b)| a.a == b.a && a.b == b.b);
        if self.buffer.is_some()
            && self.body_count == contacts.body_count()
            && self.pair_stride == contacts.pair_contact_stride()
            && self.ground_enabled == contacts.has_ground()
            && self.ground_stride == contacts.ground_contact_stride()
            && self.dt_bits == dt.to_bits()
            && same_pairs
        {
            return true;
        }
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident sphere cached impulses"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.pairs = contacts.pairs().to_vec();
        self.pair_stride = contacts.pair_contact_stride();
        self.body_count = contacts.body_count();
        self.ground_enabled = contacts.has_ground();
        self.ground_stride = contacts.ground_contact_stride();
        self.dt_bits = dt.to_bits();
        false
    }

    /// Retain GPU impulse slots for pairs that survive a broad-phase update.
    pub(crate) fn remap_candidate_pairs(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
    ) -> Result<(), GpuRigidSphereSolveError> {
        let Some(previous) = &self.buffer else {
            return Ok(());
        };
        if self.body_count != contacts.body_count()
            || self.pair_stride != contacts.pair_contact_stride()
            || self.ground_enabled != contacts.has_ground()
            || self.ground_stride != contacts.ground_contact_stride()
            || self.dt_bits != dt.to_bits()
        {
            self.clear();
            return Ok(());
        }
        let updated = contacts.pairs();
        if self.pairs.len() == updated.len()
            && self
                .pairs
                .iter()
                .zip(updated)
                .all(|(old, new)| old.a == new.a && old.b == new.b)
        {
            return Ok(());
        }
        let pair_stride = u64::from(self.pair_stride);
        let ground_stride = u64::from(self.ground_stride);
        let ground_slots = if self.ground_enabled {
            self.body_count as u64 * ground_stride
        } else {
            0
        };
        let bytes = (updated.len() as u64)
            .checked_mul(pair_stride)
            .and_then(|slots| slots.checked_add(ground_slots))
            .and_then(|slots| slots.checked_mul(IMPULSE_SLOT_BYTES as u64))
            .ok_or(GpuRigidSphereSolveError::Capacity)?;
        if bytes > u64::from(device.limits().max_storage_buffer_binding_size)
            || bytes > device.limits().max_buffer_size
        {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let replacement = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera remapped resident contact impulses"),
            size: bytes.max(IMPULSE_SLOT_BYTES as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let previous_pairs: BTreeMap<_, _> = self
            .pairs
            .iter()
            .enumerate()
            .map(|(index, pair)| ((pair.a, pair.b), index))
            .collect();
        let matches: Vec<_> = updated
            .iter()
            .enumerate()
            .filter_map(|(new_index, pair)| {
                previous_pairs
                    .get(&(pair.a, pair.b))
                    .map(|&old_index| (new_index, old_index))
            })
            .collect();
        let pair_bytes = pair_stride * IMPULSE_SLOT_BYTES as u64;
        let mut run_start = 0;
        while run_start < matches.len() {
            let mut run_end = run_start + 1;
            while run_end < matches.len()
                && matches[run_end].0 == matches[run_end - 1].0 + 1
                && matches[run_end].1 == matches[run_end - 1].1 + 1
            {
                run_end += 1;
            }
            encoder.copy_buffer_to_buffer(
                previous,
                matches[run_start].1 as u64 * pair_bytes,
                &replacement,
                matches[run_start].0 as u64 * pair_bytes,
                (run_end - run_start) as u64 * pair_bytes,
            );
            run_start = run_end;
        }
        if ground_slots > 0 {
            encoder.copy_buffer_to_buffer(
                previous,
                self.pairs.len() as u64 * pair_bytes,
                &replacement,
                updated.len() as u64 * pair_bytes,
                ground_slots * IMPULSE_SLOT_BYTES as u64,
            );
        }
        self.buffer = Some(replacement);
        self.pairs = updated.to_vec();
        Ok(())
    }

    /// Invalidate one edited body's contacts without discarding other islands.
    pub fn invalidate_body(&self, queue: &wgpu::Queue, body: usize) {
        let Some(buffer) = &self.buffer else {
            return;
        };
        let slot_bytes = IMPULSE_SLOT_BYTES as u64;
        let zero = [0u8; IMPULSE_SLOT_BYTES];
        for (slot, pair) in self.pairs.iter().enumerate() {
            if pair.a as usize == body || pair.b as usize == body {
                let first = slot * self.pair_stride as usize;
                for point in first..first + self.pair_stride as usize {
                    queue.write_buffer(buffer, point as u64 * slot_bytes, &zero);
                }
            }
        }
        if self.ground_enabled && body < self.body_count {
            let first =
                self.pairs.len() * self.pair_stride as usize + body * self.ground_stride as usize;
            for slot in first..first + self.ground_stride as usize {
                queue.write_buffer(buffer, slot as u64 * slot_bytes, &zero);
            }
        }
    }
}

/// Solver coefficients for one resident sphere substep.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidSphereSolveParams {
    /// Default Coulomb friction for colliders without an override.
    pub friction: f32,
    /// Default restitution for colliders without an override.
    pub restitution: f32,
    /// Fraction of penetration depth resolved as a velocity bias per step.
    pub bias_factor: f32,
    /// Sequential impulse iterations within one GPU dispatch.
    pub iterations: u32,
}

impl Default for GpuRigidSphereSolveParams {
    fn default() -> Self {
        Self {
            friction: 1.0,
            restitution: 0.0,
            bias_factor: 0.2,
            iterations: 12,
        }
    }
}

/// Soft temporal contact coefficients for one substep.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidTemporalSolveParams {
    /// Default Coulomb friction when no collider material overrides it.
    pub friction: f32,
    /// Default restitution when no collider material overrides it.
    pub restitution: f32,
    /// Normal spring frequency in hertz; must be positive.
    pub normal_frequency: f32,
    /// Normal spring damping ratio; must be nonnegative.
    pub damping_ratio: f32,
    /// Normal spring frequency for contacts with a fixed body or ground.
    pub static_normal_frequency: f32,
    /// Normal damping ratio for contacts with a fixed body or ground.
    pub static_damping_ratio: f32,
    /// Maximum penetration correction speed in metres per second.
    pub max_corrective_velocity: f32,
    /// Sequential impulse sweeps in each bias or relax pass.
    pub iterations: u32,
}
impl Default for GpuRigidTemporalSolveParams {
    fn default() -> Self {
        Self {
            friction: 1.0,
            restitution: 0.0,
            normal_frequency: 30.0,
            damping_ratio: 1.0,
            static_normal_frequency: 60.0,
            static_damping_ratio: 1.0,
            max_corrective_velocity: 3.0,
            iterations: 1,
        }
    }
}
impl GpuRigidTemporalSolveParams {
    pub(crate) fn validate(self, dt: f32) -> Result<(), GpuRigidSphereSolveError> {
        self.settings(dt, false).map(|_| ())
    }
    fn settings(self, dt: f32, relax: bool) -> Result<SolveSettings, GpuRigidSphereSolveError> {
        if !dt.is_finite()
            || dt <= 0.0
            || !self.normal_frequency.is_finite()
            || self.normal_frequency <= 0.0
            || !self.damping_ratio.is_finite()
            || self.damping_ratio < 0.0
            || !self.static_normal_frequency.is_finite()
            || self.static_normal_frequency <= 0.0
            || !self.static_damping_ratio.is_finite()
            || self.static_damping_ratio < 0.0
            || !self.max_corrective_velocity.is_finite()
            || self.max_corrective_velocity <= 0.0
            || !self.friction.is_finite()
            || !(0.0..=f32::MAX.sqrt()).contains(&self.friction)
            || !self.restitution.is_finite()
            || !(0.0..=f32::MAX.sqrt()).contains(&self.restitution)
            || self.iterations == 0
        {
            return Err(GpuRigidSphereSolveError::InvalidInput);
        }
        let temporal = self.softness(dt, relax, self.normal_frequency, self.damping_ratio);
        let static_temporal = self.softness(
            dt,
            relax,
            self.static_normal_frequency,
            self.static_damping_ratio,
        );
        if temporal
            .iter()
            .chain(static_temporal.iter())
            .any(|x| !x.is_finite())
        {
            return Err(GpuRigidSphereSolveError::InvalidInput);
        }
        Ok(SolveSettings {
            params: GpuRigidSphereSolveParams {
                friction: self.friction,
                restitution: self.restitution,
                bias_factor: 0.0,
                iterations: self.iterations,
            },
            temporal,
            static_temporal,
        })
    }
    fn softness(self, dt: f32, relax: bool, frequency: f32, damping_ratio: f32) -> [f32; 4] {
        let omega = 2.0 * core::f32::consts::PI * frequency;
        let a1 = 2.0 * damping_ratio + dt * omega;
        let a2 = dt * omega * a1;
        let inverse = 1.0 / (1.0 + a2);
        [
            a2 * inverse,
            inverse,
            omega / a1,
            if relax {
                -self.max_corrective_velocity
            } else {
                self.max_corrective_velocity
            },
        ]
    }
}

/// Invalid coefficients, incompatible buffers, or excessive serial work.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidSphereSolveError {
    /// A timestep or coefficient was invalid.
    #[error("invalid GPU resident sphere solve input")]
    InvalidInput,
    /// An island exceeds storage capacity or the serial island work limit.
    #[error("GPU resident sphere solve exceeds capacity")]
    Capacity,
    /// Mapping the GPU solve status failed.
    #[error("GPU resident sphere solve status readback failed: {0}")]
    Readback(String),
}

/// Status of a GPU-resident LBVH candidate solve.
///
/// The solve writes one status word. Read it after submission to detect a
/// candidate overflow or serial work limit before using the updated state.
#[derive(Debug)]
pub struct GpuRigidCandidateSolve {
    /// Zero on success; one when candidate or serial-work capacity was exceeded.
    pub status: wgpu::Buffer,
}

impl GpuRigidCandidateSolve {
    /// Check the GPU status after the command encoder has been submitted.
    pub fn readback_status(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), GpuRigidSphereSolveError> {
        self.readback_status_inner(device, queue, None).map(|_| ())
    }

    /// Check the solve and return only the valid candidate count in one map.
    pub fn readback_status_and_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidates: &GpuLbvhResidentPairs,
    ) -> Result<u32, GpuRigidSphereSolveError> {
        self.readback_status_inner(device, queue, Some(candidates))
    }

    fn readback_status_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidates: Option<&GpuLbvhResidentPairs>,
    ) -> Result<u32, GpuRigidSphereSolveError> {
        let bytes = if candidates.is_some() { 12 } else { 4 };
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident candidate solve status readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera candidate solve status readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.status, 0, &readback, 0, 4);
        if let Some(candidates) = candidates {
            encoder.copy_buffer_to_buffer(&candidates.counter, 0, &readback, 4, 8);
        }
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| GpuRigidSphereSolveError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| GpuRigidSphereSolveError::Readback(error.to_string()))?
            .map_err(|error| GpuRigidSphereSolveError::Readback(error.to_string()))?;
        let view = readback.slice(..).get_mapped_range();
        let status = u32::from_le_bytes([view[0], view[1], view[2], view[3]]);
        let count = if candidates.is_some() {
            u32::from_le_bytes([view[4], view[5], view[6], view[7]])
        } else {
            0
        };
        let overflow = if candidates.is_some() {
            u32::from_le_bytes([view[8], view[9], view[10], view[11]])
        } else {
            0
        };
        drop(view);
        readback.unmap();
        if status == 0
            && overflow == 0
            && candidates.is_none_or(|candidates| count <= candidates.pair_capacity)
        {
            Ok(count)
        } else {
            Err(GpuRigidSphereSolveError::Capacity)
        }
    }
}

/// GPU impulse history indexed by the stable pair of body IDs.
///
/// This cache keeps the full pair-capacity buffer between resident LBVH steps.
/// Clear it after body, shape, material, or collision-filter edits.
#[derive(Debug, Default)]
pub struct GpuRigidCandidateImpulseCache {
    buffer: Option<wgpu::Buffer>,
    body_count: u32,
    pair_capacity: u32,
    pair_stride: u32,
    ground_stride: u32,
    ground_enabled: bool,
    dt_bits: u32,
    generation: u32,
}

impl GpuRigidCandidateImpulseCache {
    /// Read current candidate impulses without transferring the dense pair cache.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidates: &GpuLbvhResidentPairs,
    ) -> Result<GpuRigidContactImpulseReadback, GpuRigidSphereSolveError> {
        if self.buffer.is_none() {
            return Ok(GpuRigidContactImpulseReadback::default());
        }
        if candidates.collider_count != self.body_count
            || candidates.pair_capacity != self.pair_capacity
        {
            return Err(GpuRigidSphereSolveError::InvalidInput);
        }
        let pairs = candidates
            .readback(device, queue)
            .map_err(|e| GpuRigidSphereSolveError::Readback(e.to_string()))?;
        let mut slots = Vec::new();
        for pair in pairs {
            let a = pair.a.min(pair.b);
            let b = pair.a.max(pair.b);
            if b >= self.body_count || a == b {
                return Err(GpuRigidSphereSolveError::InvalidInput);
            }
            let row = u64::from(a) * (2 * u64::from(self.body_count) - u64::from(a) - 1) / 2;
            for point in 0..self.pair_stride {
                slots.push((
                    (row + u64::from(b - a - 1)) * u64::from(self.pair_stride) + u64::from(point),
                    Some(pair.a),
                    pair.b,
                    point,
                ));
            }
        }
        if self.ground_enabled {
            let start = u64::from(self.pair_capacity) * u64::from(self.pair_stride);
            for body in 0..self.body_count {
                for point in 0..self.ground_stride {
                    slots.push((
                        start + u64::from(body) * u64::from(self.ground_stride) + u64::from(point),
                        None,
                        body,
                        point,
                    ));
                }
            }
        }
        read_impulse_slots(device, queue, self.buffer.as_ref(), self.dt_bits, &slots)
    }

    /// Discard all resident candidate impulse history.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        scratch_size: u64,
        contacts: &GpuRigidSphereContacts,
        candidates: &GpuLbvhResidentPairs,
        output: &GpuRigidCandidateContacts,
        dt: f32,
    ) {
        let same = self.buffer.is_some()
            && self.body_count == candidates.collider_count
            && self.pair_capacity == candidates.pair_capacity
            && self.pair_stride == output.pair_stride
            && self.ground_stride == output.ground_stride
            && self.ground_enabled == contacts.has_ground()
            && self.dt_bits == dt.to_bits()
            && self.generation < 16_777_216;
        if same {
            self.generation += 1;
            return;
        }
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident candidate cached impulses"),
            size: scratch_size.max(IMPULSE_SLOT_BYTES as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.body_count = candidates.collider_count;
        self.pair_capacity = candidates.pair_capacity;
        self.pair_stride = output.pair_stride;
        self.ground_stride = output.ground_stride;
        self.ground_enabled = contacts.has_ground();
        self.dt_bits = dt.to_bits();
        self.generation = 1;
    }

    fn buffer(&self) -> Result<&wgpu::Buffer, GpuRigidSphereSolveError> {
        self.buffer
            .as_ref()
            .ok_or(GpuRigidSphereSolveError::Capacity)
    }
}

/// Reusable velocity-level impulse kernel for spheres and finite ground.
#[derive(Debug)]
pub struct GpuRigidSphereSolver {
    pipeline: wgpu::ComputePipeline,
    candidate_pipeline: wgpu::ComputePipeline,
    candidate_label_pipeline: wgpu::ComputePipeline,
    candidate_island_pipeline: wgpu::ComputePipeline,
    candidate_warm_pipeline: wgpu::ComputePipeline,
    candidate_colored_pipeline: wgpu::ComputePipeline,
    coupled_warm_pipeline: wgpu::ComputePipeline,
    coupled_first_pipeline: wgpu::ComputePipeline,
    coupled_next_pipeline: wgpu::ComputePipeline,
    temporal_capture_pipeline: wgpu::ComputePipeline,
    temporal_bias_pipeline: wgpu::ComputePipeline,
    temporal_relax_pipeline: wgpu::ComputePipeline,
}

/// Contact passes for one step of an alternating contact and joint solve.
#[derive(Debug)]
pub struct GpuRigidSphereCoupledStep<'a> {
    solver: &'a GpuRigidSphereSolver,
    bind_group: wgpu::BindGroup,
    island_count: u32,
}

/// Prepared soft bias and rigid unbiased relax passes sharing one impulse buffer.
///
/// Encode bias once per prepared substep, integrate positions and refresh the
/// captured contact anchors, then encode relax. Reprepare for the next substep
/// with the same cache and duration. This type does not schedule a full TGS frame.
#[derive(Debug)]
pub struct GpuRigidTemporalContactStep<'a> {
    solver: &'a GpuRigidSphereSolver,
    bias: wgpu::BindGroup,
    relax: wgpu::BindGroup,
    island_count: u32,
}
impl GpuRigidTemporalContactStep<'_> {
    /// Capture each body's own local contact point once after narrow phase per frame.
    /// Capture the restitution target from the incoming contact velocity as well;
    /// bias and relax preserve this target until the next frame capture.
    ///
    /// Encode before bias or pose integration, at the same poses as contact transport
    /// capture. Subsequent substeps preserve these anchors through warm start and relax.
    pub fn encode_capture_anchors(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.temporal_capture_pipeline, &self.bias);
    }

    /// Warm start once, then solve soft normal and Coulomb friction rows with bias.
    pub fn encode_bias(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.temporal_bias_pipeline, &self.bias);
    }
    /// Warm start once before alternating temporal contact and joint sweeps.
    pub fn encode_warm(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.coupled_warm_pipeline, &self.bias);
    }
    /// Solve one biased sweep, preserving the captured frame restitution target.
    pub fn encode_bias_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.coupled_next_pipeline, &self.bias);
    }
    /// Solve one relaxation sweep without applying cached impulses again.
    pub fn encode_relax_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.coupled_next_pipeline, &self.relax);
    }
    /// Solve with refreshed speculative targets and no penetration correction bias.
    ///
    /// Retains soft coefficients and accumulated impulses without reapplying warm start.
    pub fn encode_relax(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.solver.temporal_relax_pipeline, &self.relax);
    }
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        inputs: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera temporal contact solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, inputs, &[]);
        pass.dispatch_workgroups(self.island_count.div_ceil(64), 1, 1);
    }
}

impl GpuRigidSphereCoupledStep<'_> {
    /// Apply cached contact impulses once, before all coupled iterations.
    pub fn encode_warm(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_pass(encoder, &self.solver.coupled_warm_pipeline);
    }

    /// Solve one contact iteration, retaining accumulated impulses on the GPU.
    pub fn encode_iteration(&self, encoder: &mut wgpu::CommandEncoder, first: bool) {
        let pipeline = if first {
            &self.solver.coupled_first_pipeline
        } else {
            &self.solver.coupled_next_pipeline
        };
        self.encode_pass(encoder, pipeline);
    }

    fn encode_pass(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident coupled contact pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.island_count.div_ceil(64), 1, 1);
    }
}

impl GpuRigidSphereSolver {
    /// Compile the rigid sphere impulse kernel for this device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident rigid sphere solver"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_sphere_solver.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera resident contact solver bindings"),
            entries: &[
                storage(0, false),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(6, true),
                storage(7, true),
                storage(8, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera coupled contact pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident rigid sphere impulse pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let coupled_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let coupled_warm_pipeline =
            coupled_pipeline("Tessera coupled contact warm", "coupled_warm");
        let coupled_first_pipeline =
            coupled_pipeline("Tessera coupled contact first", "coupled_first");
        let coupled_next_pipeline =
            coupled_pipeline("Tessera coupled contact next", "coupled_next");
        let temporal_capture_pipeline =
            coupled_pipeline("Tessera temporal anchor capture", "temporal_capture");
        let temporal_bias_pipeline =
            coupled_pipeline("Tessera temporal bias solve", "temporal_bias");
        let temporal_relax_pipeline =
            coupled_pipeline("Tessera temporal relax solve", "temporal_relax");
        let candidate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident candidate solver"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("gpu_rigid_sphere_solver.wgsl"),
                    include_str!("gpu_rigid_candidate_solver.wgsl")
                )
                .into(),
            ),
        });
        let candidate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera resident candidate solver bindings"),
            entries: &[
                storage(0, false),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(8, true),
                storage(9, true),
                storage(10, false),
            ],
        });
        let candidate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Tessera resident candidate solver pipeline layout"),
                bind_group_layouts: &[&candidate_layout],
                immediate_size: 0,
            });
        let candidate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident candidate impulse pipeline"),
            layout: Some(&candidate_pipeline_layout),
            module: &candidate_shader,
            entry_point: Some("candidate_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let candidate_phase = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&candidate_pipeline_layout),
                module: &candidate_shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let candidate_label_pipeline = candidate_phase(
            "Tessera resident candidate island labels",
            "candidate_label",
        );
        let candidate_island_pipeline = candidate_phase(
            "Tessera resident candidate island impulses",
            "candidate_islands",
        );
        let candidate_warm_pipeline = candidate_phase(
            "Tessera resident candidate warm islands",
            "candidate_island_warm",
        );
        let candidate_colored_pipeline = candidate_phase(
            "Tessera resident colored candidate impulses",
            "candidate_colored",
        );
        Self {
            pipeline,
            temporal_capture_pipeline,
            temporal_bias_pipeline,
            temporal_relax_pipeline,
            candidate_pipeline,
            candidate_label_pipeline,
            candidate_island_pipeline,
            candidate_warm_pipeline,
            candidate_colored_pipeline,
            coupled_warm_pipeline,
            coupled_first_pipeline,
            coupled_next_pipeline,
        }
    }

    /// Solve contacts from GPU-resident LBVH candidates in the same encoder.
    ///
    /// The candidate count stays on the GPU. Larger workloads partition
    /// independent contact islands on the GPU before solving them in parallel.
    /// Submit the encoder before checking the returned status buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_candidates(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        candidates: &GpuLbvhResidentPairs,
        output: &GpuRigidCandidateContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
    ) -> Result<GpuRigidCandidateSolve, GpuRigidSphereSolveError> {
        self.encode_resident_candidates_inner(
            device, encoder, contacts, candidates, output, dt, params, None,
        )
    }

    /// Solve GPU-resident LBVH contacts with pair-stable impulse history.
    ///
    /// The cache needs no CPU pair list. Clear it after editing body state,
    /// collision geometry, materials, or filters.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_candidates_cached(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        candidates: &GpuLbvhResidentPairs,
        output: &GpuRigidCandidateContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
        cache: &mut GpuRigidCandidateImpulseCache,
    ) -> Result<GpuRigidCandidateSolve, GpuRigidSphereSolveError> {
        self.encode_resident_candidates_inner(
            device,
            encoder,
            contacts,
            candidates,
            output,
            dt,
            params,
            Some(cache),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_resident_candidates_inner(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        candidates: &GpuLbvhResidentPairs,
        output: &GpuRigidCandidateContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
        cache: Option<&mut GpuRigidCandidateImpulseCache>,
    ) -> Result<GpuRigidCandidateSolve, GpuRigidSphereSolveError> {
        if !dt.is_finite()
            || dt <= 0.0
            || !params.friction.is_finite()
            || params.friction < 0.0
            || params.friction > f32::MAX.sqrt()
            || !params.restitution.is_finite()
            || params.restitution < 0.0
            || params.restitution > f32::MAX.sqrt()
            || !params.bias_factor.is_finite()
            || !(0.0..=1.0).contains(&params.bias_factor)
            || params.iterations == 0
            || candidates.collider_count as usize != contacts.body_count()
            || candidates.pair_capacity != output.pair_capacity
            || output.pair_stride != contacts.pair_contact_stride()
            || output.ground_stride != contacts.ground_contact_stride()
        {
            return Err(GpuRigidSphereSolveError::InvalidInput);
        }
        if params.iterations > 20_000 {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let body_count =
            u32::try_from(contacts.body_count()).map_err(|_| GpuRigidSphereSolveError::Capacity)?;
        let ground_slots = if contacts.has_ground() {
            body_count
                .checked_mul(output.ground_stride)
                .ok_or(GpuRigidSphereSolveError::Capacity)?
        } else {
            0
        };
        let scratch_slots = candidates
            .pair_capacity
            .checked_mul(output.pair_stride)
            .and_then(|slots| slots.checked_add(ground_slots))
            .ok_or(GpuRigidSphereSolveError::Capacity)?;
        let scratch_size = u64::from(scratch_slots) * IMPULSE_SLOT_BYTES as u64;
        if scratch_size > u64::from(device.limits().max_storage_buffer_binding_size)
            || scratch_size > device.limits().max_buffer_size
        {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let islanded = u64::from(scratch_slots) * u64::from(params.iterations) > 20_000;
        // Status, seven body-sized arrays, pair/body indices, and pair colors.
        let status_words = if islanded {
            u64::from(body_count) * 8 + u64::from(candidates.pair_capacity) * 2 + 1
        } else {
            1
        };
        let status_size = status_words * 4;
        if status_size > u64::from(device.limits().max_storage_buffer_binding_size)
            || status_size > device.limits().max_buffer_size
            || (islanded && body_count > device.limits().max_compute_workgroups_per_dimension)
        {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let scratch_owned;
        let (scratch, cached, generation) = if let Some(cache) = cache {
            cache.prepare(device, scratch_size, contacts, candidates, output, dt);
            (cache.buffer()?, true, cache.generation)
        } else {
            scratch_owned = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera resident candidate accumulated impulses"),
                size: scratch_size.max(IMPULSE_SLOT_BYTES as u64),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            (&scratch_owned, false, 0)
        };
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident candidate solve status"),
            size: status_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident candidate solver params"),
            contents: bytemuck::bytes_of(&SolverParams {
                temporal: [0.0; 4],
                static_temporal: [0.0; 4],
                values: [dt, params.friction, params.restitution, params.bias_factor],
                counts: [
                    generation,
                    candidates.pair_capacity,
                    u32::from(contacts.has_ground()),
                    params.iterations,
                ],
                flags: [
                    u32::from(cached),
                    output.ground_stride,
                    body_count,
                    output.pair_stride,
                ],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident candidate solver inputs"),
            layout: &self.candidate_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: contacts.state_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates.pairs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.pairs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.ground.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: solver_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: contacts.material_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: candidates.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident candidate impulse solve"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        if islanded {
            pass.set_pipeline(&self.candidate_label_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.candidate_warm_pipeline);
            pass.dispatch_workgroups(body_count.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.candidate_colored_pipeline);
            pass.dispatch_workgroups(body_count, 1, 1);
            pass.set_pipeline(&self.candidate_island_pipeline);
            pass.dispatch_workgroups(body_count.div_ceil(64), 1, 1);
        } else {
            pass.set_pipeline(&self.candidate_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        drop(pass);
        Ok(GpuRigidCandidateSolve { status })
    }

    /// Encode contact impulses after `GpuRigidSphereContacts::encode`.
    ///
    /// The solver updates the shared state buffer in place without readback.
    /// Explicit per-collider materials override the supplied default values.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
    ) -> Result<(), GpuRigidSphereSolveError> {
        self.encode_inner(device, encoder, contacts, dt, params, None)
    }

    /// Encode a solve that reuses impulses when contact topology is unchanged.
    ///
    /// The cache must be cleared when bodies or contact materials are edited.
    /// Clear it if the command encoder is discarded without submission.
    pub fn encode_cached(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
        cache: &mut GpuRigidSphereImpulseCache,
    ) -> Result<(), GpuRigidSphereSolveError> {
        self.encode_inner(device, encoder, contacts, dt, params, Some(cache))
    }

    /// Prepare soft bias and rigid relax passes for fixed resident contact topology.
    ///
    /// Contacts must include signed depth from captured anchors. Bias and relax
    /// share cached impulses; refresh anchors after pose integration before relax.
    /// If an encoder is discarded, clear the cache before preparing another step.
    pub fn prepare_temporal_cached<'a>(
        &'a self,
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        params: GpuRigidTemporalSolveParams,
        cache: &mut GpuRigidSphereImpulseCache,
    ) -> Result<Option<GpuRigidTemporalContactStep<'a>>, GpuRigidSphereSolveError> {
        let bias_settings = params.settings(dt, false)?;
        let relax_settings = params.settings(dt, true)?;
        let Some((bias, island_count)) =
            self.prepare_bind_group(device, contacts, dt, bias_settings, Some(cache))?
        else {
            return Ok(None);
        };
        let (relax, _) = self
            .prepare_bind_group(device, contacts, dt, relax_settings, Some(cache))?
            .ok_or(GpuRigidSphereSolveError::InvalidInput)?;
        Ok(Some(GpuRigidTemporalContactStep {
            solver: self,
            bias,
            relax,
            island_count,
        }))
    }

    /// Prepare contact passes for alternating contact and joint iterations.
    pub fn prepare_coupled_cached<'a>(
        &'a self,
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
        cache: &mut GpuRigidSphereImpulseCache,
    ) -> Result<Option<GpuRigidSphereCoupledStep<'a>>, GpuRigidSphereSolveError> {
        Ok(self
            .prepare_bind_group(
                device,
                contacts,
                dt,
                SolveSettings::pgs(params),
                Some(cache),
            )?
            .map(|(bind_group, island_count)| GpuRigidSphereCoupledStep {
                solver: self,
                bind_group,
                island_count,
            }))
    }

    fn encode_inner(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        params: GpuRigidSphereSolveParams,
        cache: Option<&mut GpuRigidSphereImpulseCache>,
    ) -> Result<(), GpuRigidSphereSolveError> {
        let Some((bind_group, island_count)) =
            self.prepare_bind_group(device, contacts, dt, SolveSettings::pgs(params), cache)?
        else {
            return Ok(());
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident sphere impulse solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(island_count.div_ceil(64), 1, 1);
        Ok(())
    }

    fn prepare_bind_group(
        &self,
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        dt: f32,
        settings: SolveSettings,
        cache: Option<&mut GpuRigidSphereImpulseCache>,
    ) -> Result<Option<(wgpu::BindGroup, u32)>, GpuRigidSphereSolveError> {
        let params = settings.params;
        if !dt.is_finite()
            || dt <= 0.0
            || !params.friction.is_finite()
            || params.friction < 0.0
            || params.friction > f32::MAX.sqrt()
            || !params.restitution.is_finite()
            || params.restitution < 0.0
            || params.restitution > f32::MAX.sqrt()
            || !params.bias_factor.is_finite()
            || !(0.0..=1.0).contains(&params.bias_factor)
            || params.iterations == 0
        {
            return Err(GpuRigidSphereSolveError::InvalidInput);
        }
        let pair_count = u32::try_from(contacts.pairs().len())
            .map_err(|_| GpuRigidSphereSolveError::Capacity)?;
        let body_count =
            u32::try_from(contacts.body_count()).map_err(|_| GpuRigidSphereSolveError::Capacity)?;
        let ground_count = if contacts.has_ground() { body_count } else { 0 };
        if contacts.island_count() == 0 {
            if let Some(cache) = cache {
                cache.clear();
            }
            return Ok(None);
        }
        if u64::from(params.iterations) * u64::from(contacts.max_island_contacts()) > 20_000 {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let scratch_count = pair_count
            .checked_mul(contacts.pair_contact_stride())
            .and_then(|pairs| {
                pairs.checked_add(ground_count.checked_mul(contacts.ground_contact_stride())?)
            })
            .ok_or(GpuRigidSphereSolveError::Capacity)?;
        let scratch_size = u64::from(scratch_count) * IMPULSE_SLOT_BYTES as u64;
        if scratch_size > u64::from(device.limits().max_storage_buffer_binding_size)
            || scratch_size > device.limits().max_buffer_size
        {
            return Err(GpuRigidSphereSolveError::Capacity);
        }
        let scratch_owned: wgpu::Buffer;
        let (scratch, warm) = if let Some(cache) = cache {
            let warm = cache.prepare(device, contacts, scratch_size, dt);
            (
                cache
                    .buffer
                    .as_ref()
                    .ok_or(GpuRigidSphereSolveError::Capacity)?,
                warm,
            )
        } else {
            scratch_owned = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera resident sphere accumulated impulses"),
                size: scratch_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            (&scratch_owned, false)
        };
        let solver_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere solver params"),
            contents: bytemuck::bytes_of(&SolverParams {
                values: [dt, params.friction, params.restitution, params.bias_factor],
                temporal: settings.temporal,
                static_temporal: settings.static_temporal,
                counts: [
                    contacts.island_count(),
                    pair_count,
                    u32::from(contacts.has_ground()),
                    params.iterations,
                ],
                flags: [
                    u32::from(warm),
                    contacts.ground_contact_stride(),
                    body_count,
                    contacts.pair_contact_stride(),
                ],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident sphere solver inputs"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: contacts.state_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contacts.pair_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: contacts.pair_contact_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: contacts.ground_buffer_raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: solver_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: contacts.island_range_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: contacts.island_index_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: contacts.material_buffer().as_entire_binding(),
                },
            ],
        });
        Ok(Some((bind_group, contacts.island_count())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_broad_phase::GpuPair;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use crate::gpu_lbvh::GpuLbvh;
    use crate::gpu_rigid_sphere_contact::GpuRigidSphereContacts;
    use crate::gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession};
    use crate::material::{CoefficientCombineRule, ColliderMaterial};
    use crate::sleep::SleepSettings;
    use core::time::Duration;
    use std::sync::mpsc;

    fn body(position: [f32; 3], velocity: [f32; 3]) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [position[0], position[1], position[2], 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [velocity[0], velocity[1], velocity[2], 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        }
    }

    #[test]
    fn lbvh_candidates_solve_pair_without_cpu_pair_readback() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN) else {
            return;
        };
        check_lbvh_candidates_solve_pair_without_cpu_pair_readback(&context);
    }

    #[test]
    fn dx12_lbvh_candidates_solve_pair_without_cpu_pair_readback() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::DX12) else {
            return;
        };
        check_lbvh_candidates_solve_pair_without_cpu_pair_readback(&context);
    }

    fn check_lbvh_candidates_solve_pair_without_cpu_pair_readback(context: &GpuContactDevice) {
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| match index {
                0 => body([0.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                1 => body([1.5, 0.0, 3.0], [-1.0, 0.0, 0.0]),
                _ => body([index as f32 * 10.0, 0.0, 3.0], [0.0; 3]),
            })
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0; 70], &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident LBVH candidate solve test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let mut cache = GpuRigidCandidateImpulseCache::default();
        let result = solver
            .encode_resident_candidates_cached(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
                &mut cache,
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        result.readback_status(device, queue).unwrap();
        let actual = state.readback(device, queue).unwrap();
        assert!((actual[0].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((actual[1].linear_velocity[0] - 1.0).abs() < 1e-4);
        assert_eq!(actual[2].linear_velocity[0], 0.0);
        let impulses = cache.readback(device, queue, &candidates).unwrap();
        assert_eq!(impulses.dt, Some(0.01));
        assert_eq!(impulses.contacts.len(), 1);
        let impulse = impulses.contacts[0];
        assert_eq!(
            (impulse.body_a, impulse.body_b, impulse.point_index),
            (Some(0), 1, 0)
        );
        let vector = impulse.impulse_on_body_b();
        assert!((vector[0] - 2.0).abs() < 1e-4);
        assert!(vector[1].abs() < 1e-4 && vector[2].abs() < 1e-4);
        cache.clear();
        assert_eq!(
            cache.readback(device, queue, &candidates).unwrap(),
            GpuRigidContactImpulseReadback::default()
        );
    }

    #[test]
    fn candidate_solve_capacity_leaves_state_unchanged() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| body([index as f32 * 1.9, 0.0, 3.0], [0.0; 3]))
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0; 70], &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident candidate capacity test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    iterations: 400,
                    ..Default::default()
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        assert!(matches!(
            result.readback_status(device, queue),
            Err(GpuRigidSphereSolveError::Capacity)
        ));
        let actual = state.readback(device, queue).unwrap();
        assert!(actual.iter().zip(&states).all(|(left, right)| {
            left.position_inverse_mass == right.position_inverse_mass
                && left.linear_velocity == right.linear_velocity
        }));
    }

    #[test]
    fn high_degree_candidate_island_uses_serial_fallback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut states = vec![body([0.0, 0.0, 3.0], [0.0; 3])];
        let mut radii = vec![10.0];
        for index in 0..34 {
            let angle = index as f32 * core::f32::consts::TAU / 34.0;
            let (sin, cos) = angle.sin_cos();
            states.push(body([10.05 * cos, 10.05 * sin, 3.0], [-cos, -sin, 0.0]));
            radii.push(0.1);
        }
        while states.len() < 70 {
            states.push(body(
                [1000.0 + states.len() as f32 * 10.0, 0.0, 3.0],
                [0.0; 3],
            ));
            radii.push(1.0);
        }
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &radii, &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera high degree island fallback test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 12,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        result.readback_status(device, queue).unwrap();
        assert_eq!(candidates.readback_count(device, queue).unwrap(), 34);

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera island color marker readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera island color marker copy"),
        });
        encoder.copy_buffer_to_buffer(&result.status, (1 + 6 * 70) * 4, &readback, 0, 4);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let view = readback.slice(..).get_mapped_range();
        assert_eq!(u32::from_le_bytes(view[..4].try_into().unwrap()), u32::MAX);
        drop(view);
        readback.unmap();

        let actual = state.readback(device, queue).unwrap();
        assert!(actual[1].linear_velocity[0] > states[1].linear_velocity[0] + 1e-3);
        assert!(
            actual
                .iter()
                .all(|state| state.linear_velocity[..3].iter().all(|v| v.is_finite()))
        );
    }

    #[test]
    fn candidate_islands_solve_more_total_work_than_one_serial_island() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| {
                let pair = index / 2;
                body(
                    [pair as f32 * 10.0 + (index % 2) as f32 * 1.5, 0.0, 3.0],
                    [if index % 2 == 0 { 1.0 } else { -1.0 }, 0.0, 0.0],
                )
            })
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0; 70], &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera independent candidate island test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 600,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        result.readback_status(device, queue).unwrap();
        assert_eq!(candidates.readback_count(device, queue).unwrap(), 35);
        let actual = state.readback(device, queue).unwrap();
        for pair in 0..35 {
            assert!((actual[pair * 2].linear_velocity[0] + 1.0).abs() < 1e-4);
            assert!((actual[pair * 2 + 1].linear_velocity[0] - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn candidate_islands_keep_independent_ground_impulses() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| {
                if index == 0 {
                    body([0.0, 0.0, 0.99], [0.0, 0.0, -1.0])
                } else {
                    body([index as f32 * 10.0, 0.0, 3.0], [0.0; 3])
                }
            })
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 70], &[], Some(100.0)).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera independent candidate ground test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 10,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        result.readback_status(device, queue).unwrap();
        let actual = state.readback(device, queue).unwrap();
        assert!((actual[0].linear_velocity[2] - 1.0).abs() < 1e-4);
        assert_eq!(actual[1].linear_velocity[2], 0.0);
    }

    #[test]
    fn compact_candidate_islands_match_explicit_shared_body_solver() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| match index {
                0 => body([0.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                1 => body([1.5, 0.0, 3.0], [0.0; 3]),
                2 => body([3.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
                3 => body([20.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                4 => body([21.5, 0.0, 3.0], [-1.0, 0.0, 0.0]),
                _ => body([100.0 + index as f32 * 10.0, 0.0, 3.0], [0.0; 3]),
            })
            .collect::<Vec<_>>();
        let resident = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let resident_contacts =
            GpuRigidSphereContacts::new(device, &resident, &[1.0; 70], &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut candidate_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera compact candidate reference build"),
            });
        let (candidates, output) = resident_contacts
            .encode_state_lbvh_contacts(device, &mut candidate_encoder, &lbvh, None)
            .unwrap();
        let _submission = queue.submit(Some(candidate_encoder.finish()));
        let pairs = candidates.readback(device, queue).unwrap();
        assert_eq!(pairs.len(), 3);

        let reference = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let explicit =
            GpuRigidSphereContacts::new(device, &reference, &[1.0; 70], &pairs, None).unwrap();
        let params = GpuRigidSphereSolveParams {
            friction: 0.0,
            restitution: 0.0,
            bias_factor: 0.2,
            iterations: 20,
        };
        let mut solve_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera compact candidate reference solve"),
        });
        explicit.encode(&mut solve_encoder);
        solver
            .encode(device, &mut solve_encoder, &explicit, 0.01, params)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut solve_encoder,
                &resident_contacts,
                &candidates,
                &output,
                0.01,
                params,
            )
            .unwrap();
        let _submission = queue.submit(Some(solve_encoder.finish()));
        result.readback_status(device, queue).unwrap();
        let actual = resident.readback(device, queue).unwrap();
        let expected = reference.readback(device, queue).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            for axis in 0..3 {
                assert!(
                    (actual.linear_velocity[axis] - expected.linear_velocity[axis]).abs() < 1e-4
                );
            }
        }
    }

    #[test]
    fn lbvh_candidates_solve_ground_contact() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| {
                if index == 0 {
                    body([0.0, 0.0, 0.99], [0.0, 0.0, -1.0])
                } else {
                    body([index as f32 * 10.0, 0.0, 3.0], [0.0; 3])
                }
            })
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 70], &[], Some(100.0)).unwrap();
        let lbvh = GpuLbvh::new(device);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident candidate ground solve test"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        let result = solver
            .encode_resident_candidates(
                device,
                &mut encoder,
                &contacts,
                &candidates,
                &output,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        result.readback_status(device, queue).unwrap();
        let actual = state.readback(device, queue).unwrap();
        assert!((actual[0].linear_velocity[2] - 1.0).abs() < 1e-4);
        assert_eq!(actual[1].linear_velocity[2], 0.0);
    }

    #[test]
    fn candidate_remap_preserves_persistent_pair_and_ground_impulses() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = [body([0.0, 0.0, 1.0], [0.0; 3]); 4];
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let mut contacts = GpuRigidSphereContacts::new(
            device,
            &state,
            &[1.0; 4],
            &[GpuPair { a: 0, b: 1 }, GpuPair { a: 2, b: 3 }],
            Some(10.0),
        )
        .unwrap();
        let mut cache = GpuRigidSphereImpulseCache::default();
        let slot_bytes = IMPULSE_SLOT_BYTES as u64;
        assert!(!cache.prepare(device, &contacts, 6 * slot_bytes, 0.01));
        let previous = cache.buffer.as_ref().unwrap();
        for (slot, marker) in [0x11, 0x22, 0x31, 0x32, 0x33, 0x34].into_iter().enumerate() {
            queue.write_buffer(
                previous,
                slot as u64 * slot_bytes,
                &[marker; IMPULSE_SLOT_BYTES],
            );
        }

        contacts
            .set_candidate_pairs(
                device,
                &[
                    GpuPair { a: 2, b: 3 },
                    GpuPair { a: 0, b: 2 },
                    GpuPair { a: 0, b: 3 },
                ],
            )
            .unwrap();
        let mut remap = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera remapped impulse test encoder"),
        });
        cache
            .remap_candidate_pairs(device, &mut remap, &contacts, 0.01)
            .unwrap();
        let _submission = queue.submit(Some(remap.finish()));
        assert!(cache.prepare(device, &contacts, 7 * slot_bytes, 0.01));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera remapped impulse test readback"),
            size: 7 * slot_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera remapped impulse test copy"),
        });
        encoder.copy_buffer_to_buffer(
            cache.buffer.as_ref().unwrap(),
            0,
            &staging,
            0,
            7 * slot_bytes,
        );
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let view = staging.slice(..).get_mapped_range();
        for (slot, marker) in [0x22, 0, 0, 0x31, 0x32, 0x33, 0x34].into_iter().enumerate() {
            assert!(
                view[slot * IMPULSE_SLOT_BYTES..(slot + 1) * IMPULSE_SLOT_BYTES]
                    .iter()
                    .all(|byte| *byte == marker)
            );
        }
        drop(view);
        staging.unmap();
    }

    #[test]
    fn elastic_pair_exchanges_velocity_without_cpu_contact_transfer() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
            ],
        )
        .unwrap();
        let contacts = GpuRigidSphereContacts::new(
            device,
            &state,
            &[1.0, 1.0],
            &[GpuPair { a: 0, b: 1 }],
            None,
        )
        .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident pair collision test"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let states = state.readback(device, queue).unwrap();
        assert!((states[0].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((states[1].linear_velocity[0] - 1.0).abs() < 1e-4);
        assert!((states[0].position_inverse_mass[0] + 0.99).abs() < 1e-5);
    }

    #[test]
    fn ground_friction_reduces_slip_and_creates_spin() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN) else {
            return;
        };
        check_ground_friction_reduces_slip_and_creates_spin(&context);
    }

    #[test]
    fn dx12_ground_friction_reduces_slip_and_creates_spin() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::DX12) else {
            return;
        };
        check_ground_friction_reduces_slip_and_creates_spin(&context);
    }

    fn check_ground_friction_reduces_slip_and_creates_spin(context: &GpuContactDevice) {
        let device = context.device();
        let queue = context.queue();
        let state =
            GpuRigidStateSession::new(device, queue, &[body([0.0, 0.0, 1.0], [1.0, 0.0, -1.0])])
                .unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0], &[], Some(2.0)).unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident ground friction test"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        let mut cache = GpuRigidSphereImpulseCache::default();
        solver
            .encode_cached(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 1.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
                &mut cache,
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap()[0];
        assert!(actual.linear_velocity[0] > 0.45 && actual.linear_velocity[0] < 0.55);
        assert!(actual.linear_velocity[2].abs() < 1e-5);
        assert!(actual.angular_velocity[1] > 0.45);
        let impulses = cache.readback(device, queue).unwrap();
        assert_eq!(impulses.contacts.len(), 1);
        let impulse = impulses.contacts[0];
        assert_eq!(
            (impulse.body_a, impulse.body_b, impulse.point_index),
            (None, 0, 0)
        );
        let vector = impulse.impulse_on_body_b();
        for (axis, initial) in [1.0, 0.0, -1.0].into_iter().enumerate() {
            assert!((vector[axis] - (actual.linear_velocity[axis] - initial)).abs() < 1e-4);
        }
        cache.invalidate_body(queue, 0);
        assert!(cache.readback(device, queue).unwrap().contacts.is_empty());
        cache.clear();
        assert_eq!(cache.readback(device, queue).unwrap().dt, None);
    }

    #[test]
    fn two_sphere_stack_stays_supported_without_intermediate_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 0.55], [0.0; 3]),
                body([0.0, 0.0, 1.55], [0.0; 3]),
            ],
        )
        .unwrap();
        let contacts = GpuRigidSphereContacts::new(
            device,
            &state,
            &[0.5, 0.5],
            &[GpuPair { a: 0, b: 1 }],
            Some(5.0),
        )
        .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident sphere stack test"),
        });
        for _ in 0..240 {
            state
                .encode_step(device, &mut encoder, 0.005, [0.0, 0.0, -9.81])
                .unwrap();
            contacts.encode(&mut encoder);
            solver
                .encode(
                    device,
                    &mut encoder,
                    &contacts,
                    0.005,
                    GpuRigidSphereSolveParams::default(),
                )
                .unwrap();
        }
        let _submission = queue.submit(Some(encoder.finish()));
        let bodies = state.readback(device, queue).unwrap();
        assert!((bodies[0].position_inverse_mass[2] - 0.5).abs() < 0.06);
        assert!((bodies[1].position_inverse_mass[2] - 1.5).abs() < 0.08);
        assert!(bodies.iter().all(|body| {
            body.position_inverse_mass[2].is_finite()
                && body.linear_velocity[2].is_finite()
                && body.linear_velocity[2].abs() < 0.2
        }));
    }

    #[test]
    fn independent_contact_islands_run_across_workgroups() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut initial = Vec::new();
        let mut pairs = Vec::new();
        for index in 0..65 {
            let center = index as f32 * 10.0;
            let first = initial.len() as u32;
            initial.push(body([center - 1.0, 0.0, 3.0], [1.0, 0.0, 0.0]));
            initial.push(body([center + 1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]));
            pairs.push(GpuPair {
                a: first,
                b: first + 1,
            });
        }
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &vec![1.0; 130], &pairs, None).unwrap();
        assert_eq!(contacts.island_count(), 65);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera independent resident sphere islands"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 1.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let result = state.readback(device, queue).unwrap();
        for pair in &pairs {
            assert!((result[pair.a as usize].linear_velocity[0] + 1.0).abs() < 1e-4);
            assert!((result[pair.b as usize].linear_velocity[0] - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn independent_ground_contacts_run_across_workgroups() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let initial = (0..130)
            .map(|index| body([index as f32 * 3.0, 0.0, 1.0], [0.0, 0.0, -1.0]))
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &vec![1.0; 130], &[], Some(500.0)).unwrap();
        assert_eq!(contacts.island_count(), 130);
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera independent resident ground islands"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let result = state.readback(device, queue).unwrap();
        assert!(
            result
                .iter()
                .all(|body| body.linear_velocity[2].abs() < 1e-4)
        );
    }

    #[test]
    fn gpu_ground_material_rules_match_cpu_combination() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 1.0], [1.0, 0.0, -1.0]),
                body([3.0, 0.0, 1.0], [1.0, 0.0, -1.0]),
            ],
        )
        .unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0, 1.0], &[], Some(10.0)).unwrap();
        let ground = ColliderMaterial {
            friction: 0.2,
            restitution: 0.8,
            friction_combine_rule: CoefficientCombineRule::Multiply,
            restitution_combine_rule: CoefficientCombineRule::Multiply,
        };
        let body_min = ColliderMaterial {
            friction: 0.1,
            restitution: 0.2,
            friction_combine_rule: CoefficientCombineRule::Min,
            restitution_combine_rule: CoefficientCombineRule::Min,
        };
        let body_max = ColliderMaterial {
            friction_combine_rule: CoefficientCombineRule::Max,
            restitution_combine_rule: CoefficientCombineRule::Max,
            ..body_min
        };
        contacts.set_ground_material(queue, ground).unwrap();
        contacts.set_body_material(queue, 0, body_min).unwrap();
        contacts.set_body_material(queue, 1, body_max).unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident ground material test"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        for (index, body_material) in [body_min, body_max].into_iter().enumerate() {
            let combined = ground.combine(body_material);
            let expected_x = 1.0 - combined.friction * (1.0 + combined.restitution);
            assert!((f64::from(actual[index].linear_velocity[0]) - expected_x).abs() < 1e-3);
            assert!(
                (f64::from(actual[index].linear_velocity[2]) - combined.restitution).abs() < 1e-3
            );
        }
    }

    #[test]
    fn pair_material_restitution_varies_by_collider() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
                body([9.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
                body([11.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
            ],
        )
        .unwrap();
        let contacts = GpuRigidSphereContacts::new(
            device,
            &state,
            &[1.0; 4],
            &[GpuPair { a: 0, b: 1 }, GpuPair { a: 2, b: 3 }],
            None,
        )
        .unwrap();
        contacts
            .set_body_material(queue, 0, ColliderMaterial::new(0.0, 1.0))
            .unwrap();
        contacts
            .set_body_material(queue, 1, ColliderMaterial::new(0.0, 1.0))
            .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident pair material test"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        assert!((actual[0].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((actual[1].linear_velocity[0] - 1.0).abs() < 1e-4);
        assert!(actual[2].linear_velocity[0].abs() < 1e-4);
        assert!(actual[3].linear_velocity[0].abs() < 1e-4);

        contacts.clear_body_material(queue, 0).unwrap();
        contacts.clear_body_material(queue, 1).unwrap();
        for (index, initial) in [
            body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
            body([9.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([11.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
        ]
        .into_iter()
        .enumerate()
        {
            state.write_body(queue, index, initial).unwrap();
        }
        let mut restored = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera restored default pair materials"),
        });
        state
            .encode_step(device, &mut restored, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut restored);
        solver
            .encode(
                device,
                &mut restored,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(restored.finish()));
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.linear_velocity[0].abs() < 1e-4)
        );
    }

    #[test]
    fn all_material_combine_rules_match_cpu_restitution() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let initial = (0..4)
            .map(|index| body([index as f32 * 3.0, 0.0, 1.0], [0.0, 0.0, -1.0]))
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 4], &[], Some(10.0)).unwrap();
        let ground = ColliderMaterial::new(0.0, 0.8);
        contacts.set_ground_material(queue, ground).unwrap();
        let rules = [
            CoefficientCombineRule::Average,
            CoefficientCombineRule::Min,
            CoefficientCombineRule::Multiply,
            CoefficientCombineRule::Max,
        ];
        for (index, rule) in rules.iter().copied().enumerate() {
            contacts
                .set_body_material(
                    queue,
                    index,
                    ColliderMaterial {
                        restitution_combine_rule: rule,
                        ..ColliderMaterial::new(0.0, 0.2)
                    },
                )
                .unwrap();
        }
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident material combine rules"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        for (index, rule) in rules.into_iter().enumerate() {
            let body_material = ColliderMaterial {
                restitution_combine_rule: rule,
                ..ColliderMaterial::new(0.0, 0.2)
            };
            let expected = ground.combine(body_material).restitution;
            assert!((f64::from(actual[index].linear_velocity[2]) - expected).abs() < 1e-4);
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_resident_material_contact_matches_expected_bounce() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; skipping resident material test");
            return;
        };
        eprintln!(
            "Tessera resident solver adapter: {:?}",
            adapter.get_info().backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let state =
            GpuRigidStateSession::new(&device, &queue, &[body([0.0, 0.0, 1.0], [0.0, 0.0, -1.0])])
                .unwrap();
        let contacts =
            GpuRigidSphereContacts::new(&device, &state, &[1.0], &[], Some(5.0)).unwrap();
        contacts
            .set_body_material(&queue, 0, ColliderMaterial::new(0.0, 0.5))
            .unwrap();
        contacts
            .set_ground_material(&queue, ColliderMaterial::new(0.0, 0.5))
            .unwrap();
        let solver = GpuRigidSphereSolver::new(&device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera DX12 resident material test"),
        });
        state
            .encode_step(&device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                &device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.0,
                    iterations: 4,
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(&device, &queue).unwrap()[0];
        assert!((actual.linear_velocity[2] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn changing_candidate_pairs_preserves_materials_and_solver_islands() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let initial = [
            body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
            body([9.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([11.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
        ];
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 4], &[], None).unwrap();
        for index in [2, 3] {
            contacts
                .set_body_material(queue, index, ColliderMaterial::new(0.0, 1.0))
                .unwrap();
        }
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 0, b: 1 }, GpuPair { a: 2, b: 3 }])
            .unwrap();
        assert_eq!(contacts.island_count(), 2);
        let solver = GpuRigidSphereSolver::new(device);
        let solve_params = GpuRigidSphereSolveParams {
            friction: 0.0,
            restitution: 0.0,
            bias_factor: 0.0,
            iterations: 4,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera updated pair island solve"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(device, &mut encoder, &contacts, 0.01, solve_params)
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        assert!(actual[0].linear_velocity[0].abs() < 1e-4);
        assert!(actual[1].linear_velocity[0].abs() < 1e-4);
        assert!((actual[2].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((actual[3].linear_velocity[0] - 1.0).abs() < 1e-4);

        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 2, b: 3 }])
            .unwrap();
        assert_eq!(contacts.island_count(), 1);
        for (index, body_state) in initial.into_iter().enumerate() {
            state.write_body(queue, index, body_state).unwrap();
        }
        let mut next = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera changed pair island solve"),
        });
        state
            .encode_step(device, &mut next, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut next);
        solver
            .encode(device, &mut next, &contacts, 0.01, solve_params)
            .unwrap();
        let _submission = queue.submit(Some(next.finish()));
        let actual = state.readback(device, queue).unwrap();
        assert!((actual[0].linear_velocity[0] - 1.0).abs() < 1e-4);
        assert!((actual[1].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((actual[2].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((actual[3].linear_velocity[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ground_impulses_allow_resident_body_to_sleep_under_gravity() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state =
            GpuRigidStateSession::new(device, queue, &[body([0.0, 0.0, 1.0], [0.0; 3])]).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0], &[], Some(5.0)).unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let settings = SleepSettings {
            time_threshold: 0.05,
            ..Default::default()
        };
        for _ in 0..120 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident resting body substep"),
            });
            state
                .encode_step(device, &mut encoder, 0.01, [0.0, 0.0, -9.81])
                .unwrap();
            contacts.encode(&mut encoder);
            solver
                .encode(
                    device,
                    &mut encoder,
                    &contacts,
                    0.01,
                    GpuRigidSphereSolveParams::default(),
                )
                .unwrap();
            contacts
                .encode_sleep(device, &mut encoder, 0.01, settings)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
        }
        let actual = state.readback(device, queue).unwrap()[0];
        assert_eq!(actual.inverse_inertia_sleep[3], 1.0);
        assert!(actual.linear_velocity[2].abs() < 1e-4);
        assert!((actual.position_inverse_mass[2] - 1.0).abs() < 0.005);
    }

    #[test]
    fn moving_sphere_impulse_wakes_sleeping_resident_body() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut sleeping = body([-1.0, 0.0, 3.0], [0.0; 3]);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let moving = body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]);
        let state = GpuRigidStateSession::new(device, queue, &[sleeping, moving]).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 2], &[GpuPair { a: 0, b: 1 }], None)
                .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera sleeping resident collision"),
        });
        state
            .encode_step(device, &mut encoder, 0.01, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams::default(),
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        assert_eq!(actual[0].inverse_inertia_sleep[3], 0.0);
        assert!(actual[0].linear_velocity[0] < -0.1);
    }

    #[test]
    fn static_sphere_support_does_not_wake_sleeping_body_from_bias() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut fixed = body([0.0, 0.0, 0.0], [0.0; 3]);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut sleeping = body([0.0, 0.0, 1.95], [0.0; 3]);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let state = GpuRigidStateSession::new(device, queue, &[fixed, sleeping]).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 2], &[GpuPair { a: 0, b: 1 }], None)
                .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera static support sleeping body"),
        });
        contacts.encode(&mut encoder);
        solver
            .encode(
                device,
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams::default(),
            )
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = state.readback(device, queue).unwrap();
        assert_eq!(actual[1].inverse_inertia_sleep[3], 1.0);
        assert_eq!(actual[1].linear_velocity[2], 0.0);
    }

    #[test]
    fn cached_impulses_reduce_one_iteration_stack_error() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let simulate = |cached: bool| {
            let state = GpuRigidStateSession::new(
                device,
                queue,
                &[
                    body([0.0, 0.0, 1.0], [0.0; 3]),
                    body([0.0, 0.0, 3.0], [0.0; 3]),
                ],
            )
            .unwrap();
            let contacts = GpuRigidSphereContacts::new(
                device,
                &state,
                &[1.0; 2],
                &[GpuPair { a: 0, b: 1 }],
                Some(5.0),
            )
            .unwrap();
            let solver = GpuRigidSphereSolver::new(device);
            let mut cache = GpuRigidSphereImpulseCache::default();
            let params = GpuRigidSphereSolveParams {
                iterations: 1,
                ..Default::default()
            };
            for _ in 0..20 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera one-iteration stack"),
                });
                state
                    .encode_step(device, &mut encoder, 0.01, [0.0, 0.0, -9.81])
                    .unwrap();
                contacts.encode(&mut encoder);
                if cached {
                    solver
                        .encode_cached(device, &mut encoder, &contacts, 0.01, params, &mut cache)
                        .unwrap();
                } else {
                    solver
                        .encode(device, &mut encoder, &contacts, 0.01, params)
                        .unwrap();
                }
                let _submission = queue.submit(Some(encoder.finish()));
            }
            state.readback(device, queue).unwrap()
        };
        let warm = simulate(true);
        let cold = simulate(false);
        let warm_error = (warm[0].position_inverse_mass[2] - 1.0).abs()
            + (warm[1].position_inverse_mass[2] - 3.0).abs();
        let cold_error = (cold[0].position_inverse_mass[2] - 1.0).abs()
            + (cold[1].position_inverse_mass[2] - 3.0).abs();
        assert!(warm_error < cold_error);
    }

    #[test]
    fn warm_start_rejects_impulse_from_different_local_contact_point() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let solve = |stale_a: bool, stale_b: bool| {
            let mut static_body = body([0.0, 0.0, 3.0], [0.0; 3]);
            static_body.position_inverse_mass[3] = 0.0;
            static_body.inverse_inertia_sleep = [0.0; 4];
            let state = GpuRigidStateSession::new(
                device,
                queue,
                &[static_body, body([1.5, 0.0, 3.0], [0.0; 3])],
            )
            .unwrap();
            let contacts = GpuRigidSphereContacts::new(
                device,
                &state,
                &[1.0; 2],
                &[GpuPair { a: 0, b: 1 }],
                None,
            )
            .unwrap();
            let solver = GpuRigidSphereSolver::new(device);
            let mut cache = GpuRigidSphereImpulseCache::default();
            let scratch_size =
                u64::from(contacts.pair_contact_stride()) * IMPULSE_SLOT_BYTES as u64;
            assert!(!cache.prepare(device, &contacts, scratch_size, 0.01));
            let anchor_a = if stale_a { 100.0 } else { 0.75 };
            let anchor_b = if stale_b { 100.0 } else { -0.75 };
            let impulse: [[f32; 4]; 4] = [
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [anchor_a, 0.0, 0.0, 0.0],
                [anchor_b, 0.0, 0.0, 0.0],
            ];
            queue.write_buffer(
                cache.buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&impulse),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident contact anchor cache test"),
            });
            contacts.encode(&mut encoder);
            let step = solver
                .prepare_coupled_cached(
                    device,
                    &contacts,
                    0.01,
                    GpuRigidSphereSolveParams {
                        friction: 0.0,
                        restitution: 0.0,
                        bias_factor: 0.0,
                        iterations: 1,
                    },
                    &mut cache,
                )
                .unwrap()
                .unwrap();
            step.encode_warm(&mut encoder);
            let _submission = queue.submit(Some(encoder.finish()));
            state.readback(device, queue).unwrap()[1].linear_velocity[1]
        };
        let current = solve(false, false);
        assert!(current > 0.9, "cached velocity: {current}");
        assert!(solve(true, false).abs() < 1e-5);
        assert!(solve(false, true).abs() < 1e-5);
    }

    #[test]
    fn changed_pair_topology_discards_cached_impulses() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let initial = [
            body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
            body([8.0, 0.0, 3.0], [0.0; 3]),
        ];
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 3], &[GpuPair { a: 0, b: 1 }], None)
                .unwrap();
        let solver = GpuRigidSphereSolver::new(device);
        let mut cache = GpuRigidSphereImpulseCache::default();
        let params = GpuRigidSphereSolveParams::default();
        let mut first = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera cache first pair"),
        });
        contacts.encode(&mut first);
        solver
            .encode_cached(device, &mut first, &contacts, 0.01, params, &mut cache)
            .unwrap();
        let _submission = queue.submit(Some(first.finish()));

        let changed = [
            body([-8.0, 0.0, 3.0], [0.0; 3]),
            body([-1.0, 0.0, 3.0], [1.0, 0.0, 0.0]),
            body([1.0, 0.0, 3.0], [-1.0, 0.0, 0.0]),
        ];
        for (index, body) in changed.iter().copied().enumerate() {
            state.write_body(queue, index, body).unwrap();
        }
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 1, b: 2 }])
            .unwrap();
        let fresh_state = GpuRigidStateSession::new(device, queue, &changed).unwrap();
        let fresh_contacts = GpuRigidSphereContacts::new(
            device,
            &fresh_state,
            &[1.0; 3],
            &[GpuPair { a: 1, b: 2 }],
            None,
        )
        .unwrap();
        let mut second = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera changed pair cache invalidation"),
        });
        contacts.encode(&mut second);
        solver
            .encode_cached(device, &mut second, &contacts, 0.01, params, &mut cache)
            .unwrap();
        fresh_contacts.encode(&mut second);
        solver
            .encode(device, &mut second, &fresh_contacts, 0.01, params)
            .unwrap();
        let _submission = queue.submit(Some(second.finish()));
        let actual = state.readback(device, queue).unwrap();
        let expected = fresh_state.readback(device, queue).unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left.linear_velocity[0] - right.linear_velocity[0]).abs() < 1e-5);
        }

        for (index, body) in changed.iter().copied().enumerate() {
            state.write_body(queue, index, body).unwrap();
            fresh_state.write_body(queue, index, body).unwrap();
        }
        let mut third = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera repeated impact cache invalidation"),
        });
        contacts.encode(&mut third);
        solver
            .encode_cached(device, &mut third, &contacts, 0.01, params, &mut cache)
            .unwrap();
        fresh_contacts.encode(&mut third);
        solver
            .encode(device, &mut third, &fresh_contacts, 0.01, params)
            .unwrap();
        let _submission = queue.submit(Some(third.finish()));
        let actual = state.readback(device, queue).unwrap();
        let expected = fresh_state.readback(device, queue).unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left.linear_velocity[0] - right.linear_velocity[0]).abs() < 1e-5);
        }
    }
}
