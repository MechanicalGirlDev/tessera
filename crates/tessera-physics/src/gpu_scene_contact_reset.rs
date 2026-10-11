//! Resident contact geometry and prescribed-motion templates for selected scene resets.
use super::{GpuArticulatedGroundContactBatch, PackedSphere, PackedSphereOrbit};
use crate::gpu_articulated_dynamics::resident_reset::snapshot_buffer;

#[derive(Debug)]
pub(crate) struct SceneContactResetTemplates {
    source: wgpu::Buffer,
    spheres: wgpu::Buffer,
    orbits: wgpu::Buffer,
    static_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_sphere_motion: Vec<Vec<[f32; 6]>>,
    static_capsule_sphere_motion: Vec<Vec<[f32; 6]>>,
    static_box_sphere_motion: Vec<Vec<[f32; 6]>>,
    static_axial_sphere_motion: Vec<Vec<[f32; 6]>>,
    static_convex_sphere_motion: Vec<Vec<[f32; 6]>>,
    static_capsule_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_box_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_axial_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_convex_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_sphere_box_poses: Vec<Vec<[f32; 7]>>,
    static_capsule_box_poses: Vec<Vec<[f32; 7]>>,
    static_box_box_poses: Vec<Vec<[f32; 7]>>,
    scene_convex_rounded_poses: Vec<Vec<[f32; 7]>>,
    static_axial_convex_poses: Vec<Vec<[f32; 7]>>,
    static_convex_pair_poses: Vec<Vec<[f32; 7]>>,
    static_axial_box_poses: Vec<Vec<[f32; 7]>>,
}

impl GpuArticulatedGroundContactBatch {
    pub(crate) fn snapshot_scene_contacts(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SceneContactResetTemplates {
        SceneContactResetTemplates {
            source: self.spheres.clone(),
            spheres: snapshot_buffer(&self.device, encoder, &self.spheres),
            orbits: snapshot_buffer(&self.device, encoder, &self.sphere_orbits),
            static_sphere_centers: self.static_sphere_centers.clone(),
            static_sphere_motion: self.static_sphere_motion.clone(),
            static_capsule_sphere_motion: self.static_capsule_sphere_motion.clone(),
            static_box_sphere_motion: self.static_box_sphere_motion.clone(),
            static_axial_sphere_motion: self.static_axial_sphere_motion.clone(),
            static_convex_sphere_motion: self.static_convex_sphere_motion.clone(),
            static_capsule_sphere_centers: self.static_capsule_sphere_centers.clone(),
            static_box_sphere_centers: self.static_box_sphere_centers.clone(),
            static_axial_sphere_centers: self.static_axial_sphere_centers.clone(),
            static_convex_sphere_centers: self.static_convex_sphere_centers.clone(),
            static_sphere_box_poses: self.static_sphere_box_poses.clone(),
            static_capsule_box_poses: self.static_capsule_box_poses.clone(),
            static_box_box_poses: self.static_box_box_poses.clone(),
            scene_convex_rounded_poses: self.scene_convex_rounded_poses.clone(),
            static_axial_convex_poses: self.static_axial_convex_poses.clone(),
            static_convex_pair_poses: self.static_convex_pair_poses.clone(),
            static_axial_box_poses: self.static_axial_box_poses.clone(),
        }
    }

    pub(crate) fn matches_scene_contacts(&self, templates: &SceneContactResetTemplates) -> bool {
        self.spheres == templates.source
    }

    pub(crate) fn encode_reset_scene_contacts(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        templates: &SceneContactResetTemplates,
        environments: &[usize],
    ) {
        for &environment in environments {
            let rows = &self.contact_ranges[environment];
            if !rows.is_empty() {
                for (source, destination, stride) in [
                    (
                        &templates.spheres,
                        &self.spheres,
                        size_of::<PackedSphere>() as u64,
                    ),
                    (
                        &templates.orbits,
                        &self.sphere_orbits,
                        size_of::<PackedSphereOrbit>() as u64,
                    ),
                ] {
                    encoder.copy_buffer_to_buffer(
                        source,
                        rows.start as u64 * stride,
                        destination,
                        rows.start as u64 * stride,
                        rows.len() as u64 * stride,
                    );
                }
            }
            self.static_sphere_centers[environment]
                .clone_from(&templates.static_sphere_centers[environment]);
            self.static_sphere_motion[environment]
                .clone_from(&templates.static_sphere_motion[environment]);
            self.static_capsule_sphere_motion[environment]
                .clone_from(&templates.static_capsule_sphere_motion[environment]);
            self.static_box_sphere_motion[environment]
                .clone_from(&templates.static_box_sphere_motion[environment]);
            self.static_axial_sphere_motion[environment]
                .clone_from(&templates.static_axial_sphere_motion[environment]);
            self.static_convex_sphere_motion[environment]
                .clone_from(&templates.static_convex_sphere_motion[environment]);
            self.static_capsule_sphere_centers[environment]
                .clone_from(&templates.static_capsule_sphere_centers[environment]);
            self.static_box_sphere_centers[environment]
                .clone_from(&templates.static_box_sphere_centers[environment]);
            self.static_axial_sphere_centers[environment]
                .clone_from(&templates.static_axial_sphere_centers[environment]);
            self.static_convex_sphere_centers[environment]
                .clone_from(&templates.static_convex_sphere_centers[environment]);
            self.static_sphere_box_poses[environment]
                .clone_from(&templates.static_sphere_box_poses[environment]);
            self.static_capsule_box_poses[environment]
                .clone_from(&templates.static_capsule_box_poses[environment]);
            self.static_box_box_poses[environment]
                .clone_from(&templates.static_box_box_poses[environment]);
            self.scene_convex_rounded_poses[environment]
                .clone_from(&templates.scene_convex_rounded_poses[environment]);
            self.static_axial_convex_poses[environment]
                .clone_from(&templates.static_axial_convex_poses[environment]);
            self.static_convex_pair_poses[environment]
                .clone_from(&templates.static_convex_pair_poses[environment]);
            self.static_axial_box_poses[environment]
                .clone_from(&templates.static_axial_box_poses[environment]);
            // Reset contact history and component sleep diagnostics, not other environments.
            self.encode_clear_environment(encoder, environment);
        }
    }
}
