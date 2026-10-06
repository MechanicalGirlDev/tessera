//! Three-dimensional Material Point Method simulation for Tessera.
//!
//! A deterministic `f64` CPU reference provides the oracle for the optional
//! WebGPU P2G, hybrid grid-update, and G2P compute stages. Material stress runs in
//! P2G for regular deformations; near-singular corotated states fall back to
//! CPU stress. G2P updates deformation, projects fluids, sand, and snow, and
//! integrates particles against bounds and rigid obstacles. Ill-conditioned
//! plastic states fall back to the CPU.
//! Bounded worlds can run fixed substeps in one GPU submission with particle
//! state retained between substeps. A CPU material fallback or grid escape
//! aborts that batch without applying its result to the world.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![allow(clippy::std_instead_of_alloc)]

pub mod emitter;
#[cfg(feature = "gpu-mpm")]
pub mod gpu;
#[cfg(feature = "gpu-rigid-coupling")]
pub mod gpu_rigid_sync;
#[cfg(feature = "gpu-mpm")]
mod gpu_topology;
pub mod material;
pub mod obstacle;
#[cfg(feature = "rigid-coupling")]
pub mod rigid_sync;
pub mod sampling;
pub mod world;

pub use emitter::{BoxEmitter, MeshEmitter, MeshEmitterError};
#[cfg(feature = "gpu-mpm")]
pub use gpu::{GpuMpmError, GpuMpmParticleSnapshot, GpuMpmResidentSession, GpuMpmTransfers};
#[cfg(feature = "gpu-rigid-coupling")]
pub use gpu_rigid_sync::{GpuRigidMpmCoupler, GpuRigidMpmCouplerError};
pub use material::{MaterialModel, PlasticState};
pub use obstacle::{ObstacleBoundary, ObstacleShape, RigidObstacle};
#[cfg(all(feature = "rigid-coupling", feature = "gpu-mpm"))]
pub use rigid_sync::ResidentRigidSyncError;
#[cfg(feature = "gpu-rigid-coupling")]
pub use rigid_sync::gpu_rigid_world_obstacles;
#[cfg(feature = "rigid-coupling")]
pub use rigid_sync::{RigidSyncError, articulated_world_obstacles, sphere_world_obstacles};
pub use sampling::{
    SurfaceSamplingError, TriangleSurfaceSample, VolumeSamplingError, sample_closed_mesh_volume,
    sample_triangle_mesh,
};
pub use world::{
    MpmError, MpmParams, MpmParticle, MpmWorld, ObstacleReaction, ParticleChunkId, WorldBounds,
};
