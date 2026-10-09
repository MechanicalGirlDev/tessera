//! A device snapshot must preserve numerical failure, not hide it.

use super::*;

#[test]
fn selective_reset_preserves_captured_faults_without_hiding_failed_templates() {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Some(fixture::Fixture {
            context,
            state,
            poses,
            spherical,
            ..
        }) = fixture::build(backend)
        else {
            continue;
        };
        tested += 1;
        let selection = GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: None,
            root_velocity: None,
        };
        // A snapshot of a fault is not a way to conceal that fault.
        context
            .queue()
            .write_buffer(state.status_buffer(), 0, bytemuck::bytes_of(&1u32));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let faulted = state
            .snapshot_reset_templates(&mut encoder, &poses, Some(&spherical))
            .unwrap();
        let _submission = context.queue().submit(Some(encoder.finish()));
        context
            .queue()
            .write_buffer(state.status_buffer(), 0, bytemuck::bytes_of(&0u32));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        state
            .encode_reset_envs_from_templates(&mut encoder, &faulted, &[selection])
            .unwrap();
        let _submission = context.queue().submit(Some(encoder.finish()));
        assert!(matches!(
            state.readback(),
            Err(GpuGeneralizedStateError::NonFinite(1))
        ));
        assert_eq!(
            read_buffer(context.device(), context.queue(), state.status_buffer()).unwrap(),
            bytemuck::cast_slice(&[0u32, 1u32]),
        );
    }
    assert!(tested > 0, "no native GPU backend tested");
}
