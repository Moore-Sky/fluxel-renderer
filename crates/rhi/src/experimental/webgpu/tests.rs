//! Unit contracts for the closed browser executor types.

use super::*;

#[test]
fn formats_are_closed() {
    assert_eq!(
        WebGpuCanvasFormat::parse("rgba8unorm"),
        Some(WebGpuCanvasFormat::Rgba8Unorm)
    );
    assert!(WebGpuCanvasFormat::parse("rgba8unorm-srgb").is_none());
}

#[test]
fn lifecycle_has_distinct_loss_and_dispose() {
    assert_ne!(WebGpuSessionState::Lost, WebGpuSessionState::Disposed);
    assert_ne!(MAX_FRAMES_IN_FLIGHT, 0);
}
