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

#[test]
fn recovery_publication_requires_its_original_token_and_state() {
    let mut shared = Shared {
        state: WebGpuSessionState::Recovering,
        generation: 7,
        token: 11,
        ..Shared::default()
    };
    assert!(recovery_attempt_current(shared.state, shared.token, 11));

    // `dispose` invalidates the token before it waits for the in-flight
    // recovery. The candidate must therefore never publish its facts.
    shared.token += 1;
    shared.state = WebGpuSessionState::Disposing;
    assert!(!recovery_attempt_current(shared.state, shared.token, 11));
    assert_eq!(shared.generation, 7);
}

#[test]
fn recovery_candidate_is_not_committable_from_a_terminal_state() {
    let shared = Shared {
        state: WebGpuSessionState::Disposed,
        token: 3,
        ..Shared::default()
    };
    assert!(!recovery_attempt_current(shared.state, shared.token, 3));
}
