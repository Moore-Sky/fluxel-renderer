//! Tests the type-level local versus sendable execution modes.

use super::{FrameExecution, SendMode};

fn assert_send<T: Send>() {}

#[test]
fn send_mode_execution_is_send_for_send_frame_data() {
    assert_send::<FrameExecution<(), SendMode>>();
}
