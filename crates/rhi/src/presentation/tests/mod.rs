//! Regression tests for the presentation façade's local lifecycle decisions.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::{may_best_effort_shutdown, quarantine_after_teardown_failure};

#[test]
fn drop_teardown_requires_configured_surface_without_live_frame() {
    assert!(may_best_effort_shutdown(true, false));
    assert!(!may_best_effort_shutdown(false, false));
    // An acquired or accepted-unknown token retains the surface and window
    // lease, so Drop must not attempt a competing unconfigure operation.
    assert!(!may_best_effort_shutdown(true, true));
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[test]
fn failed_best_effort_teardown_quarantines_ownership() {
    let leaked_flag = Arc::new(AtomicBool::new(false));
    assert!(quarantine_after_teardown_failure::<_, ()>(
        Err(()),
        DropFlag(Arc::clone(&leaked_flag)),
    ));
    assert!(
        !leaked_flag.load(Ordering::Acquire),
        "an unknown native teardown must retain ownership instead of dropping it"
    );

    let released_flag = Arc::new(AtomicBool::new(false));
    assert!(!quarantine_after_teardown_failure::<_, ()>(
        Ok(()),
        DropFlag(Arc::clone(&released_flag)),
    ));
    assert!(released_flag.load(Ordering::Acquire));
}
