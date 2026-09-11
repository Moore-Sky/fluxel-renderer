//! Regression tests for the presentation façade's local lifecycle decisions.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::{
    ShutdownTransition, SurfaceExtent, SurfaceGeneration, SurfaceStatus, SurfaceTransition,
    classify_shutdown, classify_transition, may_best_effort_shutdown,
    quarantine_after_teardown_failure, quarantine_live_drop_ownership,
};

#[test]
fn drop_teardown_requires_configured_surface_without_live_frame() {
    assert!(may_best_effort_shutdown(true, false));
    assert!(!may_best_effort_shutdown(false, false));
    // An acquired or accepted-unknown token retains the surface and window
    // lease, so Drop must not attempt a competing unconfigure operation.
    assert!(!may_best_effort_shutdown(true, true));
}

#[test]
fn generation_transition_stops_active_acquire_before_zero_or_reconfigure() {
    let active = SurfaceStatus::Active {
        generation: SurfaceGeneration(7),
        extent: SurfaceExtent::new(640, 480),
    };
    assert_eq!(
        classify_transition(active, SurfaceExtent::new(640, 480)),
        SurfaceTransition::Noop
    );
    assert_eq!(
        classify_transition(active, SurfaceExtent::new(0, 480)),
        SurfaceTransition::RetireThenSuspend
    );
    assert_eq!(
        classify_transition(active, SurfaceExtent::new(800, 600)),
        SurfaceTransition::RetireThenConfigure
    );
    assert_eq!(
        classify_transition(SurfaceStatus::Suspended, SurfaceExtent::new(0, 0)),
        SurfaceTransition::Suspend
    );
}

#[test]
fn poisoned_generation_refuses_every_transition() {
    assert_eq!(
        classify_transition(SurfaceStatus::Poisoned, SurfaceExtent::new(800, 600)),
        SurfaceTransition::RefusePoisoned
    );
}

#[test]
fn poisoned_and_closed_states_cannot_be_reconfigured_or_cleanly_shutdown() {
    assert_eq!(
        classify_transition(SurfaceStatus::Poisoned, SurfaceExtent::new(800, 600)),
        SurfaceTransition::RefusePoisoned
    );
    assert_eq!(
        classify_transition(SurfaceStatus::Closed, SurfaceExtent::new(800, 600)),
        SurfaceTransition::RefuseClosed
    );
    assert_eq!(
        classify_shutdown(SurfaceStatus::Poisoned, false),
        ShutdownTransition::RefusePoisoned
    );
    assert_eq!(
        classify_shutdown(SurfaceStatus::Closed, true),
        ShutdownTransition::RefuseClosed
    );
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

#[test]
fn live_frame_drop_quarantines_surface_ownership() {
    let retained_flag = Arc::new(AtomicBool::new(false));
    assert!(quarantine_live_drop_ownership(
        true,
        DropFlag(Arc::clone(&retained_flag)),
    ));
    assert!(
        !retained_flag.load(Ordering::Acquire),
        "a live completion gate must retain native/window ownership on surface drop"
    );

    let released_flag = Arc::new(AtomicBool::new(false));
    assert!(!quarantine_live_drop_ownership(
        false,
        DropFlag(Arc::clone(&released_flag)),
    ));
    assert!(released_flag.load(Ordering::Acquire));
}

#[test]
fn multiple_live_tickets_keep_drop_quarantine_until_every_ticket_retires() {
    let tickets = crate::imp::NativePresentationTickets::new(2);
    let first = tickets.try_acquire().unwrap();
    let second = tickets.try_acquire().unwrap();
    assert!(tickets.any_live());
    drop(first);
    assert!(
        tickets.any_live(),
        "one retired frame must not release another"
    );

    let retained_flag = Arc::new(AtomicBool::new(false));
    assert!(quarantine_live_drop_ownership(
        tickets.any_live(),
        DropFlag(Arc::clone(&retained_flag)),
    ));
    assert!(!retained_flag.load(Ordering::Acquire));

    drop(second);
    assert!(!tickets.any_live());
}

#[test]
fn presentation_ticket_capacity_refuses_aliasing_and_recovers_one_exact_slot() {
    let capacity = crate::imp::DX12_PRESENTABLE_IMAGE_COUNT;
    assert_eq!(
        capacity,
        crate::imp::DX12_MAXIMUM_FRAME_LATENCY as usize + 1
    );
    let tickets = crate::imp::NativePresentationTickets::new(capacity);
    assert_eq!(tickets.capacity(), 3);
    let first = tickets.try_acquire().unwrap();
    let second = tickets.try_acquire().unwrap();
    let third = tickets.try_acquire().unwrap();
    assert!(tickets.try_acquire().is_none());

    drop(second);
    let replacement = tickets.try_acquire().unwrap();
    assert!(tickets.try_acquire().is_none());
    assert_eq!(tickets.live_count(), 3);

    drop((first, third, replacement));
    assert_eq!(tickets.live_count(), 0);
}
