//! Contract tests for bounded admission and completion-driven slot reuse.

use std::collections::VecDeque;

use super::{BackPressure, FrameRing, FrameWork, StartError, WorkStatus};

struct ControlledWork {
    observations: VecDeque<WorkStatus<&'static str>>,
}

impl ControlledWork {
    fn new(observations: impl IntoIterator<Item = WorkStatus<&'static str>>) -> Self {
        Self {
            observations: observations.into_iter().collect(),
        }
    }
}

impl FrameWork for ControlledWork {
    type Error = &'static str;

    fn poll_once(&mut self) -> WorkStatus<Self::Error> {
        self.observations.pop_front().unwrap_or(WorkStatus::Pending)
    }
}

#[test]
fn nth_plus_one_is_constrained_before_work_starts() {
    let mut ring = FrameRing::new(3).unwrap();
    for _ in 0..3 {
        let reservation = ring.try_reserve().unwrap();
        ring.commit(reservation, ControlledWork::new([WorkStatus::Pending]));
    }

    assert_eq!(ring.try_reserve(), Err(BackPressure));
    assert_eq!(ring.live_count(), 3);
    assert_eq!(ring.high_watermark(), 3);
    assert_eq!(ring.back_pressure_count(), 1);
}

#[test]
fn production_admission_helper_never_calls_start_at_capacity() {
    let mut ring = FrameRing::new(3).unwrap();
    let mut starts = 0;
    for _ in 0..3 {
        ring.try_start(|_| {
            starts += 1;
            Ok::<_, ()>(ControlledWork::new([WorkStatus::Pending]))
        })
        .unwrap();
    }

    let blocked = ring.try_start(|_| {
        starts += 1;
        Ok::<_, ()>(ControlledWork::new([WorkStatus::Pending]))
    });
    assert!(matches!(blocked, Err(StartError::BackPressure)));
    assert_eq!(
        starts, 3,
        "the fourth acquire/record callback must stay inert"
    );

    let mut replacement_starts = 0;
    let second = ring.poll_once().unwrap();
    assert!(second.retired().is_empty());
    assert!(matches!(
        ring.try_start(|_| {
            replacement_starts += 1;
            Ok::<_, ()>(ControlledWork::new([WorkStatus::Pending]))
        }),
        Err(StartError::BackPressure)
    ));
    assert_eq!(replacement_starts, 0);
}

#[test]
fn production_admission_helper_cancels_pre_accept_failure() {
    let mut ring: FrameRing<ControlledWork> = FrameRing::new(1).unwrap();
    let failed = ring.try_start(|_| Err::<ControlledWork, _>("pre-accept"));
    assert!(matches!(failed, Err(StartError::Start("pre-accept"))));
    assert!(ring.is_empty());

    let replacement = ring
        .try_start(|_| Ok::<_, &str>(ControlledWork::new([WorkStatus::Pending])))
        .unwrap();
    assert_eq!(replacement.slot(), 0);
}

#[test]
fn only_the_completed_slot_is_reused() {
    let mut ring = FrameRing::new(3).unwrap();
    let first = ring.try_reserve().unwrap();
    ring.commit(first, ControlledWork::new([WorkStatus::Pending]));
    let second = ring.try_reserve().unwrap();
    ring.commit(second, ControlledWork::new([WorkStatus::Complete]));
    let third = ring.try_reserve().unwrap();
    ring.commit(third, ControlledWork::new([WorkStatus::Pending]));

    assert_eq!(ring.try_reserve(), Err(BackPressure));
    let report = ring.poll_once().unwrap();
    assert_eq!(report.retired().len(), 1);
    assert_eq!(report.retired()[0].slot, second.slot());
    assert_eq!(report.retired()[0].serial, second.serial());
    assert_eq!(report.pending(), 2);

    let replacement = ring.try_reserve().unwrap();
    assert_eq!(replacement.slot(), second.slot());
    assert_ne!(replacement.serial(), second.serial());
    assert_eq!(ring.live_count(), 3);
}

#[test]
fn submission_milestone_advances_only_the_selected_new_slot() {
    let mut ring = FrameRing::new(3).unwrap();
    let first = ring.try_reserve().unwrap();
    ring.commit(
        first,
        ControlledWork::new([WorkStatus::Pending, WorkStatus::Submitted]),
    );
    let second = ring.try_reserve().unwrap();
    ring.commit(second, ControlledWork::new([WorkStatus::Submitted]));

    let pending = ring.poll_reserved_once(first).unwrap();
    assert_eq!(pending.pending(), 1);
    assert!(pending.submitted().is_empty());

    let submitted = ring.poll_reserved_once(first).unwrap();
    assert_eq!(submitted.submitted().len(), 1);
    assert_eq!(submitted.submitted()[0].serial, first.serial());

    let all = ring.poll_once().unwrap();
    assert_eq!(all.submitted().len(), 1);
    assert_eq!(all.submitted()[0].serial, second.serial());
    assert_eq!(ring.live_count(), 2);
}

#[test]
fn proven_pre_accept_failure_returns_only_its_reservation() {
    let mut ring: FrameRing<ControlledWork> = FrameRing::new(3).unwrap();
    let reservation = ring.try_reserve().unwrap();
    ring.cancel(reservation);

    let replacement = ring.try_reserve().unwrap();
    assert_ne!(replacement.serial(), reservation.serial());
    assert_eq!(ring.live_count(), 1);
}

#[test]
fn pending_timeout_never_reuses_the_slot() {
    let mut ring = FrameRing::new(1).unwrap();
    let reservation = ring.try_reserve().unwrap();
    ring.commit(
        reservation,
        ControlledWork::new([WorkStatus::Pending, WorkStatus::Pending]),
    );

    assert!(ring.poll_once().unwrap().retired().is_empty());
    assert_eq!(ring.try_reserve(), Err(BackPressure));
    assert!(ring.poll_once().unwrap().retired().is_empty());
    assert_eq!(ring.try_reserve(), Err(BackPressure));
}

#[test]
fn terminal_error_stops_observation_without_reusing_any_slot() {
    let mut ring = FrameRing::new(1).unwrap();
    let reservation = ring.try_reserve().unwrap();
    ring.commit(
        reservation,
        ControlledWork::new([WorkStatus::Failed("lost")]),
    );

    assert!(matches!(ring.poll_once(), Err("lost")));
    assert_eq!(ring.live_count(), 1);
    assert_eq!(ring.try_reserve(), Err(BackPressure));
}
