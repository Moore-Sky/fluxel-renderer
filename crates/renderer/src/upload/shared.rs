//! Shared upload-generation synchronization and retained-state contracts.

use std::{
    sync::atomic::AtomicU64,
    sync::{Arc, Mutex},
};

pub(super) static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Why a ready snapshot cannot start another renderer draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SnapshotUseError {
    /// An accepted draw for this generation has not reached a terminal state.
    InFlight,
    /// A prior accepted draw did not prove that the fixed outgoing states held.
    Poisoned,
}

/// One private reservation for a snapshot generation.
///
/// Dropping an accepted operation without proving completion poisons the gate;
/// a pre-submit caller explicitly releases this reservation instead.
#[derive(Debug)]
pub(crate) struct SnapshotDrawReservation {
    gate: Arc<SnapshotUseGate>,
    active: bool,
}

impl SnapshotDrawReservation {
    /// Releases a reservation whose graph execution was rejected before submit.
    pub(crate) fn release_before_submit(mut self) {
        self.gate.release();
        self.active = false;
    }

    /// Releases a reservation after a complete submission restored known state.
    pub(crate) fn release_complete(&mut self) {
        if self.active {
            self.gate.release();
            self.active = false;
        }
    }

    /// Makes this generation permanently unusable after an uncertain outcome.
    pub(crate) fn poison(&mut self) {
        if self.active {
            self.gate.poison();
            self.active = false;
        }
    }
}

impl Drop for SnapshotDrawReservation {
    fn drop(&mut self) {
        if self.active {
            self.gate.poison();
        }
    }
}

#[derive(Debug)]
pub(super) struct SnapshotUseGate {
    state: Mutex<SnapshotUseState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotUseState {
    Ready,
    InFlight,
    Poisoned,
}

impl SnapshotUseGate {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(SnapshotUseState::Ready),
        }
    }

    pub(super) fn reserve(self: &Arc<Self>) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *state {
            SnapshotUseState::Ready => {
                *state = SnapshotUseState::InFlight;
                Ok(SnapshotDrawReservation {
                    gate: Arc::clone(self),
                    active: true,
                })
            }
            SnapshotUseState::InFlight => Err(SnapshotUseError::InFlight),
            SnapshotUseState::Poisoned => Err(SnapshotUseError::Poisoned),
        }
    }

    fn release(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert_eq!(*state, SnapshotUseState::InFlight);
        *state = SnapshotUseState::Ready;
    }

    fn poison(&self) {
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SnapshotUseState::Poisoned;
    }
}

/// Readiness retained by a closed multi-stream mesh upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RetainedUploadState {
    Missing,
    Pending,
    Ready,
}
