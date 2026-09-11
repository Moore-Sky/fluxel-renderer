//! Owns the proof harness's bounded admission, completion, and slot reuse.
//!
//! This module does not acquire a surface image or define renderer policy. It
//! only ensures the harness reserves one of a fixed number of private slots
//! before starting work and does not reuse that slot until its work reports a
//! known completion.

use std::fmt;

use fluxel_renderer::{
    FixedFrameFailure, RenderPacketFailure, RenderPacketStatus, RenderPacketSubmission,
    VisibleFrameStatus, VisibleFrameSubmission,
};

/// One operation whose known completion permits its private slot to be reused.
pub(crate) trait FrameWork {
    /// Terminal error reported by this operation.
    type Error;

    /// Advances completion observation once without waiting.
    fn poll_once(&mut self) -> WorkStatus<Self::Error>;
}

impl FrameWork for VisibleFrameSubmission {
    type Error = VisibleWorkError;

    fn poll_once(&mut self) -> WorkStatus<Self::Error> {
        match self.poll() {
            VisibleFrameStatus::Pending | VisibleFrameStatus::Busy => WorkStatus::Pending,
            VisibleFrameStatus::Submitted => WorkStatus::Submitted,
            VisibleFrameStatus::Complete => WorkStatus::Complete,
            VisibleFrameStatus::Failed(error) => {
                WorkStatus::Failed(VisibleWorkError::Failed(error))
            }
            status => WorkStatus::Failed(VisibleWorkError::Unsupported(format!("{status:?}"))),
        }
    }
}

impl FrameWork for RenderPacketSubmission {
    type Error = VisibleWorkError;

    fn poll_once(&mut self) -> WorkStatus<Self::Error> {
        match self.poll() {
            RenderPacketStatus::Pending | RenderPacketStatus::Busy => WorkStatus::Pending,
            RenderPacketStatus::Submitted => WorkStatus::Submitted,
            RenderPacketStatus::Presented => WorkStatus::Complete,
            RenderPacketStatus::Failed(error) => {
                WorkStatus::Failed(VisibleWorkError::PacketFailed(error))
            }
            status => WorkStatus::Failed(VisibleWorkError::Unsupported(format!("{status:?}"))),
        }
    }
}

/// Terminal visible-work error that stops the proof scheduler.
#[derive(Debug)]
pub(crate) enum VisibleWorkError {
    /// Renderer reported a structured fixed-frame failure.
    Failed(FixedFrameFailure),
    /// Renderer reported a structured multi-draw packet failure.
    PacketFailed(RenderPacketFailure),
    /// A future status was not assigned safe reuse semantics by this harness.
    Unsupported(String),
}

impl fmt::Display for VisibleWorkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed(error) => write!(formatter, "visible frame failed: {error}"),
            Self::PacketFailed(error) => write!(formatter, "visible packet failed: {error}"),
            Self::Unsupported(status) => {
                write!(
                    formatter,
                    "visible frame returned unsupported status: {status}"
                )
            }
        }
    }
}

/// Result of one nonblocking work observation.
#[derive(Debug)]
pub(crate) enum WorkStatus<E> {
    /// Work remains accepted but incomplete.
    Pending,
    /// Work was accepted by the raster/presentation queue but is not complete.
    Submitted,
    /// Work reached known completion and its slot may retire.
    Complete,
    /// Work reached an error; the caller must stop rather than reuse blindly.
    Failed(E),
}

/// Admission failed because all private slots remain reserved or in flight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackPressure;

/// Starting work failed either at bounded admission or before queue acceptance.
#[derive(Debug)]
pub(crate) enum StartError<E> {
    /// Every private slot remains reserved or in flight; the callback was not run.
    BackPressure,
    /// The callback ran but failed before returning accepted work.
    Start(E),
}

/// A pre-acquire reservation for one exact slot and serial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameReservation {
    slot: usize,
    serial: u64,
}

impl FrameReservation {
    pub(crate) const fn slot(self) -> usize {
        self.slot
    }

    pub(crate) const fn serial(self) -> u64 {
        self.serial
    }
}

/// Identity shared by submission and retirement observations for one slot use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameIdentity {
    /// Private slot made reusable by this retirement.
    pub(crate) slot: usize,
    /// Submission serial that owned the slot.
    pub(crate) serial: u64,
}

/// Aggregate result of one fair nonblocking pass over all live slots.
pub(crate) struct PollReport {
    retired: Vec<FrameIdentity>,
    submitted: Vec<FrameIdentity>,
    pending: usize,
}

impl PollReport {
    /// Returns slots whose exact work reached known completion.
    pub(crate) fn retired(&self) -> &[FrameIdentity] {
        &self.retired
    }

    /// Returns slots that reached the accepted submission milestone this pass.
    pub(crate) fn submitted(&self) -> &[FrameIdentity] {
        &self.submitted
    }

    /// Returns how many slots explicitly remained pending or busy.
    pub(crate) const fn pending(&self) -> usize {
        self.pending
    }
}

/// Private fixed-capacity ring used by the Windows proof harness.
pub(crate) struct FrameRing<W> {
    slots: Vec<FrameSlot<W>>,
    next_slot: usize,
    next_serial: u64,
    high_watermark: usize,
    back_pressure_count: u64,
}

enum FrameSlot<W> {
    Reusable,
    Reserved { serial: u64 },
    InFlight { serial: u64, work: W },
}

impl<W> FrameRing<W> {
    /// Creates a private ring. A zero capacity is rejected rather than acting
    /// like an always-blocked scheduler.
    pub(crate) fn new(capacity: usize) -> Option<Self> {
        (capacity > 0).then(|| Self {
            slots: (0..capacity).map(|_| FrameSlot::Reusable).collect(),
            next_slot: 0,
            next_serial: 0,
            high_watermark: 0,
            back_pressure_count: 0,
        })
    }

    /// Reserves capacity before the caller acquires any native surface image.
    pub(crate) fn try_reserve(&mut self) -> Result<FrameReservation, BackPressure> {
        for offset in 0..self.slots.len() {
            let slot = (self.next_slot + offset) % self.slots.len();
            if matches!(self.slots[slot], FrameSlot::Reusable) {
                let serial = self.next_serial;
                self.next_serial = self
                    .next_serial
                    .checked_add(1)
                    .expect("proof frame serial exhausted");
                self.slots[slot] = FrameSlot::Reserved { serial };
                self.next_slot = (slot + 1) % self.slots.len();
                self.high_watermark = self.high_watermark.max(self.live_count());
                return Ok(FrameReservation { slot, serial });
            }
        }
        self.back_pressure_count += 1;
        Err(BackPressure)
    }

    /// Reserves before invoking `start`, and installs only successfully started work.
    ///
    /// Keeping admission and callback invocation in one helper makes the key
    /// ordering testable: at capacity, native acquire/record code is never run.
    pub(crate) fn try_start<E>(
        &mut self,
        start: impl FnOnce(FrameReservation) -> Result<W, E>,
    ) -> Result<FrameReservation, StartError<E>> {
        let reservation = self.try_reserve().map_err(|_| StartError::BackPressure)?;
        match start(reservation) {
            Ok(work) => {
                self.commit(reservation, work);
                Ok(reservation)
            }
            Err(error) => {
                self.cancel(reservation);
                Err(StartError::Start(error))
            }
        }
    }

    /// Returns a reservation after work was proven not to have been accepted.
    pub(crate) fn cancel(&mut self, reservation: FrameReservation) {
        match self.slots.get(reservation.slot) {
            Some(FrameSlot::Reserved { serial }) if *serial == reservation.serial => {
                self.slots[reservation.slot] = FrameSlot::Reusable;
            }
            _ => panic!("frame reservation does not own the selected slot"),
        }
    }

    /// Installs accepted work into its exact pre-acquire reservation.
    pub(crate) fn commit(&mut self, reservation: FrameReservation, work: W) {
        match self.slots.get(reservation.slot) {
            Some(FrameSlot::Reserved { serial }) if *serial == reservation.serial => {
                self.slots[reservation.slot] = FrameSlot::InFlight {
                    serial: reservation.serial,
                    work,
                };
            }
            _ => panic!("frame reservation does not own the selected slot"),
        }
    }

    /// Returns whether no reservation or submission remains live.
    pub(crate) fn is_empty(&self) -> bool {
        self.live_count() == 0
    }

    /// Returns the current number of reserved and in-flight slots.
    pub(crate) fn live_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| !matches!(slot, FrameSlot::Reusable))
            .count()
    }

    /// Returns the fixed private capacity.
    pub(crate) fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Returns the largest observed live-slot count.
    pub(crate) const fn high_watermark(&self) -> usize {
        self.high_watermark
    }

    /// Returns how many reservations were refused at capacity.
    pub(crate) const fn back_pressure_count(&self) -> u64 {
        self.back_pressure_count
    }
}

impl<W: FrameWork> FrameRing<W> {
    /// Polls one exact slot until orchestration observes its submission
    /// milestone, without advancing completion of older slots.
    pub(crate) fn poll_reserved_once(
        &mut self,
        reservation: FrameReservation,
    ) -> Result<PollReport, W::Error> {
        let state = self
            .slots
            .get_mut(reservation.slot)
            .expect("frame reservation slot is in range");
        let FrameSlot::InFlight { serial, work } = state else {
            panic!("frame reservation has no in-flight work");
        };
        assert_eq!(*serial, reservation.serial, "frame serial was replaced");
        let identity = FrameIdentity {
            slot: reservation.slot,
            serial: reservation.serial,
        };
        match work.poll_once() {
            WorkStatus::Pending => Ok(PollReport {
                retired: Vec::new(),
                submitted: Vec::new(),
                pending: 1,
            }),
            WorkStatus::Submitted => Ok(PollReport {
                retired: Vec::new(),
                submitted: vec![identity],
                pending: 0,
            }),
            WorkStatus::Complete => {
                *state = FrameSlot::Reusable;
                Ok(PollReport {
                    retired: vec![identity],
                    submitted: Vec::new(),
                    pending: 0,
                })
            }
            WorkStatus::Failed(error) => Err(error),
        }
    }

    /// Polls every in-flight slot once and retires only known completions.
    pub(crate) fn poll_once(&mut self) -> Result<PollReport, W::Error> {
        let mut retired = Vec::new();
        let mut submitted = Vec::new();
        let mut pending = 0;
        for (slot, state) in self.slots.iter_mut().enumerate() {
            let FrameSlot::InFlight { serial, work } = state else {
                continue;
            };
            match work.poll_once() {
                WorkStatus::Pending => pending += 1,
                WorkStatus::Submitted => {
                    submitted.push(FrameIdentity {
                        slot,
                        serial: *serial,
                    });
                }
                WorkStatus::Complete => {
                    retired.push(FrameIdentity {
                        slot,
                        serial: *serial,
                    });
                    *state = FrameSlot::Reusable;
                }
                WorkStatus::Failed(error) => return Err(error),
            }
        }
        Ok(PollReport {
            retired,
            submitted,
            pending,
        })
    }
}

#[cfg(test)]
mod tests;
