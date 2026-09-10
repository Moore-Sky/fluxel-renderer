//! GPU completion ownership and non-blocking in-flight retirement.

use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

use crate::backend::{CompletionStatus, ExecutionBackend, ExecutionError};

pub(crate) struct PendingRetirement<B: ExecutionBackend> {
    completion: B::Completion,
    leases: Vec<B::Lease>,
}

pub(crate) type RetirementInbox<B> = Arc<Mutex<Vec<PendingRetirement<B>>>>;

/// One submitted frame whose strong leases remain live until GPU completion.
pub struct FrameSubmission<B: ExecutionBackend> {
    pub(crate) backend: Arc<Mutex<B>>,
    pub(crate) retirement_inbox: RetirementInbox<B>,
    pub(crate) completion: Option<B::Completion>,
    pub(crate) leases: Option<Vec<B::Lease>>,
    settled: bool,
}

impl<B: ExecutionBackend> FrameSubmission<B> {
    pub(crate) fn new(
        backend: Arc<Mutex<B>>,
        retirement_inbox: RetirementInbox<B>,
        completion: B::Completion,
        leases: Vec<B::Lease>,
    ) -> Self {
        Self {
            backend,
            retirement_inbox,
            completion: Some(completion),
            leases: Some(leases),
            settled: false,
        }
    }

    /// Returns the backend completion object while this submission is live.
    pub fn completion(&self) -> &B::Completion {
        self.completion
            .as_ref()
            .expect("live submission always owns its completion")
    }

    /// Polls this submission and releases executor-held leases on terminal state.
    pub fn status(&mut self) -> Result<CompletionStatus, ExecutionError<B::Error>> {
        let backend = try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        let status = backend.completion_status(self.completion());
        if matches!(
            status,
            CompletionStatus::Complete | CompletionStatus::Failed(_)
        ) {
            self.leases.take();
            self.settled = true;
        }
        Ok(status)
    }

    /// Returns the number of leases currently retained directly by this value.
    pub fn retained_lease_count(&self) -> usize {
        self.leases.as_ref().map_or(0, Vec::len)
    }
}

impl<B: ExecutionBackend> Drop for FrameSubmission<B> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let Some(completion) = self.completion.take() else {
            return;
        };
        // Dropping the public handle cannot release resources still referenced
        // by accepted GPU work. Transfer completion plus leases into the shared
        // inbox; the executor will retire them only after a terminal query.
        let leases = self.leases.take().unwrap_or_default();
        lock_inbox(&self.retirement_inbox).push(PendingRetirement { completion, leases });
    }
}

pub(crate) fn drain_retirement_inbox<B: ExecutionBackend>(
    backend: &mut B,
    inbox: &RetirementInbox<B>,
) {
    for pending in lock_inbox(inbox).drain(..) {
        backend.retire(pending.completion, pending.leases);
    }
}

pub(crate) fn try_lock_backend<B>(backend: &Arc<Mutex<B>>) -> Result<MutexGuard<'_, B>, ()> {
    match backend.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => Err(()),
    }
}

fn lock_inbox<B: ExecutionBackend>(
    inbox: &RetirementInbox<B>,
) -> MutexGuard<'_, Vec<PendingRetirement<B>>> {
    // A producer panic cannot invalidate already-owned completion/lease pairs.
    // Recover the inner queue so they remain conservatively retained.
    inbox
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
