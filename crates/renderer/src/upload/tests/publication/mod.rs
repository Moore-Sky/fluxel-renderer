//! Atomic publication-policy contracts.

use super::*;

pub(crate) fn publication_policy_never_exposes_a_partially_ready_or_failed_pair() {
    let pending = RetainedUploadState::Pending;
    let ready = RetainedUploadState::Ready;
    assert!(!SnapshotPublication::new(pending, pending, false).can_publish());
    assert!(!SnapshotPublication::new(ready, pending, false).can_publish());
    assert!(!SnapshotPublication::new(pending, ready, false).can_publish());
    assert!(SnapshotPublication::new(ready, ready, false).can_publish());

    // A failure is monotonic from the renderer's point of view. The
    // sibling can still be retained and later reach Ready, but that must
    // never repair this generation into a published snapshot.
    assert!(!SnapshotPublication::new(ready, ready, true).can_publish());
}

pub(crate) fn publication_policy_represents_partial_acceptance_without_ready_output() {
    assert_eq!(
        SnapshotPublication::new(
            RetainedUploadState::Pending,
            RetainedUploadState::Missing,
            true,
        ),
        SnapshotPublication {
            positions: RetainedUploadState::Pending,
            indices: RetainedUploadState::Missing,
            failed: true,
        }
    );
}
