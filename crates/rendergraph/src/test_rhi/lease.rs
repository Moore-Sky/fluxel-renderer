//! Observable strong-lease primitives used by the CPU test backend.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// A probe for the strong leases retained by one test object.
#[derive(Clone, Debug)]
pub struct TestLeaseProbe(pub(crate) Arc<AtomicUsize>);

impl TestLeaseProbe {
    /// Returns the number of currently live leases for this object.
    pub fn active_leases(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// A cloneable strong lease used for all test objects.
#[derive(Debug)]
pub struct TestLease {
    counter: Arc<AtomicUsize>,
}

impl TestLease {
    pub(crate) fn fresh() -> (Self, TestLeaseProbe) {
        let counter = Arc::new(AtomicUsize::new(1));
        (
            Self {
                counter: Arc::clone(&counter),
            },
            TestLeaseProbe(counter),
        )
    }

    /// Returns an observable probe for this lease's shared counter.
    pub fn probe(&self) -> TestLeaseProbe {
        TestLeaseProbe(Arc::clone(&self.counter))
    }
}

impl Clone for TestLease {
    fn clone(&self) -> Self {
        self.counter.fetch_add(1, Ordering::SeqCst);
        Self {
            counter: Arc::clone(&self.counter),
        }
    }
}

impl Drop for TestLease {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}
