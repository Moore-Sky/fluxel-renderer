//! Deterministic CPU-only execution fixtures.
//!
//! This module keeps the test backend behind the same public module path while
//! allowing its implementation to be decomposed without exposing test-only
//! internals through the main execution contract.

mod backend;
mod lease;
mod registry;
mod trace;

pub use backend::*;
pub use lease::{TestLease, TestLeaseProbe};
pub use registry::*;
pub use trace::*;
