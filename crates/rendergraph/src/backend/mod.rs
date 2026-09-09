//! Minimal backend-facing execution contract.
//!
//! This module deliberately models only the boundary from a compiled graph to
//! one ordered submission. It is not a general-purpose graphics API: graph
//! validation owns access legality and transition planning, while a backend
//! receives already-legal semantic operations to record and submit.

mod contract;
mod error;
mod resource;

pub use contract::*;
pub use error::*;
pub use resource::*;
