//! Reference validation orchestrator.
//!
//! Ownership checks must run before downstream passes dereference declarations. Range helpers
//! centralize the portable descriptor invariant so dependency and conflict checks agree.

mod graph;
mod range;

pub(in crate::compile) use graph::validate_references;
