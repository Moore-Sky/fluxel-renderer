//! Backend-independent executable plans derived from compiled graphs.

mod build;
mod model;
mod state;

pub use model::*;

pub(crate) use build::{build_execution_plan, required_state};
