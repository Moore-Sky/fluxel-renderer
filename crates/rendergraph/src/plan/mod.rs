//! Derives backend-independent execution plans from validated compiled graphs.
//!
//! Plans preserve the compiler's pass order, resource requirements, and transition intent;
//! they do not allocate resources, emit backend barriers, or submit work. Execution consumes
//! this immutable description to resolve frame resources and lower transitions, so every
//! planned access must have one consistent state before the next planned use.

mod build;
mod model;
mod state;

pub use model::*;

pub(crate) use build::{build_execution_plan, required_state};
