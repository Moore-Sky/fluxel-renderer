//! Public pass-facing types, split between declaration, resource resolution, and recording.
//!
//! Builders describe graph-local resource uses; recording contexts only accept the resulting
//! typed handles. This boundary prevents execution callbacks from manufacturing undeclared access.

mod authoring;
mod commands;
mod resolver;
mod types;

pub use authoring::*;
pub use commands::*;
pub use resolver::*;
pub use types::*;
