//! Groups validation passes by the graph invariant each one proves.

mod capabilities;
mod initialization;
mod references;
mod roots;

pub(super) use capabilities::validate_capabilities;
pub(super) use initialization::validate_initialization;
pub(super) use references::validate_references;
pub(super) use roots::{
    buffer_boundary_state, buffer_state_supported, texture_boundary_state, texture_state_supported,
    validate_roots,
};
