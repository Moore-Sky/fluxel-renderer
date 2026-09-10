//! Closed vertex-color upload domain assembled from data and lifecycle modules.

mod domain;
mod upload;

pub use domain::*;
pub use upload::*;

#[cfg(test)]
pub(crate) use domain::VertexColorMeshPayload;
#[cfg(test)]
pub(in crate::upload) use upload::{
    VertexColorSnapshotPublication, completion_failure, observation_failure,
    unknown_completion_failure,
};
