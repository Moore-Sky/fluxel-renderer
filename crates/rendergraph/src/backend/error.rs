//! Execution and frame-binding error types.

use std::{error::Error, fmt};

use crate::{
    error::RecordingError,
    handles::{ImportBufferSlot, ImportTextureSlot, ResourceId, SurfaceBindingId},
};

/// Machine-readable reason that a frame input could not be resolved.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FrameBindingErrorKind {
    /// A required imported texture was not supplied.
    MissingTexture,
    /// A required imported buffer was not supplied.
    MissingBuffer,
    /// A required acquired surface image was not supplied.
    MissingSurface,
    /// A supplied object belongs to a different backend device.
    DeviceMismatch,
    /// The supplied descriptor does not satisfy the import contract.
    DescriptorMismatch,
    /// The supplied incoming state differs from the import contract.
    InitialStateMismatch,
    /// The physical resource does not permit every compiled operation.
    UsageMismatch,
    /// A binding identity resolved to the wrong resource kind.
    WrongResourceKind,
    /// Multiple binding categories targeted the same import slot.
    ConflictingSlotBinding,
    /// Distinct logical resources resolved to the same physical generation.
    AliasedPhysicalResource,
}

/// Structured failure while resolving per-frame imported resources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameBindingError {
    /// Stable category for programmatic handling.
    pub kind: FrameBindingErrorKind,
    /// Imported texture slot involved in the failure, if applicable.
    pub texture_slot: Option<ImportTextureSlot>,
    /// Imported buffer slot involved in the failure, if applicable.
    pub buffer_slot: Option<ImportBufferSlot>,
    /// Logical resource involved in the failure, if known.
    pub resource: Option<ResourceId>,
    /// Acquired surface binding involved in the failure, if applicable.
    pub surface_binding: Option<SurfaceBindingId>,
    /// Human-readable context not intended for programmatic matching.
    pub detail: String,
}

impl fmt::Display for FrameBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "render graph frame binding failed ({:?}): {}",
            self.kind, self.detail
        )
    }
}

impl Error for FrameBindingError {}

/// Failure produced while executing a compiled graph through one backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExecutionError<E> {
    /// A required imported frame resource could not be resolved.
    FrameBinding(FrameBindingError),
    /// A retained callback or pass-local validation failed.
    Recording(RecordingError),
    /// Runtime backend capabilities differ from the compiled fingerprint.
    CapabilityMismatch,
    /// A frame instance was created from a different compiled graph.
    WrongCompiledGraph,
    /// The executor backend is already borrowed by another operation.
    ExecutorBusy,
    /// The plan requires a feature excluded from this executor milestone.
    UnsupportedExecutionFeature(&'static str),
    /// The backend rejected resource creation, recording, or submission.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for ExecutionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameBinding(error) => error.fmt(formatter),
            Self::Recording(error) => error.fmt(formatter),
            Self::CapabilityMismatch => {
                formatter.write_str("render graph execution capability mismatch")
            }
            Self::WrongCompiledGraph => {
                formatter.write_str("frame execution belongs to another compiled graph")
            }
            Self::ExecutorBusy => formatter.write_str("render graph executor is already busy"),
            Self::UnsupportedExecutionFeature(feature) => {
                write!(
                    formatter,
                    "unsupported render graph execution feature: {feature}"
                )
            }
            Self::Backend(error) => write!(formatter, "render graph execution failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for ExecutionError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::FrameBinding(error) => Some(error),
            Self::Recording(error) => Some(error),
            Self::Backend(error) => Some(error),
            Self::CapabilityMismatch
            | Self::WrongCompiledGraph
            | Self::ExecutorBusy
            | Self::UnsupportedExecutionFeature(_) => None,
        }
    }
}
