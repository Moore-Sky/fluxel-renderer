//! Structured graph compilation errors.

use std::{error::Error, fmt};

use crate::{
    handles::{ImportBufferSlot, ImportTextureSlot, PassId, ResourceId},
    rhi::DeviceCapabilities,
};

/// Machine-readable categories of graph compilation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompileErrorKind {
    /// A resource version is stale or belongs to another graph.
    StaleOrForeignVersion,
    /// A transient resource is read before its first write.
    ReadBeforeInitialization,
    /// More than one writer consumes the same whole-resource version.
    InvalidResourceBranch,
    /// Declared accesses conflict or require unsupported pass-internal ordering.
    ConflictingAccess,
    /// A texture or buffer range is outside its descriptor.
    InvalidSubresourceRange,
    /// Resource and explicit-order edges contain a cycle.
    DependencyCycle,
    /// An imported resource does not have a complete contract.
    MissingImportContract,
    /// The device cannot preserve a requested semantic operation.
    UnsupportedSemanticRequirement,
    /// An export or presentation root is invalid.
    InvalidExportOrPresent,
}

/// Machine-readable categories of command-recording failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RecordingErrorKind {
    /// An access handle was not declared by the pass being recorded.
    ForeignOrUndeclaredPassAccess,
    /// A binding recipe requires a use incompatible with the declared access.
    DeclaredUseMismatch,
    /// A required physical resource was not supplied for this frame.
    MissingFrameBinding,
    /// A dynamic offset, index format, or command range is invalid.
    InvalidCommandArgument,
    /// Renderer/RHI binding metadata is incompatible with the selected pipeline.
    IncompatibleBindingRecipe,
    /// Creation of a backend-owned pipeline or binding object failed after validation.
    BackendObjectCreation,
}

/// Optional identities and details attached to a compilation error.
#[derive(Clone, Debug)]
pub struct DiagnosticContext {
    /// Passes directly involved in the failure.
    pub passes: Vec<PassId>,
    /// Logical resource involved in the failure.
    pub resource: Option<ResourceId>,
    /// Imported texture slot involved in the failure.
    pub texture_slot: Option<ImportTextureSlot>,
    /// Imported buffer slot involved in the failure.
    pub buffer_slot: Option<ImportBufferSlot>,
    /// Human-readable detail that is not intended for programmatic matching.
    pub detail: String,
    /// Capabilities observed when a semantic requirement was rejected.
    pub capabilities: Option<Box<DeviceCapabilities>>,
}

/// A structured failure returned by render graph compilation.
#[derive(Clone, Debug)]
pub struct CompileError {
    /// Stable category for programmatic handling.
    pub kind: CompileErrorKind,
    /// Resource, pass, and capability evidence for diagnostics.
    pub context: DiagnosticContext,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "render graph compilation failed ({:?}): {}",
            self.kind, self.context.detail
        )
    }
}

impl Error for CompileError {}

/// A structured failure produced while resolving pass resources or recording commands.
#[derive(Clone, Debug)]
pub struct RecordingError {
    /// Stable category for programmatic handling.
    pub kind: RecordingErrorKind,
    /// Pass/resource/binding evidence suitable for diagnostics.
    pub context: DiagnosticContext,
}

impl fmt::Display for RecordingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "render graph recording failed ({:?}): {}",
            self.kind, self.context.detail
        )
    }
}

impl Error for RecordingError {}

/// Result returned by pass recording callbacks and validating command methods.
pub type RecordResult<T = ()> = Result<T, RecordingError>;
