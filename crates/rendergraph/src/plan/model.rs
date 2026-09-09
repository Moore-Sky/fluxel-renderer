use core::fmt;

use crate::{
    access::{BufferRange, TextureRange},
    handles::{PassId, ResourceId},
    pass::{ColorAttachmentDesc, DepthStencilAttachmentDesc, PassKind},
    rhi::{QueueId, ResourceAccessState},
};

/// One texture operation that must be enabled when a physical texture is
/// created or accepted for an execution plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TextureUsageKind {
    /// Shader sampled reads.
    Sampled,
    /// Shader storage reads.
    StorageRead,
    /// Shader storage writes.
    StorageWrite,
    /// Color-attachment reads or writes.
    ColorAttachment,
    /// Depth-stencil-attachment reads or writes.
    DepthStencilAttachment,
    /// Copy-source reads.
    CopySource,
    /// Copy-destination writes.
    CopyDestination,
    /// Presentation after graph execution.
    Present,
}

/// The creation-time operations required by one texture.
///
/// The compiler derives this value from retained accesses and import/export
/// boundary states. It deliberately models domain operations rather than
/// exposing backend-specific flag bits.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct TextureUsage(u16);

const TEXTURE_USAGE_KINDS: [TextureUsageKind; 8] = [
    TextureUsageKind::Sampled,
    TextureUsageKind::StorageRead,
    TextureUsageKind::StorageWrite,
    TextureUsageKind::ColorAttachment,
    TextureUsageKind::DepthStencilAttachment,
    TextureUsageKind::CopySource,
    TextureUsageKind::CopyDestination,
    TextureUsageKind::Present,
];

impl fmt::Debug for TextureUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TextureUsage")?;
        formatter
            .debug_set()
            .entries(
                TEXTURE_USAGE_KINDS
                    .into_iter()
                    .filter(|kind| self.contains(*kind)),
            )
            .finish()
    }
}

impl TextureUsage {
    /// Creates an empty operation set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates an operation set from domain-level texture operations.
    ///
    /// This is appropriate when a resource provider reports the operations a
    /// physical texture actually permits. It intentionally does not expose
    /// backend flag bits.
    pub fn from_kinds(kinds: impl IntoIterator<Item = TextureUsageKind>) -> Self {
        kinds
            .into_iter()
            .fold(Self::empty(), |usage, kind| usage.with(kind))
    }

    /// Returns this set with `kind` enabled.
    #[must_use]
    pub const fn with(self, kind: TextureUsageKind) -> Self {
        Self(self.0 | texture_usage_bit(kind))
    }

    /// Returns whether this set contains `kind`.
    pub const fn contains(&self, kind: TextureUsageKind) -> bool {
        self.0 & texture_usage_bit(kind) != 0
    }

    /// Returns whether this allowed-operation set covers every operation in
    /// `required`.
    pub const fn contains_all(&self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub(super) fn insert(&mut self, kind: TextureUsageKind) {
        *self = self.with(kind);
    }
}

const fn texture_usage_bit(kind: TextureUsageKind) -> u16 {
    match kind {
        TextureUsageKind::Sampled => 1 << 0,
        TextureUsageKind::StorageRead => 1 << 1,
        TextureUsageKind::StorageWrite => 1 << 2,
        TextureUsageKind::ColorAttachment => 1 << 3,
        TextureUsageKind::DepthStencilAttachment => 1 << 4,
        TextureUsageKind::CopySource => 1 << 5,
        TextureUsageKind::CopyDestination => 1 << 6,
        TextureUsageKind::Present => 1 << 7,
    }
}

/// One buffer operation that must be enabled when a physical buffer is
/// created or accepted for an execution plan.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BufferUsageKind {
    /// Uniform-buffer reads.
    Uniform,
    /// Shader storage reads.
    StorageRead,
    /// Shader storage writes.
    StorageWrite,
    /// Vertex-buffer reads.
    Vertex,
    /// Index-buffer reads.
    Index,
    /// Indirect-command reads.
    Indirect,
    /// Copy-source reads.
    CopySource,
    /// Copy-destination writes.
    CopyDestination,
}

/// The creation-time operations required by one buffer.
///
/// The compiler derives this value from retained accesses and import/export
/// boundary states. It deliberately models domain operations rather than
/// exposing backend-specific flag bits.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
pub struct BufferUsage(u16);

const BUFFER_USAGE_KINDS: [BufferUsageKind; 8] = [
    BufferUsageKind::Uniform,
    BufferUsageKind::StorageRead,
    BufferUsageKind::StorageWrite,
    BufferUsageKind::Vertex,
    BufferUsageKind::Index,
    BufferUsageKind::Indirect,
    BufferUsageKind::CopySource,
    BufferUsageKind::CopyDestination,
];

impl fmt::Debug for BufferUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BufferUsage")?;
        formatter
            .debug_set()
            .entries(
                BUFFER_USAGE_KINDS
                    .into_iter()
                    .filter(|kind| self.contains(*kind)),
            )
            .finish()
    }
}

impl BufferUsage {
    /// Creates an empty operation set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates an operation set from domain-level buffer operations.
    ///
    /// This is appropriate when a resource provider reports the operations a
    /// physical buffer actually permits. It intentionally does not expose
    /// backend flag bits.
    pub fn from_kinds(kinds: impl IntoIterator<Item = BufferUsageKind>) -> Self {
        kinds
            .into_iter()
            .fold(Self::empty(), |usage, kind| usage.with(kind))
    }

    /// Returns this set with `kind` enabled.
    #[must_use]
    pub const fn with(self, kind: BufferUsageKind) -> Self {
        Self(self.0 | buffer_usage_bit(kind))
    }

    /// Returns whether this set contains `kind`.
    pub const fn contains(&self, kind: BufferUsageKind) -> bool {
        self.0 & buffer_usage_bit(kind) != 0
    }

    /// Returns whether this allowed-operation set covers every operation in
    /// `required`.
    pub const fn contains_all(&self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    pub(super) fn insert(&mut self, kind: BufferUsageKind) {
        *self = self.with(kind);
    }
}

const fn buffer_usage_bit(kind: BufferUsageKind) -> u16 {
    match kind {
        BufferUsageKind::Uniform => 1 << 0,
        BufferUsageKind::StorageRead => 1 << 1,
        BufferUsageKind::StorageWrite => 1 << 2,
        BufferUsageKind::Vertex => 1 << 3,
        BufferUsageKind::Index => 1 << 4,
        BufferUsageKind::Indirect => 1 << 5,
        BufferUsageKind::CopySource => 1 << 6,
        BufferUsageKind::CopyDestination => 1 << 7,
    }
}

/// The creation-time operation summary for one logical resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceUsageSummary {
    /// Texture operations.
    Texture(TextureUsage),
    /// Buffer operations.
    Buffer(BufferUsage),
}

/// Creation-time requirements for one live logical resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ResourceRequirement {
    /// Logical resource whose physical allocation or import is constrained.
    pub resource: ResourceId,
    /// Backend-independent operations required for that resource.
    pub usage: ResourceUsageSummary,
}

/// A texture or buffer range named by an executable transition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlannedResourceRange {
    /// Texture subresources.
    Texture(TextureRange),
    /// Buffer bytes.
    Buffer(BufferRange),
}

/// One semantic state transition emitted before a pass or export.
///
/// Equal `before` and `after` states represent a required same-state memory
/// barrier, not a no-op.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlannedTransition {
    /// Logical resource whose physical object is transitioned.
    pub resource: ResourceId,
    /// Exact range affected by this transition.
    pub range: PlannedResourceRange,
    /// State expected before the transition.
    pub before: ResourceAccessState,
    /// State required after the transition.
    pub after: ResourceAccessState,
}

/// One color attachment retained by an executable raster pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedColorAttachment {
    /// Logical texture used by the attachment.
    pub resource: ResourceId,
    /// Complete setup-time attachment declaration.
    pub descriptor: ColorAttachmentDesc,
}

/// One depth-stencil attachment retained by an executable raster pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedDepthStencilAttachment {
    /// Logical texture used by the attachment.
    pub resource: ResourceId,
    /// Complete setup-time attachment declaration.
    pub descriptor: DepthStencilAttachmentDesc,
}

/// Executable raster-pass attachment information.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RasterPassPlan {
    /// Color attachments ordered by their declared slot.
    pub colors: Vec<PlannedColorAttachment>,
    /// Optional depth-stencil attachment.
    pub depth_stencil: Option<PlannedDepthStencilAttachment>,
}

/// One retained pass in executable order.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedPass {
    /// Stable pass identity.
    pub pass: PassId,
    /// Command family recorded by this pass.
    pub kind: PassKind,
    /// Semantic transitions emitted before the pass begins.
    pub transitions: Vec<PlannedTransition>,
    /// Attachment information for raster passes.
    pub raster: Option<RasterPassPlan>,
}

/// Immutable single-queue execution plan produced during graph compilation.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan {
    queue: Option<QueueId>,
    passes: Vec<PlannedPass>,
    final_transitions: Vec<PlannedTransition>,
    resource_requirements: Vec<ResourceRequirement>,
}

impl ExecutionPlan {
    /// Returns the one logical queue selected by the current lowering.
    pub fn queue(&self) -> Option<QueueId> {
        self.queue
    }

    /// Returns retained passes in deterministic execution order.
    pub fn passes(&self) -> &[PlannedPass] {
        &self.passes
    }

    /// Returns transitions required after the last pass for exported roots.
    pub fn final_transitions(&self) -> &[PlannedTransition] {
        &self.final_transitions
    }

    /// Returns creation-time requirements for every live logical resource in
    /// deterministic resource-creation order.
    pub fn resource_requirements(&self) -> &[ResourceRequirement] {
        &self.resource_requirements
    }

    /// Returns creation-time requirements for `resource`, when it is live.
    pub fn resource_requirement(&self, resource: ResourceId) -> Option<ResourceUsageSummary> {
        self.resource_requirements
            .binary_search_by_key(&resource.0, |requirement| requirement.resource.0)
            .ok()
            .map(|index| self.resource_requirements[index].usage)
    }
}

pub(super) fn execution_plan(
    queue: Option<QueueId>,
    passes: Vec<PlannedPass>,
    final_transitions: Vec<PlannedTransition>,
    resource_requirements: Vec<ResourceRequirement>,
) -> ExecutionPlan {
    ExecutionPlan {
        queue,
        passes,
        final_transitions,
        resource_requirements,
    }
}
