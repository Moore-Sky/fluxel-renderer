//! Opaque identities used by graph declarations and diagnostics.

macro_rules! copy_handle {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) u64);
    };
}

copy_handle!(
    /// Stable identity of one pass inside a render graph.
    PassId
);
copy_handle!(
    /// Stable identity of one logical texture or buffer.
    ResourceId
);
copy_handle!(
    /// Stable key used to bind an external texture for one frame.
    ImportTextureSlot
);
copy_handle!(
    /// Stable key used to bind an external buffer for one frame.
    ImportBufferSlot
);
copy_handle!(
    /// Stable key describing one exported texture result.
    ExportTextureSlot
);
copy_handle!(
    /// Stable key describing one exported buffer result.
    ExportBufferSlot
);
copy_handle!(
    /// Stable identity of a presentation root.
    PresentTarget
);
copy_handle!(
    /// Renderer-owned identity of an opaque shader binding recipe.
    BindingSetId
);
copy_handle!(
    /// Renderer-owned identity of one physical texture frame binding.
    TextureBindingId
);
copy_handle!(
    /// Renderer-owned identity of one physical buffer frame binding.
    BufferBindingId
);
copy_handle!(
    /// Renderer-owned identity of one acquired presentation image.
    SurfaceBindingId
);

/// One immutable logical version of a texture.
///
/// Reads borrow a version. Writes consume it and return a successor version.
/// The compiler remains the authority for stale, foreign, and branching-use
/// validation.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TextureVersion {
    pub(crate) resource: ResourceId,
    pub(crate) version: u32,
}

/// One immutable logical version of a buffer.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferVersion {
    pub(crate) resource: ResourceId,
    pub(crate) version: u32,
}

/// A texture read declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TextureRead(pub(crate) u64);

/// A texture write declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TextureWrite(pub(crate) u64);

/// A texture read-modify-write declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TextureReadWrite(pub(crate) u64);

/// A buffer read declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferRead(pub(crate) u64);

/// A buffer write declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferWrite(pub(crate) u64);

/// A buffer read-modify-write declared by exactly one pass.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferReadWrite(pub(crate) u64);

/// Renderer-owned identity of a raster pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RasterPipelineId(u64);

impl RasterPipelineId {
    /// Creates an opaque renderer pipeline identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// Renderer-owned identity of a compute pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputePipelineId(u64);

impl ComputePipelineId {
    /// Creates an opaque renderer pipeline identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

impl BindingSetId {
    /// Creates an opaque renderer binding-recipe identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

impl TextureBindingId {
    /// Creates an opaque physical-texture binding identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

impl BufferBindingId {
    /// Creates an opaque physical-buffer binding identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

impl SurfaceBindingId {
    /// Creates an opaque acquired-surface binding identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

impl TextureVersion {
    pub(crate) fn new(resource: ResourceId, version: u32) -> Self {
        Self { resource, version }
    }
}

impl BufferVersion {
    pub(crate) fn new(resource: ResourceId, version: u32) -> Self {
        Self { resource, version }
    }
}
