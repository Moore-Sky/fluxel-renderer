//! Semantic resource usages and subresource ranges.

/// Whether one pass reads, writes, or modifies a resource version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessMode {
    /// The pass only observes prior contents.
    Read,
    /// The pass produces a successor version without observing prior contents.
    Write,
    /// The pass observes prior contents and produces a successor version.
    ReadWrite,
}

/// Definite initialization guaranteed by a successful write.
///
/// Coverage is relative to the declared access range and is independent of the
/// set of bytes or texels the GPU may access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WriteCoverage {
    /// The pass adds no new complete-initialization guarantee for the range.
    /// Previously valid contents remain valid unless another operation such as
    /// attachment `DontCare` or `Discard` explicitly invalidates them.
    Unknown,
    /// The pass guarantees every element in the declared range is initialized.
    Full,
}

/// A texture aspect selected by one access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureAspect {
    /// All aspects supported by the format.
    All,
    /// The color aspect.
    Color,
    /// The depth aspect.
    Depth,
    /// The stencil aspect.
    Stencil,
}

/// A mip, array-layer, and aspect range of a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureRange {
    /// Every subresource described by the logical texture.
    Whole,
    /// One explicit, non-empty subresource range.
    Subresources {
        /// First selected mip level.
        base_mip_level: u32,
        /// Number of selected mip levels.
        mip_level_count: u32,
        /// First selected array layer.
        base_array_layer: u32,
        /// Number of selected array layers.
        array_layer_count: u32,
        /// Selected format aspect.
        aspect: TextureAspect,
    },
}

impl TextureRange {
    /// Selects every subresource described by the logical texture.
    pub fn whole() -> Self {
        Self::Whole
    }
}

/// A byte range of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferRange {
    /// Every byte described by the logical buffer.
    Whole,
    /// One explicit, non-empty byte range.
    Bytes {
        /// First selected byte.
        offset: u64,
        /// Number of selected bytes.
        size: u64,
    },
}

impl BufferRange {
    /// Selects the complete logical buffer.
    pub fn whole() -> Self {
        Self::Whole
    }
}

/// Semantic reasons for reading a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureReadUse {
    /// Shader sampled-texture access.
    Sampled,
    /// Read-only shader storage access.
    Storage,
    /// Source of a copy command.
    CopySource,
}

/// Semantic reasons for writing a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureWriteUse {
    /// Write-only shader storage access.
    Storage,
    /// Destination of a copy command.
    CopyDestination,
}

/// Semantic reasons for reading and writing the same texture in one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TextureReadWriteUse {
    /// Shader storage read-modify-write access.
    Storage,
}

/// Semantic reasons for reading a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BufferReadUse {
    /// Uniform or constant-buffer access.
    Uniform,
    /// Read-only shader storage access.
    Storage,
    /// Vertex input access.
    Vertex,
    /// Index input access.
    Index,
    /// Indirect command access.
    Indirect,
    /// Source of a copy command.
    CopySource,
}

/// Semantic reasons for writing a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BufferWriteUse {
    /// Write-only shader storage access.
    Storage,
    /// Destination of a copy command.
    CopyDestination,
}

/// Semantic reasons for reading and writing the same buffer in one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BufferReadWriteUse {
    /// Shader storage read-modify-write access.
    Storage,
}

/// A logical texture-to-texture copy region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureCopyRegion {
    /// Source origin in texels.
    pub source_origin: [u32; 3],
    /// Destination origin in texels.
    pub destination_origin: [u32; 3],
    /// Copied extent in texels.
    pub extent: [u32; 3],
    /// Source mip level.
    pub source_mip_level: u32,
    /// Destination mip level.
    pub destination_mip_level: u32,
}

/// A logical buffer-to-buffer copy region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferCopyRegion {
    /// First source byte.
    pub source_offset: u64,
    /// First destination byte.
    pub destination_offset: u64,
    /// Number of copied bytes.
    pub size: u64,
}
