//! General public pass vocabulary shared by authoring, recording, and backend contracts.

use std::marker::PhantomData;

use crate::{
    access::{TextureRange, WriteCoverage},
    handles::{
        BufferRead, BufferReadWrite, BufferWrite, TextureRead, TextureReadWrite, TextureWrite,
    },
};

/// Logical kind of commands recorded by one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Rasterization and attachment work.
    Raster,
    /// Compute-dispatch work.
    Compute,
    /// Resource-copy work.
    Copy,
}

/// Contents operation performed when an attachment begins its pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Preserves and reads the previous contents.
    Load,
    /// Initializes the selected range to one clear value.
    Clear(T),
    /// Does not preserve previous contents and does not imply initialization.
    DontCare,
}

/// Contents operation performed when an attachment ends its pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Makes valid produced contents available to later accesses.
    Store,
    /// Invalidates the selected contents after the pass.
    Discard,
}

/// Initial and final content operations for one attachment aspect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttachmentOps<T> {
    /// Initial content operation.
    pub load: LoadOp<T>,
    /// Final content operation.
    pub store: StoreOp,
    /// Definite shader/draw write coverage of the selected range.
    ///
    /// `Clear` initializes the full range regardless of this value. `Load`
    /// inherits valid input contents. `DontCare` needs `Full` before a later
    /// read can rely on newly initialized contents.
    pub write_coverage: WriteCoverage,
}

/// Declaration of one color attachment and its content operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachmentDesc {
    /// Zero-based color attachment index.
    pub index: u32,
    /// Attachment view selected from the logical texture.
    pub range: TextureRange,
    /// Initial and final color-content operations.
    pub operations: AttachmentOps<[f32; 4]>,
}

/// Declaration of one depth-stencil attachment and its content operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStencilAttachmentDesc {
    /// Attachment view selected from the logical texture.
    pub range: TextureRange,
    /// Depth-aspect operations, or `None` when unused.
    pub depth: Option<AttachmentOps<f32>>,
    /// Stencil-aspect operations, or `None` when unused.
    pub stencil: Option<AttachmentOps<u32>>,
}

/// Dynamic raster viewport supplied while recording a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Viewport width in pixels.
    pub width: f32,
    /// Viewport height in pixels.
    pub height: f32,
    /// Minimum depth value.
    pub min_depth: f32,
    /// Maximum depth value.
    pub max_depth: f32,
}

/// Dynamic integer scissor rectangle supplied while recording a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Rectangle width in pixels.
    pub width: u32,
    /// Rectangle height in pixels.
    pub height: u32,
}

/// One declared resource supplied to a renderer-owned opaque binding recipe.
#[derive(Clone, Copy, Debug)]
pub enum BindingResource<'a> {
    /// A declared sampled or read-only texture.
    TextureRead(&'a TextureRead),
    /// A declared writable texture.
    TextureWrite(&'a TextureWrite),
    /// A declared read-modify-write texture.
    TextureReadWrite(&'a TextureReadWrite),
    /// A declared read-only buffer.
    BufferRead(&'a BufferRead),
    /// A declared writable buffer.
    BufferWrite(&'a BufferWrite),
    /// A declared read-modify-write buffer.
    BufferReadWrite(&'a BufferReadWrite),
}

/// A pass-validated opaque binding object ready for one command context.
pub struct ResolvedBindings<'a> {
    pub(crate) ticket: u64,
    pub(crate) session: u64,
    pub(super) _marker: PhantomData<&'a mut ()>,
}
