//! Private graph declaration model shared by authoring and compilation.

use crate::{
    access::{
        AccessMode, BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureRange,
        TextureReadUse, TextureReadWriteUse, TextureWriteUse, WriteCoverage,
    },
    handles::{ImportBufferSlot, ImportTextureSlot, ResourceId},
    pass::{ColorAttachmentDesc, DepthStencilAttachmentDesc},
    resource::{ImportBufferContract, ImportTextureContract, SurfaceTextureContract},
    rhi::{BufferDesc, TextureDesc},
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ResourceKind {
    Texture(TextureDesc),
    Buffer(BufferDesc),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ResourceOrigin {
    Transient,
    TextureImport(ImportTextureSlot, ImportTextureContract),
    Surface(ImportTextureSlot, SurfaceTextureContract),
    BufferImport(ImportBufferSlot, ImportBufferContract),
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceDecl {
    pub id: ResourceId,
    pub name: String,
    pub kind: ResourceKind,
    pub origin: ResourceOrigin,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DeclRange {
    Texture(TextureRange),
    Buffer(BufferRange),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AccessSemantic {
    TextureRead(TextureReadUse),
    TextureWrite(TextureWriteUse),
    TextureReadWrite(TextureReadWriteUse),
    BufferRead(BufferReadUse),
    BufferWrite(BufferWriteUse),
    BufferReadWrite(BufferReadWriteUse),
    ColorAttachment { index: u32 },
    DepthStencilAttachment { depth: bool, stencil: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AccessDetails {
    None,
    ColorAttachment(ColorAttachmentDesc),
    DepthStencilAttachment(DepthStencilAttachmentDesc),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AccessDecl {
    pub handle: u64,
    pub resource: ResourceId,
    pub input_version: u32,
    pub output_version: Option<u32>,
    pub mode: AccessMode,
    pub range: DeclRange,
    pub semantic: AccessSemantic,
    pub details: AccessDetails,
    pub read_required: bool,
    pub coverage: WriteCoverage,
    pub invalidate_before: bool,
    pub discard_after: bool,
}
