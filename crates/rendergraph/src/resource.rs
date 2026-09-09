//! Logical resource creation, import, export, and presentation contracts.

use crate::{
    handles::{BufferVersion, ImportBufferSlot, ImportTextureSlot, TextureVersion},
    rhi::{BufferDesc, ExternalOwnership, ResourceAccessState, TextureDesc},
};

/// A retained external-texture slot and its graph-visible initial version.
#[derive(Debug)]
pub struct ImportedTexture {
    /// Stable key used by a future per-frame binding API.
    pub slot: ImportTextureSlot,
    /// Initial logical version used by pass declarations.
    pub version: TextureVersion,
}

/// A retained external-buffer slot and its graph-visible initial version.
#[derive(Debug)]
pub struct ImportedBuffer {
    /// Stable key used by a future per-frame binding API.
    pub slot: ImportBufferSlot,
    /// Initial logical version used by pass declarations.
    pub version: BufferVersion,
}

/// Static requirements for an imported texture slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportTextureContract {
    /// Texture shape and format required by the compiled graph.
    pub descriptor: TextureDesc,
    /// Incoming state expected from the frame binding.
    pub initial_state: ResourceAccessState,
    /// Owner that must retain the physical texture while the frame is in flight.
    pub ownership: ExternalOwnership,
    /// Whether the imported resource begins with defined contents.
    pub initial_contents: InitialContents,
}

/// Static requirements for an acquired presentation image slot.
///
/// Surface ownership and acquisition state are supplied by the surface adapter,
/// not selected by graph authoring code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceTextureContract {
    /// Texture shape and format required by the compiled graph.
    pub descriptor: TextureDesc,
}

/// Static requirements for an imported buffer slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportBufferContract {
    /// Buffer size required by the compiled graph.
    pub descriptor: BufferDesc,
    /// Incoming state expected from the frame binding.
    pub initial_state: ResourceAccessState,
    /// Owner that must retain the physical buffer while the frame is in flight.
    pub ownership: ExternalOwnership,
    /// Whether the imported resource begins with defined contents.
    pub initial_contents: InitialContents,
}

/// Initial content validity promised by an ordinary imported resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InitialContents {
    /// Every element described by the import contract contains defined data.
    Defined,
    /// Contents are not valid until initialized by graph work.
    Undefined,
}

/// Required outgoing state for an exported texture version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportTextureContract {
    /// State required after the graph's last access.
    pub final_state: ResourceAccessState,
}

/// Required outgoing state for an exported buffer version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportBufferContract {
    /// State required after the graph's last access.
    pub final_state: ResourceAccessState,
}

/// Contract marking a surface-owned texture version as a presentation root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentContract {
    _private: (),
}

impl PresentContract {
    /// Creates the portable presentation contract.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for PresentContract {
    fn default() -> Self {
        Self::new()
    }
}
