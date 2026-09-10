//! Coordinates the deliberately closed headless indexed-frame rendering slice.
//!
//! This boundary owns fixed-frame graph declarations, closed raster recipes,
//! draw-start validation, and the two-phase submission lifecycle. It does not
//! expose native resources, configurable pipelines, or general scheduling:
//! those remain RHI and RenderGraph responsibilities. Recipes select the
//! graph and binding contract; the renderer validates and starts work; the
//! submission module preserves the accepted-versus-pre-accept lifecycle and
//! releases or poisons snapshot reservations accordingly.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    AttachmentOps, BindingResource, BindingSetId, BoundBuffer, BufferBindingId, BufferRange,
    ColorAttachmentDesc, CompletionFailure, CompletionStatus, DeviceCapabilities, DeviceIdentity,
    ExecutionError, ExportBufferContract, ExportTextureContract, Extent3d, ExternalOwnership,
    FrameBindingError, FrameBindingErrorKind, FrameInputs, FrameResourceProvider,
    ImportBufferContract, ImportTextureContract, IndexFormat, InitialContents, LoadOp,
    RasterPipelineId, RenderGraph, ResourceAccessState, StoreOp, TextureBindingId, TextureDesc,
    TextureDimension, TextureFormat, TextureRange, TextureReadUse, Viewport, WriteCoverage,
};
use fluxel_rhi::{
    Buffer, BufferDescriptor, BufferUploadError, Device, MemoryPolicy, PendingBufferUpload,
    RasterBackend, RasterKernel, RasterObjectProvider, ResourceLease, Texture, UploadedBuffer,
};

use crate::upload::{SnapshotDrawReservation, SnapshotUseError};
use crate::{
    BaseColorTextureSnapshot, IndexedMeshSnapshot, NormalIndexedMeshSnapshot,
    SrgbBaseColorTextureSnapshot, SrgbTexturedBasicMaterial, TexturedBasicMaterial,
    TexturedIndexedMeshSnapshot,
    frame_uniform::{FRAME_UNIFORM_BYTES, FrameUniform},
};

#[cfg(all(test, windows))]
fn native_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    crate::native_fixture_guard()
}

#[cfg(all(test, windows))]
fn actual_raster_capabilities(device: &Device) -> DeviceCapabilities {
    let backend = RasterBackend::new(device.clone());
    fluxel_rendergraph::ExecutionBackend::capabilities(&backend).clone()
}

/// Fixed-frame graph declarations and resource provider bindings.
mod graph;
/// Fixed identifiers shared by closed recipes and graph declarations.
mod ids;
/// Owned multi-draw packets and their non-blocking submission lifecycle.
mod packet;
/// Caller-owned snapshot bindings for graph imports.
mod provider;
/// Closed fixed raster-contract mappings.
mod recipe;
/// Fixed-frame renderer construction and draw-start validation.
mod renderer;
/// Fixed-frame two-phase submission completion lifecycle.
mod submission;
#[cfg(test)]
/// CPU and legacy fixed-frame conformance fixtures.
mod tests;

pub use packet::{
    RenderPacket, RenderPacketBuildError, RenderPacketDrawBuildError, RenderPacketExecutionError,
    RenderPacketFailure, RenderPacketRasterObservationError, RenderPacketReservationError,
    RenderPacketStartError, RenderPacketStatus, RenderPacketSubmission,
    RenderPacketUniformObservationError,
};
pub use renderer::{DrawStartError, FixedFrameFailure, FixedFrameRenderer};
pub use submission::{FixedFrameStatus, FixedFrameSubmission, FrameImage};

use graph::{
    CameraGraph, build_camera_graph, build_normal_lambert_camera_graph,
    build_textured_camera_graph, build_uv_textured_camera_graph,
};
#[cfg(all(test, windows))]
use graph::{FixedGraph, build_graph};
use ids::*;
use provider::CameraResources;
use recipe::*;
#[cfg(all(test, windows))]
use renderer::UvStartRequest;
use renderer::{FrameTexture, FrameTextureSnapshot};
#[cfg(all(test, windows))]
use submission::SnapshotResources;
use submission::{CameraPhase, FrameMeshSnapshot, missing_binding};
