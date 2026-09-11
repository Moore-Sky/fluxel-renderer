//! Owns validated multi-draw packets without exposing graph or native resources.
//!
//! Lowering copies the borrowed draw-list domain into device-affine packet data;
//! submission later compiles one graph and advances uniform uploads followed by
//! one raster submission. Packet construction never reserves snapshot generations.

use core::fmt;

use fluxel_rendergraph::{CompletionFailure, DeviceIdentity};
use fluxel_rhi::experimental::fixed_artifacts::RasterCreateError;
use fluxel_rhi::{BufferUploadError, NativeExecutionError};

use crate::{Camera, IndexedMeshSnapshot, frame_uniform::FrameUniform};

mod build;
mod graph;
mod ids;
mod preparation;
mod provider;
mod submission;
#[cfg(test)]
mod tests;

pub(super) use graph::{PacketGraph, PacketGraphBuildError};
pub(super) use provider::PacketResources;

/// An owned, device-affine sequence of validated draws.
///
/// Packets have no public constructor and cannot be cloned. They retain ready
/// snapshots and serialized per-draw uniforms, but acquire no in-flight use
/// reservation until [`FixedFrameRenderer::submit_packet`](super::FixedFrameRenderer::submit_packet).
pub struct RenderPacket {
    device: DeviceIdentity,
    extent: [u32; 2],
    _camera: Camera,
    draws: Vec<PacketDraw>,
}

#[derive(Clone)]
pub(super) struct PacketDraw {
    snapshot: IndexedMeshSnapshot,
    uniform: FrameUniform,
}

impl RenderPacket {
    pub(super) const fn device(&self) -> DeviceIdentity {
        self.device
    }

    pub(super) const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub(super) fn draws(&self) -> &[PacketDraw] {
        &self.draws
    }
}

impl PacketDraw {
    pub(super) const fn snapshot(&self) -> &IndexedMeshSnapshot {
        &self.snapshot
    }

    pub(super) const fn uniform(&self) -> &FrameUniform {
        &self.uniform
    }
}

impl fmt::Debug for RenderPacket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderPacket")
            .field("device", &self.device)
            .field("extent", &self.extent)
            .field("draw_count", &self.draws.len())
            .finish_non_exhaustive()
    }
}

/// Why a draw list could not be lowered into an owned packet.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RenderPacketBuildError {
    /// A packet must contain at least one draw.
    EmptyDrawList,
    /// The positional snapshot mapping did not cover every draw exactly once.
    SnapshotCountMismatch {
        /// Number of draw-list entries.
        draws: usize,
        /// Number of supplied snapshots.
        snapshots: usize,
    },
    /// The offscreen target width or height was zero.
    InvalidExtent,
    /// The renderer-private CPU preparation graph could not complete. This is
    /// intentionally distinct from GPU graph compilation and never exposes
    /// slot-graph identities through the renderer API.
    PreparationGraph,
    /// One draw failed validation.
    Draw {
        /// Insertion-order index of the invalid draw.
        index: usize,
        /// Stable reason for rejecting that draw.
        reason: RenderPacketDrawBuildError,
    },
}

/// Why one draw cannot be represented by the closed packet recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RenderPacketDrawBuildError {
    /// The snapshot belongs to another device.
    ForeignSnapshotDevice,
    /// CPU vertex positions differ from the immutable snapshot metadata.
    PositionMetadataMismatch,
    /// CPU indices differ from the immutable snapshot metadata.
    IndexMetadataMismatch,
    /// The closed triangle-list recipe requires a nonzero multiple of three indices.
    InvalidIndexCount,
    /// Camera or material values cannot form the closed uniform ABI.
    InvalidCameraMaterial,
    /// The finite camera and model transform overflowed while forming `P * V * M`.
    ModelTransformProductNonFinite,
    /// A vertex position was not finite.
    NonFinitePosition,
    /// Matrix application produced a non-finite clip coordinate.
    NonFiniteClipPosition,
    /// A clip-space vertex had a non-positive homogeneous w component.
    ClipWNonPositive,
    /// A vertex was outside the portable clip volume.
    ClipOutOfBounds,
}

/// Why packet submission could not start before any queue work was accepted.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RenderPacketStartError {
    /// The packet belongs to another device.
    ForeignPacketDevice,
    /// The acquired presentation image belongs to another device.
    #[cfg(windows)]
    ForeignSurfaceDevice,
    /// The acquired presentation image does not match the packet extent.
    #[cfg(windows)]
    SurfaceExtentMismatch {
        /// Extent validated while lowering the draw list.
        packet: [u32; 2],
        /// Extent supplied by the acquired presentation image.
        surface: [u32; 2],
    },
    /// A unique snapshot generation could not be reserved.
    Reservation {
        /// First draw which references the unavailable generation.
        draw_index: usize,
        /// Whether the generation is busy or permanently poisoned.
        cause: RenderPacketReservationError,
    },
    /// The portable multi-draw graph could not compile.
    Graph(fluxel_rendergraph::CompileError),
    /// Private graph binding identities could not represent the packet size.
    GraphIdentityExhausted,
    /// The closed native raster pipeline could not be created.
    Pipeline(RasterCreateError),
    /// The closed native object registry rejected the recipe.
    Provider(NativeExecutionError),
    /// The first uniform upload was rejected before queue acceptance.
    FirstUniformStart(BufferUploadError),
}

/// Snapshot reservation failures reported at packet start.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RenderPacketReservationError {
    /// Another submission currently owns this generation.
    Busy,
    /// A prior accepted raster submission did not prove completion.
    Poisoned,
}

/// A terminal failure of a packet submission.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RenderPacketFailure {
    /// A later uniform upload was rejected after earlier uploads were accepted.
    UniformStart {
        /// Insertion-order draw index.
        draw_index: usize,
        /// Structured upload-start failure.
        cause: BufferUploadError,
    },
    /// An accepted uniform upload reached terminal failure.
    UniformCompletion {
        /// Insertion-order draw index.
        draw_index: usize,
        /// Structured native completion failure.
        cause: CompletionFailure,
    },
    /// An accepted uniform upload could not be observed safely.
    UniformObservation {
        /// Insertion-order draw index.
        draw_index: usize,
        /// Structured observation failure.
        cause: super::FixedFrameUniformObservationError,
    },
    /// Recording or raster submission was rejected before acceptance.
    RasterStart {
        /// Structured graph/backend cause.
        cause: super::FixedFrameExecutionError,
    },
    /// An accepted raster submission reached terminal failure.
    RasterCompletion {
        /// Structured native completion failure.
        cause: CompletionFailure,
    },
    /// An accepted raster submission could not be observed or exported safely.
    RasterObservation {
        /// Structured observation failure.
        cause: super::FixedFrameRasterObservationError,
    },
}

/// The current non-blocking state of a packet submission.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RenderPacketStatus {
    /// Uniform or raster queue work remains incomplete.
    Pending,
    /// The shared executor is temporarily busy; polling may be retried.
    Busy,
    /// Raster and presentation work was accepted, but completion has not yet
    /// retired the packet's private resources.
    #[cfg(windows)]
    Submitted,
    /// The frame completed and owns its opaque image.
    Complete(super::FrameImage),
    /// The packet completed through an acquired presentable image.
    ///
    /// The acquired image's token is consumed by the single graph execution;
    /// completion is the point at which every draw reservation retires.
    #[cfg(windows)]
    Presented,
    /// The operation reached a terminal failure.
    Failed(RenderPacketFailure),
}

// Kept in a dedicated file because lifecycle state is independently complex.
pub use submission::RenderPacketSubmission;

impl fmt::Display for RenderPacketBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "packet build failed: {self:?}")
    }
}
impl std::error::Error for RenderPacketBuildError {}

impl fmt::Display for RenderPacketStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "packet submission did not start: {self:?}")
    }
}
impl std::error::Error for RenderPacketStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::Pipeline(error) => Some(error),
            Self::Provider(error) => Some(error),
            Self::FirstUniformStart(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for RenderPacketFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "packet submission failed: {self:?}")
    }
}
impl std::error::Error for RenderPacketFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UniformStart { cause, .. } => Some(cause),
            Self::UniformObservation { cause, .. } => Some(cause),
            Self::RasterStart { cause } => Some(cause),
            Self::RasterObservation { cause } => Some(cause),
            _ => None,
        }
    }
}
