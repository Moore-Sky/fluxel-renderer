//! Structured admission and terminal-failure errors for fixed-frame work.

use super::*;

/// Why a fixed frame could not be accepted for submission.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DrawStartError {
    /// The selected device cannot filter the fixed RGBA8 sampler texture.
    TextureFormatNotFilterable {
        /// The required fixed texture format.
        format: TextureFormat,
    },
    /// The requested target dimensions contain zero.
    InvalidExtent,
    /// The ready snapshot belongs to another native device.
    ForeignSnapshotDevice,
    /// This fixed triangle-list recipe requires a nonzero multiple of three indices.
    InvalidIndexCount,
    /// Another draw is active for this snapshot generation.
    SnapshotInFlight,
    /// A prior accepted draw did not prove its terminal state.
    SnapshotPoisoned,
    /// Another draw is active for the immutable base-color texture generation.
    TextureInFlight,
    /// A prior accepted texture draw did not prove its terminal state.
    TexturePoisoned,
    /// Camera/material values cannot produce the closed uniform ABI.
    InvalidCameraMaterial,
    /// A vertex position was NaN or infinite during CPU clip validation.
    NonFinitePosition,
    /// Matrix application produced a NaN or infinite clip coordinate.
    NonFiniteClipPosition,
    /// A clip-space vertex has a non-positive homogeneous w component.
    ClipWNonPositive,
    /// A clip-space vertex lies outside the closed D3D/WebGPU clip volume.
    ClipOutOfBounds,
    /// A referenced texture coordinate was NaN or infinite.
    NonFiniteTextureCoordinate,
    /// The immutable uniform upload was rejected before work was accepted.
    UniformStart(BufferUploadError),
    /// Graph compilation rejected the fixed declaration.
    Graph(String),
    /// The device could not create the closed raster artifact.
    Pipeline(String),
}
impl fmt::Display for DrawStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextureFormatNotFilterable { format } => write!(
                f,
                "fixed sampler texture format is not filterable: {format:?}"
            ),
            Self::InvalidExtent => f.write_str("fixed frame extent must be nonzero"),
            Self::ForeignSnapshotDevice => f.write_str("snapshot belongs to another device"),
            Self::InvalidIndexCount => {
                f.write_str("fixed frame needs a nonzero triangle-list index count")
            }
            Self::SnapshotInFlight => f.write_str("snapshot draw is already in flight"),
            Self::SnapshotPoisoned => f.write_str("snapshot draw state is poisoned"),
            Self::TextureInFlight => f.write_str("base-color texture draw is already in flight"),
            Self::TexturePoisoned => f.write_str("base-color texture draw state is poisoned"),
            Self::InvalidCameraMaterial => f.write_str("invalid camera or material uniform input"),
            Self::NonFinitePosition => f.write_str("mesh position is not finite"),
            Self::NonFiniteClipPosition => f.write_str("clip position is not finite"),
            Self::ClipWNonPositive => f.write_str("clip position has non-positive w"),
            Self::ClipOutOfBounds => f.write_str("mesh is outside the fixed clip volume"),
            Self::NonFiniteTextureCoordinate => {
                f.write_str("mesh texture coordinate is not finite")
            }
            Self::UniformStart(error) => write!(f, "frame uniform upload did not start: {error}"),
            Self::Graph(error) => write!(f, "fixed frame graph rejected: {error}"),
            Self::Pipeline(error) => write!(f, "fixed frame pipeline rejected: {error}"),
        }
    }
}
impl std::error::Error for DrawStartError {}

/// A terminal failure of the two-phase fixed frame operation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FixedFrameFailure {
    /// The uniform upload reached a terminal GPU failure.
    UniformCompletion(CompletionFailure),
    /// Uniform completion could not be observed or finalized.
    UniformObservation(String),
    /// Raster recording or submit was rejected before raster work was accepted.
    RasterStart(String),
    /// An accepted raster submission did not complete successfully.
    RasterCompletion(CompletionFailure),
}
impl fmt::Display for FixedFrameFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UniformCompletion(error) => write!(f, "uniform completion failed: {error:?}"),
            Self::UniformObservation(error) => {
                write!(f, "uniform completion observation failed: {error}")
            }
            Self::RasterStart(error) => write!(f, "raster did not start: {error}"),
            Self::RasterCompletion(error) => write!(f, "raster completion failed: {error:?}"),
        }
    }
}
impl std::error::Error for FixedFrameFailure {}
