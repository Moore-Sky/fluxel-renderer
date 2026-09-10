//! CPU-side geometry, encoded-color, and tint contracts for vertex-color draws.

use core::fmt;

use crate::Geometry;

/// A logical stream in the closed vertex-color mesh ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorGeometryStream {
    /// Tightly packed `f32x3` positions.
    Position,
    /// Raw encoded linear `UNORM8x4` colors.
    Color,
    /// Tightly packed `u32` triangle indices.
    Index,
}

/// Why [`VertexColorGeometry`] cannot be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorGeometryError {
    /// There are no vertex positions.
    EmptyPositions,
    /// The geometry has no index stream.
    MissingIndices,
    /// The index stream cannot describe complete triangles.
    NonTriangleIndexCount {
        /// Number of supplied indices.
        index_count: usize,
    },
    /// Color and position streams have differing vertex counts.
    ColorCountMismatch {
        /// Number of positions.
        positions: usize,
        /// Number of colors.
        colors: usize,
    },
    /// A stream count cannot fit the closed ABI.
    CountOutOfRange {
        /// Stream with the oversized count.
        stream: VertexColorGeometryStream,
        /// Original count.
        count: usize,
    },
    /// A stream byte length cannot be represented exactly.
    ByteLengthOverflow {
        /// Stream whose byte length overflowed.
        stream: VertexColorGeometryStream,
    },
}
impl fmt::Display for VertexColorGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => f.write_str("vertex-color geometry positions are empty"),
            Self::MissingIndices => f.write_str("vertex-color geometry has no indices"),
            Self::NonTriangleIndexCount { index_count } => write!(
                f,
                "vertex-color geometry has {index_count} indices, not a whole number of triangles"
            ),
            Self::ColorCountMismatch { positions, colors } => write!(
                f,
                "vertex-color geometry has {colors} colors for {positions} positions"
            ),
            Self::CountOutOfRange { stream, count } => write!(
                f,
                "vertex-color geometry {stream:?} count {count} exceeds the closed upload ABI"
            ),
            Self::ByteLengthOverflow { stream } => write!(
                f,
                "vertex-color geometry {stream:?} payload length overflows"
            ),
        }
    }
}
impl std::error::Error for VertexColorGeometryError {}

/// Indexed geometry with one encoded linear RGBA8 color per position.
#[derive(Clone, Debug, PartialEq)]
pub struct VertexColorGeometry {
    geometry: Geometry,
    colors: Vec<[u8; 4]>,
}
impl VertexColorGeometry {
    /// Validates the closed position/color/index stream shape.
    pub fn new(
        geometry: Geometry,
        colors: impl Into<Vec<[u8; 4]>>,
    ) -> Result<Self, VertexColorGeometryError> {
        let colors = colors.into();
        if geometry.positions().is_empty() {
            return Err(VertexColorGeometryError::EmptyPositions);
        }
        if !geometry.is_indexed() {
            return Err(VertexColorGeometryError::MissingIndices);
        }
        if !geometry.indices().len().is_multiple_of(3) {
            return Err(VertexColorGeometryError::NonTriangleIndexCount {
                index_count: geometry.indices().len(),
            });
        }
        if colors.len() != geometry.positions().len() {
            return Err(VertexColorGeometryError::ColorCountMismatch {
                positions: geometry.positions().len(),
                colors: colors.len(),
            });
        }
        VertexColorMeshPayload::validate_lengths(&geometry, &colors)?;
        Ok(Self { geometry, colors })
    }
    /// Returns the owned indexed geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    /// Returns encoded linear RGBA8 colors exactly as they will be uploaded.
    #[must_use]
    pub fn colors(&self) -> &[[u8; 4]] {
        &self.colors
    }
}

/// Why [`VertexColorMaterial`] cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorMaterialError {
    /// A tint component was NaN or infinite.
    NonFinite {
        /// Zero-based RGBA component index.
        component: usize,
    },
    /// A finite tint component was outside the unit interval.
    OutOfRange {
        /// Zero-based RGBA component index.
        component: usize,
    },
}
impl fmt::Display for VertexColorMaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { component } => {
                write!(f, "vertex-color tint component {component} is non-finite")
            }
            Self::OutOfRange { component } => write!(
                f,
                "vertex-color tint component {component} is outside [0, 1]"
            ),
        }
    }
}
impl std::error::Error for VertexColorMaterialError {}

/// A finite, linear RGBA tint for the closed vertex-color recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexColorMaterial {
    tint: [f32; 4],
}
impl VertexColorMaterial {
    /// Creates a material after checking each linear tint component is finite and in `[0, 1]`.
    pub fn new(tint: [f32; 4]) -> Result<Self, VertexColorMaterialError> {
        for (component, value) in tint.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(VertexColorMaterialError::NonFinite { component });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(VertexColorMaterialError::OutOfRange { component });
            }
        }
        Ok(Self { tint })
    }
    /// Returns the validated linear RGBA tint.
    #[must_use]
    pub const fn tint(&self) -> &[f32; 4] {
        &self.tint
    }
}

/// Serialization used by the immutable uploader; validated geometry makes this fallible only for defensive reuse.
pub(crate) struct VertexColorMeshPayload {
    pub(crate) positions: Vec<u8>,
    pub(crate) colors: Vec<u8>,
    pub(crate) indices: Vec<u8>,
    pub(crate) position_count: u32,
    pub(crate) index_count: u32,
}
impl VertexColorMeshPayload {
    pub(crate) fn validate_lengths(
        geometry: &Geometry,
        colors: &[[u8; 4]],
    ) -> Result<(), VertexColorGeometryError> {
        for (stream, count, stride) in [
            (
                VertexColorGeometryStream::Position,
                geometry.positions().len(),
                12usize,
            ),
            (VertexColorGeometryStream::Color, colors.len(), 4),
            (
                VertexColorGeometryStream::Index,
                geometry.indices().len(),
                4,
            ),
        ] {
            u32::try_from(count)
                .map_err(|_| VertexColorGeometryError::CountOutOfRange { stream, count })?;
            let bytes = count
                .checked_mul(stride)
                .ok_or(VertexColorGeometryError::ByteLengthOverflow { stream })?;
            u64::try_from(bytes)
                .map_err(|_| VertexColorGeometryError::ByteLengthOverflow { stream })?;
        }
        Ok(())
    }
    pub(crate) fn from_geometry(
        geometry: &VertexColorGeometry,
    ) -> Result<Self, VertexColorGeometryError> {
        Self::validate_lengths(&geometry.geometry, &geometry.colors)?;
        let mut positions = Vec::with_capacity(geometry.geometry.positions().len() * 12);
        for position in geometry.geometry.positions() {
            for value in position {
                positions.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        let mut indices = Vec::with_capacity(geometry.geometry.indices().len() * 4);
        for index in geometry.geometry.indices() {
            indices.extend_from_slice(&index.to_le_bytes());
        }
        let colors = geometry.colors.iter().flatten().copied().collect();
        Ok(Self {
            positions,
            colors,
            indices,
            position_count: u32::try_from(geometry.geometry.positions().len()).map_err(|_| {
                VertexColorGeometryError::CountOutOfRange {
                    stream: VertexColorGeometryStream::Position,
                    count: geometry.geometry.positions().len(),
                }
            })?,
            index_count: u32::try_from(geometry.geometry.indices().len()).map_err(|_| {
                VertexColorGeometryError::CountOutOfRange {
                    stream: VertexColorGeometryStream::Index,
                    count: geometry.geometry.indices().len(),
                }
            })?,
        })
    }
}
