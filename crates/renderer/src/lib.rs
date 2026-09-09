//! Headless renderer domain types for Fluxel.
//!
//! This crate models cameras, geometry, meshes, basic materials, and ordered
//! draw lists. Its opt-in `gpu-upload` feature publishes immutable indexed-mesh
//! and RGBA8 texture snapshots after native completion. Its opt-in fixed-frame
//! slice lowers closed constant-color, mip-zero `textureLoad`, or fixed
//! linear-clamp `textureSampleLevel` indexed draws without exposing native
//! resources or a configurable sampler.

#![deny(missing_docs)]

#[cfg(feature = "gpu-upload")]
mod fixed_frame;
#[cfg(feature = "gpu-upload")]
mod frame_uniform;
mod shader;
#[cfg(feature = "gpu-upload")]
mod upload;

use core::fmt;

#[cfg(all(test, windows, feature = "gpu-upload"))]
pub(crate) fn native_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "gpu-upload")]
pub use fixed_frame::{
    DrawStartError, FixedFrameFailure, FixedFrameRenderer, FixedFrameStatus, FixedFrameSubmission,
    FrameImage,
};
#[cfg(feature = "gpu-upload")]
pub use upload::{
    BaseColorTextureSnapshot, BaseColorTextureUpload, BaseColorTextureUploadFailure,
    BaseColorTextureUploadStartError, BaseColorTextureUploadStatus, IndexedMeshSnapshot,
    IndexedMeshUpload, IndexedMeshUploadFailure, IndexedMeshUploadStartError,
    IndexedMeshUploadStatus, Rgba8Image, Rgba8ImageError, TexturedBasicMaterial, TexturedGeometry,
    TexturedGeometryError, TexturedGeometryStream, TexturedIndexedMeshSnapshot,
    TexturedIndexedMeshUpload, TexturedIndexedMeshUploadFailure,
    TexturedIndexedMeshUploadStartError, TexturedIndexedMeshUploadStatus,
};

/// A camera described by view and projection matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

impl Camera {
    /// Creates a camera from column-major view and projection matrices.
    #[must_use]
    pub const fn new(view: [[f32; 4]; 4], projection: [[f32; 4]; 4]) -> Self {
        Self { view, projection }
    }

    /// Returns the camera view matrix.
    #[must_use]
    pub const fn view(&self) -> &[[f32; 4]; 4] {
        &self.view
    }

    /// Returns the camera projection matrix.
    #[must_use]
    pub const fn projection(&self) -> &[[f32; 4]; 4] {
        &self.projection
    }
}

impl Default for Camera {
    fn default() -> Self {
        const IDENTITY: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self::new(IDENTITY, IDENTITY)
    }
}

/// Vertex positions and optional triangle indices for one mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl Geometry {
    /// Creates non-indexed geometry from vertex positions.
    #[must_use]
    pub fn from_positions(positions: impl Into<Vec<[f32; 3]>>) -> Self {
        Self {
            positions: positions.into(),
            indices: Vec::new(),
        }
    }

    /// Adds indices after checking that every index refers to a vertex.
    pub fn with_indices(mut self, indices: impl Into<Vec<u32>>) -> Result<Self, GeometryError> {
        let indices = indices.into();
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.positions.len())
        {
            return Err(GeometryError::IndexOutOfBounds {
                index,
                vertex_count: self.positions.len(),
            });
        }
        self.indices = indices;
        Ok(self)
    }

    /// Returns the vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// Returns the triangle indices, or an empty slice for non-indexed geometry.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Reports whether this geometry uses an index buffer.
    #[must_use]
    pub fn is_indexed(&self) -> bool {
        !self.indices.is_empty()
    }
}

/// An invalid geometry construction request.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GeometryError {
    /// An index referred to a vertex outside the position array.
    IndexOutOfBounds {
        /// The invalid index.
        index: u32,
        /// The number of available positions.
        vertex_count: usize,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                formatter,
                "geometry index {index} is outside its {vertex_count} vertex positions"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

/// A minimal unlit material with a linear RGBA base color.
#[derive(Clone, Debug, PartialEq)]
pub struct BasicMaterial {
    base_color: [f32; 4],
}

impl BasicMaterial {
    /// Creates a basic material with the supplied linear RGBA base color.
    #[must_use]
    pub const fn new(base_color: [f32; 4]) -> Self {
        Self { base_color }
    }

    /// Returns the linear RGBA base color.
    #[must_use]
    pub const fn base_color(&self) -> [f32; 4] {
        self.base_color
    }
}

impl Default for BasicMaterial {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0])
    }
}

/// A renderable geometry and material pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    geometry: Geometry,
    material: BasicMaterial,
}

impl Mesh {
    /// Creates a mesh from geometry and a basic material.
    #[must_use]
    pub fn new(geometry: Geometry, material: BasicMaterial) -> Self {
        Self { geometry, material }
    }

    /// Returns this mesh's geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Returns this mesh's material.
    #[must_use]
    pub const fn material(&self) -> &BasicMaterial {
        &self.material
    }
}

/// A borrowed mesh scheduled for a future renderer submission.
#[derive(Clone, Copy, Debug)]
pub struct DrawItem<'a> {
    mesh: &'a Mesh,
}

impl<'a> DrawItem<'a> {
    /// Returns the mesh selected by this draw item.
    #[must_use]
    pub const fn mesh(self) -> &'a Mesh {
        self.mesh
    }
}

/// An insertion-ordered list of meshes to draw for one camera.
#[derive(Clone, Debug)]
pub struct DrawList<'a> {
    camera: &'a Camera,
    items: Vec<DrawItem<'a>>,
}

impl<'a> DrawList<'a> {
    /// Creates an empty draw list for `camera`.
    #[must_use]
    pub const fn new(camera: &'a Camera) -> Self {
        Self {
            camera,
            items: Vec::new(),
        }
    }

    /// Returns the camera supplied when this list was created.
    #[must_use]
    pub const fn camera(&self) -> &'a Camera {
        self.camera
    }

    /// Appends `mesh` after all existing items.
    pub fn push(&mut self, mesh: &'a Mesh) {
        self.items.push(DrawItem { mesh });
    }

    /// Returns the scheduled items in submission order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = DrawItem<'a>> + '_ {
        self.items.iter().copied()
    }

    /// Returns the number of scheduled items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Reports whether no meshes are scheduled.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_rejects_an_out_of_bounds_index() {
        let error = Geometry::from_positions(vec![[0.0, 0.0, 0.0]])
            .with_indices(vec![1])
            .unwrap_err();
        assert_eq!(
            error,
            GeometryError::IndexOutOfBounds {
                index: 1,
                vertex_count: 1,
            }
        );
    }

    #[test]
    fn draw_list_preserves_submission_order() {
        let camera = Camera::default();
        let first = Mesh::new(
            Geometry::from_positions(vec![[0.0, 0.0, 0.0]]),
            BasicMaterial::default(),
        );
        let second = Mesh::new(
            Geometry::from_positions(vec![[1.0, 0.0, 0.0]]),
            BasicMaterial::default(),
        );
        let mut list = DrawList::new(&camera);
        list.push(&first);
        list.push(&second);

        assert_eq!(
            list.iter()
                .map(|item| item.mesh().geometry().positions()[0])
                .collect::<Vec<_>>(),
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        );
    }
}
