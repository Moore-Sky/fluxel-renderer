//! Public closed-recipe draw entry points.
//!
//! Each entry point validates its recipe-specific input before acquiring a
//! snapshot reservation. `start` owns the common transition into submission.

use super::*;

impl FixedFrameRenderer {
    /// Creates a renderer over an already-opened headless device.
    #[must_use]
    pub fn new(device: Device) -> Self {
        let backend = RasterBackend::new(device.clone());
        let capabilities = fluxel_rendergraph::ExecutionBackend::capabilities(&backend).clone();
        Self {
            executor: Arc::new(fluxel_rendergraph::FrameExecutor::new(backend)),
            capabilities,
            device,
        }
    }

    /// Starts one fixed indexed draw without waiting for the GPU.
    pub fn draw(
        &self,
        snapshot: &IndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &crate::BasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        validate_indexed_input(snapshot, &self.device, extent)?;
        let uniform = FrameUniform::new(camera, material)
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        let graph = Arc::new(
            build_camera_graph(snapshot, extent, &self.capabilities)
                .map_err(DrawStartError::Graph)?,
        );
        self.start_camera(snapshot, graph, uniform)
    }

    /// Starts the closed object-space `+Z` Lambert draw over explicit unit normals.
    pub fn draw_lambert(
        &self,
        snapshot: &NormalIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &crate::BasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
            || snapshot.normals().buffer().device_identity() != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        debug_assert_eq!(
            snapshot.normal_metadata().len(),
            snapshot.position_count() as usize,
            "NormalIndexedMeshSnapshot publication keeps both vertex streams aligned"
        );
        let uniform = FrameUniform::new(camera, material)
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        validate_clip(
            snapshot.position_metadata(),
            snapshot.index_metadata(),
            uniform.view_projection(),
        )?;
        let graph = Arc::new(
            build_normal_lambert_camera_graph(snapshot, extent, &self.capabilities)
                .map_err(DrawStartError::Graph)?,
        );
        let reservation = snapshot.reserve_for_draw().map_err(map_snapshot_use)?;
        self.begin_normal_lambert(snapshot, graph, uniform, reservation)
    }

    /// Starts the closed perspective-interpolated RGBA8 vertex-color draw.
    pub fn draw_vertex_color(
        &self,
        snapshot: &VertexColorIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &VertexColorMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.colors().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        debug_assert_eq!(
            snapshot.color_metadata().len(),
            snapshot.position_count() as usize,
            "VertexColorIndexedMeshSnapshot publication keeps vertex streams aligned"
        );
        let uniform = FrameUniform::new_vertex_color(camera, material)
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        validate_clip(
            snapshot.position_metadata(),
            snapshot.index_metadata(),
            uniform.view_projection(),
        )?;
        let graph = Arc::new(
            build_vertex_color_camera_graph(snapshot, extent, &self.capabilities)
                .map_err(DrawStartError::Graph)?,
        );
        let reservation = snapshot.reserve_for_draw().map_err(map_snapshot_use)?;
        self.begin_vertex_color(snapshot, graph, uniform, reservation)
    }

    /// Starts one fixed indexed draw with an immutable RGBA8 `textureLoad` texture.
    pub fn draw_textured(
        &self,
        snapshot: &IndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &TexturedBasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
            || material
                .base_color_texture()
                .texture()
                .texture()
                .device_identity()
                != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        let uniform = FrameUniform::new(camera, material.material())
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        validate_clip(
            snapshot.position_metadata(),
            snapshot.index_metadata(),
            uniform.view_projection(),
        )?;
        let graph = Arc::new(
            build_textured_camera_graph(
                snapshot,
                material.base_color_texture(),
                extent,
                &self.capabilities,
            )
            .map_err(DrawStartError::Graph)?,
        );
        let (mesh_reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || material.base_color_texture().reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(map_pair_error)?;
        self.start_textured(
            snapshot,
            material.base_color_texture(),
            graph,
            uniform,
            mesh_reservation,
            texture_reservation,
        )
    }

    /// Starts one fixed indexed draw using explicit per-vertex `f32x2` UVs.
    pub fn draw_textured_uv(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &TexturedBasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.draw_uv(
            snapshot,
            camera,
            material,
            extent,
            RasterRecipe::UV_TEXTURE_LOAD,
        )
    }

    /// Starts the closed explicit-UV, linear-filtering, clamp-to-edge sampler draw.
    pub fn draw_textured_uv_linear_clamp(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &TexturedBasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if !rgba8_unorm_filterable(&self.capabilities) {
            return Err(DrawStartError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            });
        }
        self.draw_uv(
            snapshot,
            camera,
            material,
            extent,
            RasterRecipe::UV_LINEAR_CLAMP_UNORM,
        )
    }

    /// Starts the closed explicit-UV sRGB decode, linear-filtering sampler draw.
    pub fn draw_textured_uv_linear_clamp_srgb(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &SrgbTexturedBasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if !rgba8_unorm_srgb_filterable(&self.capabilities) {
            return Err(DrawStartError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8UnormSrgb,
            });
        }
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
            || snapshot.texture_coordinates().buffer().device_identity() != self.device.identity()
            || material
                .base_color_texture()
                .texture()
                .texture()
                .device_identity()
                != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        let uniform = FrameUniform::new(camera, material.material())
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        validate_textured_clip(
            snapshot.position_metadata(),
            snapshot.index_metadata(),
            snapshot.texture_coordinate_metadata(),
            uniform.view_projection(),
        )?;
        let texture = FrameTextureSnapshot::Srgb(material.base_color_texture().clone());
        let graph = Arc::new(
            build_uv_textured_camera_graph(
                snapshot,
                &texture,
                extent,
                &self.capabilities,
                RasterRecipe::UV_LINEAR_CLAMP_SRGB,
            )
            .map_err(DrawStartError::Graph)?,
        );
        let (reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || texture.reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(map_pair_error)?;
        self.start_uv_with_recipe(
            snapshot,
            texture,
            UvStartRequest {
                graph,
                uniform,
                reservation,
                texture_reservation,
                recipe: RasterRecipe::UV_LINEAR_CLAMP_SRGB,
            },
        )
    }

    fn draw_uv(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &TexturedBasicMaterial,
        extent: [u32; 2],
        recipe: RasterRecipe,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
            || snapshot.texture_coordinates().buffer().device_identity() != self.device.identity()
            || material
                .base_color_texture()
                .texture()
                .texture()
                .device_identity()
                != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        let uniform = FrameUniform::new(camera, material.material())
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        validate_textured_clip(
            snapshot.position_metadata(),
            snapshot.index_metadata(),
            snapshot.texture_coordinate_metadata(),
            uniform.view_projection(),
        )?;
        let texture = FrameTextureSnapshot::Linear(material.base_color_texture().clone());
        let graph = Arc::new(
            build_uv_textured_camera_graph(snapshot, &texture, extent, &self.capabilities, recipe)
                .map_err(DrawStartError::Graph)?,
        );
        let (reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || texture.reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(map_pair_error)?;
        self.start_uv_with_recipe(
            snapshot,
            texture,
            UvStartRequest {
                graph,
                uniform,
                reservation,
                texture_reservation,
                recipe,
            },
        )
    }
}

fn validate_indexed_input(
    snapshot: &IndexedMeshSnapshot,
    device: &Device,
    extent: [u32; 2],
) -> Result<(), DrawStartError> {
    if extent[0] == 0 || extent[1] == 0 {
        return Err(DrawStartError::InvalidExtent);
    }
    if snapshot.positions().buffer().device_identity() != device.identity()
        || snapshot.indices().buffer().device_identity() != device.identity()
    {
        return Err(DrawStartError::ForeignSnapshotDevice);
    }
    if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
        return Err(DrawStartError::InvalidIndexCount);
    }
    Ok(())
}
