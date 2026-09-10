//! Validates fixed draw inputs and starts one closed offscreen raster transaction.
//!
//! This is the pre-accept boundary: it selects an audited recipe, acquires all
//! snapshot reservations atomically, builds matching graph resources, and hands
//! lifecycle ownership to `submission`. It does not expose general pipeline
//! construction or interpret completion after native submission.

/// Coordinates the fixed headless `f32x3/u32` indexed draw slice.
///
/// This type owns no application-visible native resource handles.  It accepts
/// only a completed [`IndexedMeshSnapshot`] and produces opaque image metadata
/// after non-blocking completion observation.
pub struct FixedFrameRenderer {
    pub(in crate::fixed_frame) device: Device,
    pub(in crate::fixed_frame) capabilities: DeviceCapabilities,
    pub(in crate::fixed_frame) executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
}

/// The two deliberately disjoint texture domains accepted by fixed-frame
/// lowering. Keeping the format in this enum makes it impossible for the sRGB
/// flavor to accidentally import a linear snapshot (or vice versa).
#[derive(Clone)]
pub(super) enum FrameTextureSnapshot {
    Linear(BaseColorTextureSnapshot),
    Srgb(SrgbBaseColorTextureSnapshot),
}

impl FrameTextureSnapshot {
    pub(super) fn texture(&self) -> &fluxel_rhi::UploadedTexture {
        match self {
            Self::Linear(snapshot) => snapshot.texture(),
            Self::Srgb(snapshot) => snapshot.texture(),
        }
    }

    pub(super) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        match self {
            Self::Linear(snapshot) => snapshot.reserve_for_draw(),
            Self::Srgb(snapshot) => snapshot.reserve_for_draw(),
        }
    }
}

/// Private shared view used only while declaring an imported graph resource.
/// It does not erase the domain stored by `FrameTextureSnapshot` in an
/// accepted submission.
pub(super) trait FrameTexture {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture;
}

impl FrameTexture for FrameTextureSnapshot {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture {
        self.texture()
    }
}

impl FrameTexture for BaseColorTextureSnapshot {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture {
        self.texture()
    }
}

/// Private accepted-draw inputs kept together so the lifecycle-bearing
/// reservations cannot be accidentally separated from their graph and ABI.
pub(super) struct StartResources {
    snapshot: FrameMeshSnapshot,
    texture: Option<FrameTextureSnapshot>,
    graph: Arc<CameraGraph>,
    uniform: FrameUniform,
    reservation: SnapshotDrawReservation,
    texture_reservation: Option<SnapshotDrawReservation>,
    recipe: RasterRecipe,
}

/// UV-specific front-end inputs before they are normalized into `StartResources`.
pub(super) struct UvStartRequest {
    pub(in crate::fixed_frame) graph: Arc<CameraGraph>,
    pub(in crate::fixed_frame) uniform: FrameUniform,
    pub(in crate::fixed_frame) reservation: SnapshotDrawReservation,
    pub(in crate::fixed_frame) texture_reservation: SnapshotDrawReservation,
    pub(in crate::fixed_frame) recipe: RasterRecipe,
}

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
    ///
    /// The snapshot generation is reserved until this returned operation proves
    /// completion.  A graph rejection before native submission releases the
    /// reservation; only accepted but unproven Raster work poisons it on drop.
    pub fn draw(
        &self,
        snapshot: &IndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &crate::BasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        let uniform = FrameUniform::new(camera, material)
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        let graph = Arc::new(
            build_camera_graph(snapshot, extent, &self.capabilities)
                .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        self.start_camera(snapshot, graph, uniform)
    }

    /// Starts the closed object-space `+Z` Lambert draw over explicit unit
    /// normals. The normal snapshot uses one generation gate for all three
    /// streams, so an accepted draw conservatively owns position, index, and
    /// normal lifetime together.
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
                .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        let reservation = snapshot.reserve_for_draw().map_err(map_snapshot_use)?;
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::Normal(snapshot.clone()),
            texture: None,
            graph,
            uniform,
            reservation,
            texture_reservation: None,
            recipe: RasterRecipe::NORMAL_LAMBERT,
        })
    }

    /// Starts one fixed indexed draw with one immutable RGBA8 texture accessed by fixed integer `textureLoad`.
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
            .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        let (mesh_reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || material.base_color_texture().reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(|error| match error {
            PairReservationError::First(error) => map_snapshot_use(error),
            PairReservationError::Second(error) => map_texture_use(error),
        })?;
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
            build_uv_textured_camera_graph(
                snapshot,
                &texture,
                extent,
                &self.capabilities,
                RasterRecipe::UV_TEXTURE_LOAD,
            )
            .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        let (mesh_reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || texture.reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(|error| match error {
            PairReservationError::First(error) => map_snapshot_use(error),
            PairReservationError::Second(error) => map_texture_use(error),
        })?;
        self.start_uv_with_recipe(
            snapshot,
            texture,
            UvStartRequest {
                graph,
                uniform,
                reservation: mesh_reservation,
                texture_reservation,
                recipe: RasterRecipe::UV_TEXTURE_LOAD,
            },
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
            build_uv_textured_camera_graph(
                snapshot,
                &texture,
                extent,
                &self.capabilities,
                RasterRecipe::UV_LINEAR_CLAMP_UNORM,
            )
            .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        let (mesh_reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || texture.reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(|error| match error {
            PairReservationError::First(error) => map_snapshot_use(error),
            PairReservationError::Second(error) => map_texture_use(error),
        })?;
        self.start_uv_with_recipe(
            snapshot,
            texture,
            UvStartRequest {
                graph,
                uniform,
                reservation: mesh_reservation,
                texture_reservation,
                recipe: RasterRecipe::UV_LINEAR_CLAMP_UNORM,
            },
        )
    }

    /// Starts the closed explicit-UV sRGB decode, linear-filtering,
    /// clamp-to-edge sampler draw. Encoded source bytes are decoded by the
    /// native sRGB texture before filtering; the render target remains linear
    /// [`TextureFormat::Rgba8Unorm`].
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
            .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        let (mesh_reservation, texture_reservation) = reserve_pair(
            snapshot.reserve_for_draw(),
            || texture.reserve_for_draw(),
            SnapshotDrawReservation::release_before_submit,
        )
        .map_err(|error| match error {
            PairReservationError::First(error) => map_snapshot_use(error),
            PairReservationError::Second(error) => map_texture_use(error),
        })?;
        self.start_uv_with_recipe(
            snapshot,
            texture,
            UvStartRequest {
                graph,
                uniform,
                reservation: mesh_reservation,
                texture_reservation,
                recipe: RasterRecipe::UV_LINEAR_CLAMP_SRGB,
            },
        )
    }

    pub(in crate::fixed_frame) fn start_camera(
        &self,
        snapshot: &IndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        let reservation = snapshot.reserve_for_draw().map_err(map_snapshot_use)?;
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::Indexed(snapshot.clone()),
            texture: None,
            graph,
            uniform,
            reservation,
            texture_reservation: None,
            recipe: RasterRecipe::LEGACY_UNLIT,
        })
    }

    #[cfg(all(test, windows))]
    pub(in crate::fixed_frame) fn start_normal_lambert(
        &self,
        snapshot: &NormalIndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::Normal(snapshot.clone()),
            texture: None,
            graph,
            uniform,
            reservation,
            texture_reservation: None,
            recipe: RasterRecipe::NORMAL_LAMBERT,
        })
    }

    pub(in crate::fixed_frame) fn start_textured(
        &self,
        snapshot: &IndexedMeshSnapshot,
        texture: &BaseColorTextureSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
        texture_reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::Indexed(snapshot.clone()),
            texture: Some(FrameTextureSnapshot::Linear(texture.clone())),
            graph,
            uniform,
            reservation,
            texture_reservation: Some(texture_reservation),
            recipe: RasterRecipe::POSITION_TEXTURE_LOAD,
        })
    }

    #[allow(
        dead_code,
        reason = "U06 paired fixture exercises the flavor helper directly"
    )]
    pub(in crate::fixed_frame) fn start_uv_textured(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        texture: &BaseColorTextureSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
        texture_reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.start_uv_with_recipe(
            snapshot,
            FrameTextureSnapshot::Linear(texture.clone()),
            UvStartRequest {
                graph,
                uniform,
                reservation,
                texture_reservation,
                recipe: RasterRecipe::UV_TEXTURE_LOAD,
            },
        )
    }

    pub(in crate::fixed_frame) fn start_uv_with_recipe(
        &self,
        snapshot: &TexturedIndexedMeshSnapshot,
        texture: FrameTextureSnapshot,
        request: UvStartRequest,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::TexturedUv(snapshot.clone()),
            texture: Some(texture),
            graph: request.graph,
            uniform: request.uniform,
            reservation: request.reservation,
            texture_reservation: Some(request.texture_reservation),
            recipe: request.recipe,
        })
    }

    fn start_with_resources(
        &self,
        resources: StartResources,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        let StartResources {
            snapshot,
            texture,
            graph,
            uniform,
            reservation,
            texture_reservation,
            recipe,
        } = resources;
        debug_assert!(recipe.accepts(&snapshot, texture.as_ref()));
        let release = |reservation: SnapshotDrawReservation,
                       texture_reservation: Option<SnapshotDrawReservation>| {
            reservation.release_before_submit();
            if let Some(reservation) = texture_reservation {
                reservation.release_before_submit();
            }
        };
        let pipeline = match self.device.create_raster_pipeline(recipe.kernel()) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                release(reservation, texture_reservation);
                return Err(DrawStartError::Pipeline(error.to_string()));
            }
        };
        let mut objects = RasterObjectProvider::new(&self.device);
        if let Err(error) = objects.register_raster_pipeline(graph.pipeline, pipeline) {
            release(reservation, texture_reservation);
            return Err(DrawStartError::Pipeline(error.to_string()));
        }
        let binding_result = match recipe.binding() {
            BindingRecipe::NormalLambert => objects.register_raster_normal_bindings(
                graph.bindings,
                graph.pipeline,
                snapshot.positions().buffer(),
                snapshot
                    .normals()
                    .expect("normal recipe requires normals")
                    .buffer(),
                snapshot.position_count(),
            ),
            BindingRecipe::UvLinearClamp => objects
                .register_raster_uv_linear_clamp_textured_bindings(
                    graph.bindings,
                    graph.pipeline,
                    snapshot.positions().buffer(),
                    snapshot
                        .texture_coordinates()
                        .expect("UV recipe requires coordinates")
                        .buffer(),
                    snapshot.position_count(),
                ),
            BindingRecipe::UvLinearClampSrgb => objects
                .register_raster_uv_linear_clamp_srgb_textured_bindings(
                    graph.bindings,
                    graph.pipeline,
                    snapshot.positions().buffer(),
                    snapshot
                        .texture_coordinates()
                        .expect("UV recipe requires coordinates")
                        .buffer(),
                    snapshot.position_count(),
                ),
            BindingRecipe::UvTextureLoad => objects.register_raster_uv_textured_bindings(
                graph.bindings,
                graph.pipeline,
                snapshot.positions().buffer(),
                snapshot
                    .texture_coordinates()
                    .expect("UV recipe requires coordinates")
                    .buffer(),
                snapshot.position_count(),
            ),
            BindingRecipe::TextureLoad => {
                objects.register_raster_textured_bindings(graph.bindings, graph.pipeline)
            }
            BindingRecipe::Uniform => {
                objects.register_raster_uniform_bindings(graph.bindings, graph.pipeline)
            }
        };
        if let Err(error) = binding_result {
            release(reservation, texture_reservation);
            return Err(DrawStartError::Pipeline(error.to_string()));
        }
        let pending = match self.device.upload_immutable_buffer(
            BufferDescriptor {
                buffer: fluxel_rendergraph::BufferDesc {
                    size: FRAME_UNIFORM_BYTES as u64,
                },
                usage: fluxel_rendergraph::BufferUsage::from_kinds([
                    fluxel_rendergraph::BufferUsageKind::CopyDestination,
                    fluxel_rendergraph::BufferUsageKind::Uniform,
                ]),
                memory: MemoryPolicy::DeviceOnly,
            },
            uniform.bytes(),
        ) {
            Ok(pending) => pending,
            Err(error) => {
                release(reservation, texture_reservation);
                return Err(DrawStartError::UniformStart(error));
            }
        };
        Ok(FixedFrameSubmission {
            phase: CameraPhase::Uploading(pending),
            executor: Arc::clone(&self.executor),
            snapshot,
            graph: Some(graph),
            objects: Some(objects),
            reservation: Some(reservation),
            texture_snapshot: texture,
            texture_reservation,
            target_export: None,
            image: None,
            failure: None,
            #[cfg(all(test, windows))]
            completed: None,
        })
    }
}

impl fmt::Debug for FixedFrameRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixedFrameRenderer")
            .finish_non_exhaustive()
    }
}

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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextureFormatNotFilterable { format } => {
                write!(
                    formatter,
                    "fixed sampler texture format is not filterable: {format:?}"
                )
            }
            Self::InvalidExtent => formatter.write_str("fixed frame extent must be nonzero"),
            Self::ForeignSnapshotDevice => {
                formatter.write_str("snapshot belongs to another device")
            }
            Self::InvalidIndexCount => {
                formatter.write_str("fixed frame needs a nonzero triangle-list index count")
            }
            Self::SnapshotInFlight => formatter.write_str("snapshot draw is already in flight"),
            Self::SnapshotPoisoned => formatter.write_str("snapshot draw state is poisoned"),
            Self::TextureInFlight => {
                formatter.write_str("base-color texture draw is already in flight")
            }
            Self::TexturePoisoned => {
                formatter.write_str("base-color texture draw state is poisoned")
            }
            Self::InvalidCameraMaterial => {
                formatter.write_str("invalid camera or material uniform input")
            }
            Self::NonFinitePosition => formatter.write_str("mesh position is not finite"),
            Self::NonFiniteClipPosition => formatter.write_str("clip position is not finite"),
            Self::ClipWNonPositive => formatter.write_str("clip position has non-positive w"),
            Self::ClipOutOfBounds => formatter.write_str("mesh is outside the fixed clip volume"),
            Self::NonFiniteTextureCoordinate => {
                formatter.write_str("mesh texture coordinate is not finite")
            }
            Self::UniformStart(error) => {
                write!(formatter, "frame uniform upload did not start: {error}")
            }
            Self::Graph(error) => write!(formatter, "fixed frame graph rejected: {error}"),
            Self::Pipeline(error) => write!(formatter, "fixed frame pipeline rejected: {error}"),
        }
    }
}

fn rgba8_unorm_filterable(capabilities: &DeviceCapabilities) -> bool {
    capabilities.texture_formats.iter().any(|format| {
        format.format == TextureFormat::Rgba8Unorm && format.sampled && format.filterable
    })
}

fn rgba8_unorm_srgb_filterable(capabilities: &DeviceCapabilities) -> bool {
    capabilities.texture_formats.iter().any(|format| {
        format.format == TextureFormat::Rgba8UnormSrgb && format.sampled && format.filterable
    })
}

fn map_snapshot_use(error: SnapshotUseError) -> DrawStartError {
    match error {
        SnapshotUseError::InFlight => DrawStartError::SnapshotInFlight,
        SnapshotUseError::Poisoned => DrawStartError::SnapshotPoisoned,
    }
}
fn map_texture_use(error: SnapshotUseError) -> DrawStartError {
    match error {
        SnapshotUseError::InFlight => DrawStartError::TextureInFlight,
        SnapshotUseError::Poisoned => DrawStartError::TexturePoisoned,
    }
}

pub(super) enum PairReservationError<E> {
    First(E),
    Second(E),
}

/// Acquires two independent gates transactionally: a failed second acquisition
/// rolls the first one back before the error crosses the public boundary.
pub(super) fn reserve_pair<A, B, E>(
    first: Result<A, E>,
    second: impl FnOnce() -> Result<B, E>,
    rollback: impl FnOnce(A),
) -> Result<(A, B), PairReservationError<E>> {
    let first = first.map_err(PairReservationError::First)?;
    match second() {
        Ok(second) => Ok((first, second)),
        Err(error) => {
            rollback(first);
            Err(PairReservationError::Second(error))
        }
    }
}

pub(super) fn validate_clip(
    positions: &[[f32; 3]],
    indices: &[u32],
    matrix: &[[f32; 4]; 4],
) -> Result<(), DrawStartError> {
    // FrameUniform stores a column-major projection*view matrix. The CPU gate
    // mirrors the portable D3D/WebGPU clip volume used by both backends:
    // x/y are in [-w, w] and z is in [0, w].
    for &index in indices {
        let position = positions
            .get(index as usize)
            .ok_or(DrawStartError::InvalidIndexCount)?;
        if !position.iter().all(|value| value.is_finite()) {
            return Err(DrawStartError::NonFinitePosition);
        }
        let input = [position[0], position[1], position[2], 1.0];
        let mut clip = [0.0; 4];
        for (row, component) in clip.iter_mut().enumerate() {
            for column in 0..4 {
                let product = matrix[column][row] * input[column];
                if !product.is_finite() {
                    return Err(DrawStartError::NonFiniteClipPosition);
                }
                *component += product;
                if !component.is_finite() {
                    return Err(DrawStartError::NonFiniteClipPosition);
                }
            }
        }
        let w = clip[3];
        if w <= 0.0 {
            return Err(DrawStartError::ClipWNonPositive);
        }
        if clip[0] < -w
            || clip[0] > w
            || clip[1] < -w
            || clip[1] > w
            || clip[2] < 0.0
            || clip[2] > w
        {
            return Err(DrawStartError::ClipOutOfBounds);
        }
    }
    Ok(())
}

fn validate_textured_clip(
    positions: &[[f32; 3]],
    indices: &[u32],
    texture_coordinates: &[[f32; 2]],
    matrix: &[[f32; 4]; 4],
) -> Result<(), DrawStartError> {
    validate_clip(positions, indices, matrix)?;
    for &index in indices {
        let coordinate = texture_coordinates
            .get(index as usize)
            .ok_or(DrawStartError::InvalidIndexCount)?;
        if !coordinate.iter().all(|value| value.is_finite()) {
            return Err(DrawStartError::NonFiniteTextureCoordinate);
        }
    }
    Ok(())
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UniformCompletion(error) => {
                write!(formatter, "uniform completion failed: {error:?}")
            }
            Self::UniformObservation(error) => {
                write!(formatter, "uniform completion observation failed: {error}")
            }
            Self::RasterStart(error) => write!(formatter, "raster did not start: {error}"),
            Self::RasterCompletion(error) => {
                write!(formatter, "raster completion failed: {error:?}")
            }
        }
    }
}
impl std::error::Error for FixedFrameFailure {}
use super::*;
