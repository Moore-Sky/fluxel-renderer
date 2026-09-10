//! Reservation-preserving transition from validated draw input to submission.
//!
//! Failures before native work is accepted release both reservations; after the
//! returned submission takes ownership, `submission` chooses release or poison.

use super::*;

impl FixedFrameRenderer {
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
    pub(super) fn begin_normal_lambert(
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
    pub(super) fn begin_vertex_color(
        &self,
        snapshot: &VertexColorIndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.start_with_resources(StartResources {
            snapshot: FrameMeshSnapshot::VertexColor(snapshot.clone()),
            texture: None,
            graph,
            uniform,
            reservation,
            texture_reservation: None,
            recipe: RasterRecipe::VERTEX_COLOR,
        })
    }
    #[cfg(all(test, windows))]
    pub(in crate::fixed_frame) fn start_vertex_color(
        &self,
        snapshot: &VertexColorIndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.begin_vertex_color(snapshot, graph, uniform, reservation)
    }
    #[cfg(all(test, windows))]
    pub(in crate::fixed_frame) fn start_normal_lambert(
        &self,
        snapshot: &NormalIndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        self.begin_normal_lambert(snapshot, graph, uniform, reservation)
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
            BindingRecipe::VertexColor => objects.register_raster_vertex_color_bindings(
                graph.bindings,
                graph.pipeline,
                snapshot.positions().buffer(),
                snapshot
                    .colors()
                    .expect("vertex-color recipe requires colors")
                    .buffer(),
                snapshot.position_count(),
            ),
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
