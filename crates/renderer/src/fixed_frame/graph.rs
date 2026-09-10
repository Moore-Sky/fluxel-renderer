//! Builds fixed-frame RenderGraph declarations from closed raster recipes.
//!
//! Graphs restore imported snapshot states on export and expose the offscreen target
//! as `CopySource` for the post-graph readback oracle. Native handles remain opaque:
//! this module only translates closed recipes into portable graph slots and commands.

#[cfg(all(test, windows))]
pub(super) struct FixedGraph {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph<()>,
    pub(super) position_slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) index_slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) target_export: fluxel_rendergraph::ExportTextureSlot,
    #[allow(
        dead_code,
        reason = "U02 validates restored imported states through these exports"
    )]
    pub(super) position_export: fluxel_rendergraph::ExportBufferSlot,
    #[allow(
        dead_code,
        reason = "U02 validates restored imported states through these exports"
    )]
    pub(super) index_export: fluxel_rendergraph::ExportBufferSlot,
}

pub(super) struct CameraGraph {
    pub(super) compiled: fluxel_rendergraph::CompiledGraph<()>,
    pub(super) position_slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) index_slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) uniform_slot: fluxel_rendergraph::ImportBufferSlot,
    pub(super) texture_coordinate_slot: Option<fluxel_rendergraph::ImportBufferSlot>,
    pub(super) normal_slot: Option<fluxel_rendergraph::ImportBufferSlot>,
    pub(super) texture_slot: Option<fluxel_rendergraph::ImportTextureSlot>,
    #[allow(
        dead_code,
        reason = "U04 test-only evidence checks the graph-reported source texture state"
    )]
    pub(super) texture_export: Option<fluxel_rendergraph::ExportTextureSlot>,
    pub(super) pipeline: RasterPipelineId,
    pub(super) bindings: BindingSetId,
    pub(super) target_export: fluxel_rendergraph::ExportTextureSlot,
    #[allow(
        dead_code,
        reason = "U03 test-only evidence checks restored snapshot states"
    )]
    pub(super) position_export: fluxel_rendergraph::ExportBufferSlot,
    #[allow(
        dead_code,
        reason = "U03 test-only evidence checks restored snapshot states"
    )]
    pub(super) index_export: fluxel_rendergraph::ExportBufferSlot,
    #[allow(dead_code, reason = "U05 checks restored UV buffer state")]
    pub(super) texture_coordinate_export: Option<fluxel_rendergraph::ExportBufferSlot>,
    #[allow(dead_code, reason = "U08 checks restored normal buffer state")]
    pub(super) normal_export: Option<fluxel_rendergraph::ExportBufferSlot>,
}

pub(super) fn build_camera_graph(
    snapshot: &IndexedMeshSnapshot,
    extent: [u32; 2],
    capabilities: &DeviceCapabilities,
) -> Result<CameraGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot(
        "camera-frame-positions",
        ImportBufferContract {
            descriptor: snapshot.positions().buffer().descriptor().buffer,
            initial_state: snapshot.positions().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let indices = graph.import_buffer_slot(
        "camera-frame-indices",
        ImportBufferContract {
            descriptor: snapshot.indices().buffer().descriptor().buffer,
            initial_state: snapshot.indices().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let uniform = graph.import_buffer_slot(
        "camera-frame-uniform",
        ImportBufferContract {
            descriptor: fluxel_rendergraph::BufferDesc {
                size: FRAME_UNIFORM_BYTES as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "camera-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "camera-material-indexed",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let indices_read = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            let uniform_read = pass.read_buffer(
                &uniform.version,
                fluxel_rendergraph::BufferReadUse::Uniform,
                BufferRange::Whole,
            );
            (output, (vertices, indices_read, uniform_read))
        },
        move |commands, resolver, data, _| {
            commands.set_pipeline(camera_pipeline())?;
            let bindings = resolver.resolve_bindings(
                camera_bindings(),
                &[BindingResource::BufferRead(&data.2)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(capabilities)?.graph;
    Ok(CameraGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        uniform_slot: uniform.slot,
        texture_coordinate_slot: None,
        normal_slot: None,
        texture_slot: None,
        texture_export: None,
        pipeline: camera_pipeline(),
        bindings: camera_bindings(),
        target_export,
        position_export,
        index_export,
        texture_coordinate_export: None,
        normal_export: None,
    })
}

/// Declares the closed three-stream normal/Lambert graph. Normal is a second
/// vertex read, not a uniform or a texture attribute: this keeps the portable
/// RenderGraph plan aligned with the native slot-0/slot-1 ABI.
pub(super) fn build_normal_lambert_camera_graph(
    snapshot: &NormalIndexedMeshSnapshot,
    extent: [u32; 2],
    capabilities: &DeviceCapabilities,
) -> Result<CameraGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let import = |graph: &mut RenderGraph, name: &str, buffer: &UploadedBuffer| {
        graph.import_buffer_slot(
            name,
            ImportBufferContract {
                descriptor: buffer.buffer().descriptor().buffer,
                initial_state: buffer.outgoing_state(),
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        )
    };
    let positions = import(
        &mut graph,
        "normal-lambert-frame-positions",
        snapshot.positions(),
    );
    let indices = import(
        &mut graph,
        "normal-lambert-frame-indices",
        snapshot.indices(),
    );
    let normals = import(
        &mut graph,
        "normal-lambert-frame-normals",
        snapshot.normals(),
    );
    let uniform = graph.import_buffer_slot(
        "normal-lambert-frame-uniform",
        ImportBufferContract {
            descriptor: fluxel_rendergraph::BufferDesc {
                size: FRAME_UNIFORM_BYTES as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "normal-lambert-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "normal-lambert-camera-material-indexed",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let position_read = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let index_read = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            let normal_read = pass.read_buffer(
                &normals.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let uniform_read = pass.read_buffer(
                &uniform.version,
                fluxel_rendergraph::BufferReadUse::Uniform,
                BufferRange::Whole,
            );
            (
                output,
                (position_read, index_read, normal_read, uniform_read),
            )
        },
        move |commands, resolver, data, _| {
            commands.set_pipeline(normal_lambert_pipeline())?;
            let bindings = resolver.resolve_bindings(
                normal_lambert_bindings(),
                &[BindingResource::BufferRead(&data.3)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_vertex_buffer(1, &data.2)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let normal_export = graph.export_buffer(
        normals.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(capabilities)?.graph;
    Ok(CameraGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        uniform_slot: uniform.slot,
        texture_coordinate_slot: None,
        normal_slot: Some(normals.slot),
        texture_slot: None,
        texture_export: None,
        pipeline: normal_lambert_pipeline(),
        bindings: normal_lambert_bindings(),
        target_export,
        position_export,
        index_export,
        texture_coordinate_export: None,
        normal_export: Some(normal_export),
    })
}

pub(super) fn build_textured_camera_graph(
    snapshot: &IndexedMeshSnapshot,
    texture: &BaseColorTextureSnapshot,
    extent: [u32; 2],
    capabilities: &DeviceCapabilities,
) -> Result<CameraGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot(
        "textured-camera-frame-positions",
        ImportBufferContract {
            descriptor: snapshot.positions().buffer().descriptor().buffer,
            initial_state: snapshot.positions().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let indices = graph.import_buffer_slot(
        "textured-camera-frame-indices",
        ImportBufferContract {
            descriptor: snapshot.indices().buffer().descriptor().buffer,
            initial_state: snapshot.indices().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let uniform = graph.import_buffer_slot(
        "textured-camera-frame-uniform",
        ImportBufferContract {
            descriptor: fluxel_rendergraph::BufferDesc {
                size: FRAME_UNIFORM_BYTES as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let sampled = graph.import_texture_slot(
        "textured-camera-frame-source",
        ImportTextureContract {
            descriptor: texture.texture().texture().descriptor().texture,
            initial_state: texture.texture().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "textured-camera-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "textured-camera-material-indexed",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let indices_read = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            let uniform_read = pass.read_buffer(
                &uniform.version,
                fluxel_rendergraph::BufferReadUse::Uniform,
                BufferRange::Whole,
            );
            let texture_read = pass.read_texture(
                &sampled.version,
                TextureReadUse::Sampled,
                TextureRange::Whole,
            );
            (output, (vertices, indices_read, uniform_read, texture_read))
        },
        move |commands, resolver, data, _| {
            commands.set_pipeline(textured_pipeline())?;
            let bindings = resolver.resolve_bindings(
                textured_bindings(),
                &[
                    BindingResource::BufferRead(&data.2),
                    BindingResource::TextureRead(&data.3),
                ],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let texture_export = graph.export_texture(
        sampled.version,
        ExportTextureContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(capabilities)?.graph;
    Ok(CameraGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        uniform_slot: uniform.slot,
        texture_coordinate_slot: None,
        normal_slot: None,
        texture_slot: Some(sampled.slot),
        texture_export: Some(texture_export),
        pipeline: textured_pipeline(),
        bindings: textured_bindings(),
        target_export,
        position_export,
        index_export,
        texture_coordinate_export: None,
        normal_export: None,
    })
}

pub(super) fn build_uv_textured_camera_graph<T: FrameTexture>(
    snapshot: &TexturedIndexedMeshSnapshot,
    texture: &T,
    extent: [u32; 2],
    capabilities: &DeviceCapabilities,
    recipe: RasterRecipe,
) -> Result<CameraGraph, fluxel_rendergraph::CompileError> {
    debug_assert_eq!(recipe.graph(), GraphRecipe::UvTextured);
    let (pipeline, bindings) = (recipe.pipeline(), recipe.bindings());
    let mut graph = RenderGraph::new();
    let import_buffer = |graph: &mut RenderGraph, name: &str, buffer: &UploadedBuffer| {
        graph.import_buffer_slot(
            name,
            ImportBufferContract {
                descriptor: buffer.buffer().descriptor().buffer,
                initial_state: buffer.outgoing_state(),
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        )
    };
    let positions = import_buffer(
        &mut graph,
        "uv-textured-frame-positions",
        snapshot.positions(),
    );
    let indices = import_buffer(&mut graph, "uv-textured-frame-indices", snapshot.indices());
    let texture_coordinates = import_buffer(
        &mut graph,
        "uv-textured-frame-coordinates",
        snapshot.texture_coordinates(),
    );
    let uniform = graph.import_buffer_slot(
        "uv-textured-frame-uniform",
        ImportBufferContract {
            descriptor: fluxel_rendergraph::BufferDesc {
                size: FRAME_UNIFORM_BYTES as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let sampled = graph.import_texture_slot(
        "uv-textured-frame-source",
        ImportTextureContract {
            descriptor: texture.uploaded_texture().texture().descriptor().texture,
            initial_state: texture.uploaded_texture().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "uv-textured-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "uv-textured-camera-material-indexed",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let indices_read = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            let coordinates = pass.read_buffer(
                &texture_coordinates.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let uniform_read = pass.read_buffer(
                &uniform.version,
                fluxel_rendergraph::BufferReadUse::Uniform,
                BufferRange::Whole,
            );
            let texture_read = pass.read_texture(
                &sampled.version,
                TextureReadUse::Sampled,
                TextureRange::Whole,
            );
            (
                output,
                (
                    vertices,
                    indices_read,
                    coordinates,
                    uniform_read,
                    texture_read,
                ),
            )
        },
        move |commands, resolver, data, _| {
            commands.set_pipeline(pipeline)?;
            let bindings = resolver.resolve_bindings(
                bindings,
                &[
                    BindingResource::BufferRead(&data.3),
                    BindingResource::TextureRead(&data.4),
                ],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_vertex_buffer(1, &data.2)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let texture_coordinate_export = graph.export_buffer(
        texture_coordinates.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let texture_export = graph.export_texture(
        sampled.version,
        ExportTextureContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(capabilities)?.graph;
    Ok(CameraGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        uniform_slot: uniform.slot,
        texture_coordinate_slot: Some(texture_coordinates.slot),
        normal_slot: None,
        texture_slot: Some(sampled.slot),
        texture_export: Some(texture_export),
        pipeline,
        bindings,
        target_export,
        position_export,
        index_export,
        texture_coordinate_export: Some(texture_coordinate_export),
        normal_export: None,
    })
}

#[cfg(all(test, windows))]
pub(super) fn build_graph(
    snapshot: &IndexedMeshSnapshot,
    extent: [u32; 2],
    capabilities: &DeviceCapabilities,
) -> Result<FixedGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot(
        "fixed-frame-positions",
        ImportBufferContract {
            descriptor: snapshot.positions().buffer().descriptor().buffer,
            initial_state: snapshot.positions().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let indices = graph.import_buffer_slot(
        "fixed-frame-indices",
        ImportBufferContract {
            descriptor: snapshot.indices().buffer().descriptor().buffer,
            initial_state: snapshot.indices().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "fixed-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let index_count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "fixed-frame-indexed-unlit",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let index = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            (output, (vertices, index))
        },
        move |commands, _, data, _| {
            commands.set_pipeline(fixed_pipeline())?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..index_count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(capabilities)?.graph;
    Ok(FixedGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        target_export,
        position_export,
        index_export,
    })
}
use super::*;
