use fluxel_rendergraph::*;

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn capabilities(
    compute: bool,
    storage_buffers: bool,
    storage_textures: bool,
    surface_color: bool,
) -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, compute, true, true),
        ))
        .limits(DeviceLimits::new(4, 1))
        .buffers(BufferCapabilities::new(
            storage_buffers,
            storage_buffers,
            true,
        ))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(storage_textures, storage_textures)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .surface(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            surface_color,
            true,
        ))
        .build()
}

fn assert_error(result: CompileResult, expected: CompileErrorKind) {
    assert_eq!(result.expect_err("graph should be rejected").kind, expected);
}

fn import_buffer(contents: InitialContents) -> ImportBufferContract {
    ImportBufferContract {
        descriptor: BufferDesc { size: 16 },
        initial_state: ResourceAccessState::ShaderStorageRead,
        ownership: ExternalOwnership::Caller,
        initial_contents: contents,
    }
}

#[test]
fn initialization_tracks_full_unknown_imported_and_partial_contents() {
    let caps = capabilities(true, true, true, true);

    let mut full = RenderGraph::<()>::new();
    let buffer = full.create_buffer("full", BufferDesc { size: 16 });
    let written = full.add_compute_pass(
        "initialize",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    full.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert!(full.compile(&caps).is_ok());

    let mut unknown = RenderGraph::<()>::new();
    let buffer = unknown.create_buffer("unknown", BufferDesc { size: 16 });
    let written = unknown.add_compute_pass(
        "unknown-write",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Unknown,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    unknown.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        unknown.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );

    let mut defined = RenderGraph::<()>::new();
    let imported = defined.import_buffer_slot("defined", import_buffer(InitialContents::Defined));
    let reader = defined.add_compute_pass(
        "read-import",
        |pass| {
            let _ = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    defined.mark_side_effect(reader.id, SideEffectReason::Diagnostic("keep read".into()));
    assert!(defined.compile(&caps).is_ok());

    let mut undefined = RenderGraph::<()>::new();
    let imported =
        undefined.import_buffer_slot("undefined", import_buffer(InitialContents::Undefined));
    let reader = undefined.add_compute_pass(
        "read-import",
        |pass| {
            let _ = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    undefined.mark_side_effect(reader.id, SideEffectReason::Diagnostic("keep read".into()));
    assert_error(
        undefined.compile(&caps),
        CompileErrorKind::ReadBeforeInitialization,
    );

    let mut partial = RenderGraph::<()>::new();
    let buffer = partial.create_buffer("partial", BufferDesc { size: 16 });
    let written = partial.add_compute_pass(
        "partially-initialize",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::Bytes { offset: 0, size: 8 },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    partial.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        partial.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );
}

#[test]
fn invalid_buffer_and_texture_ranges_are_rejected() {
    let caps = capabilities(true, true, true, true);

    let mut buffers = RenderGraph::<()>::new();
    let buffer = buffers.create_buffer("buffer", BufferDesc { size: 16 });
    let pass = buffers.add_compute_pass(
        "bad-buffer-range",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::Bytes {
                    offset: 12,
                    size: 8,
                },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    buffers.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        buffers.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );

    let mut textures = RenderGraph::<()>::new();
    let image = textures.create_texture("image", texture());
    let pass = textures.add_compute_pass(
        "bad-texture-range",
        |pass| {
            let (next, _) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::Subresources {
                    base_mip_level: 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: TextureAspect::Color,
                },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    textures.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        textures.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );
}

#[test]
fn compute_buffer_texture_and_surface_capabilities_are_checked() {
    let mut compute = RenderGraph::<()>::new();
    let buffer = compute.create_buffer("buffer", BufferDesc { size: 16 });
    let pass = compute.add_compute_pass(
        "compute",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    compute.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        compute.compile(&capabilities(false, true, true, true)),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );

    let mut buffer_caps = RenderGraph::<()>::new();
    let buffer = buffer_caps.create_buffer("buffer", BufferDesc { size: 16 });
    let pass = buffer_caps.add_compute_pass(
        "storage-buffer",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    buffer_caps.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        buffer_caps.compile(&capabilities(true, false, true, true)),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );

    let mut texture_caps = RenderGraph::<()>::new();
    let image = texture_caps.create_texture("image", texture());
    let pass = texture_caps.add_compute_pass(
        "storage-texture",
        |pass| {
            let (next, _) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    texture_caps.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        texture_caps.compile(&capabilities(true, true, false, true)),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );

    let mut surface_caps = RenderGraph::<()>::new();
    let surface = surface_caps.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = surface_caps.add_raster_pass(
        "surface",
        |pass| {
            let next = pass.color_attachment(
                surface.version,
                color_ops(
                    LoadOp::Clear([0.0; 4]),
                    StoreOp::Store,
                    WriteCoverage::Unknown,
                ),
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    surface_caps.present(pass.output, PresentContract::new());
    assert_error(
        surface_caps.compile(&capabilities(true, true, true, false)),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );
}

#[test]
fn srgb_format_requires_its_own_capability_entry() {
    let mut graph = RenderGraph::<()>::new();
    let mut descriptor = texture();
    descriptor.format = TextureFormat::Rgba8UnormSrgb;
    let image = graph.import_texture_slot(
        "encoded base color",
        ImportTextureContract {
            descriptor,
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let sampled = graph.add_compute_pass(
        "sample encoded base color",
        |pass| {
            let image = pass.read_texture(
                &image.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), image)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(sampled.id, SideEffectReason::Diagnostic("retain".into()));

    let mut caps = capabilities(true, true, true, true);
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );

    caps.texture_formats.push(
        TextureFormatCapabilities::builder(TextureFormat::Rgba8UnormSrgb)
            .sampled(true, true)
            .copies(true, true)
            .build(),
    );
    graph
        .compile(&caps)
        .expect("sRGB must use its own advertised format facts");
}

fn color_ops(
    load: LoadOp<[f32; 4]>,
    store: StoreOp,
    write_coverage: WriteCoverage,
) -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        index: 0,
        range: TextureRange::whole(),
        operations: AttachmentOps {
            load,
            store,
            write_coverage,
        },
    }
}

#[test]
fn attachment_clear_dont_care_and_discard_control_validity() {
    let caps = capabilities(true, true, true, true);

    let mut clear = RenderGraph::<()>::new();
    let image = clear.create_texture("clear", texture());
    let pass = clear.add_raster_pass(
        "clear",
        |pass| {
            (
                pass.color_attachment(
                    image,
                    color_ops(
                        LoadOp::Clear([0.0; 4]),
                        StoreOp::Store,
                        WriteCoverage::Unknown,
                    ),
                ),
                (),
            )
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    clear.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );
    assert!(clear.compile(&caps).is_ok());

    let mut dont_care = RenderGraph::<()>::new();
    let image = dont_care.create_texture("dont-care", texture());
    let pass = dont_care.add_raster_pass(
        "dont-care",
        |pass| {
            (
                pass.color_attachment(
                    image,
                    color_ops(LoadOp::DontCare, StoreOp::Store, WriteCoverage::Unknown),
                ),
                (),
            )
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    dont_care.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );
    assert_error(
        dont_care.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );

    let mut discard = RenderGraph::<()>::new();
    let image = discard.create_texture("discard", texture());
    let pass = discard.add_raster_pass(
        "discard",
        |pass| {
            (
                pass.color_attachment(
                    image,
                    color_ops(
                        LoadOp::Clear([0.0; 4]),
                        StoreOp::Discard,
                        WriteCoverage::Unknown,
                    ),
                ),
                (),
            )
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    discard.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );
    assert_error(
        discard.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );
}

#[test]
fn invalid_texture_descriptors_and_depth_stencil_shapes_are_rejected() {
    let caps = capabilities(true, true, true, true);
    let mut graph = RenderGraph::<()>::new();
    let mut invalid = texture();
    invalid.mip_levels = 8;
    let image = graph.create_texture("too-many-mips", invalid);
    let pass = graph.add_compute_pass(
        "write",
        |builder| {
            let (next, _) = builder.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );

    let mut graph = RenderGraph::<()>::new();
    let mut depth_desc = texture();
    depth_desc.format = TextureFormat::Depth32Float;
    let depth = graph.create_texture("depth", depth_desc);
    let pass = graph.add_raster_pass(
        "empty-depth",
        |builder| {
            let next = builder.depth_stencil_attachment(
                depth,
                DepthStencilAttachmentDesc {
                    range: TextureRange::whole(),
                    depth: None,
                    stencil: None,
                },
            );
            (next, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::DepthStencilWrite,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );
}

#[test]
fn import_and_export_boundary_contracts_are_validated_without_pass_accesses() {
    let caps = capabilities(true, true, true, true);
    let mut graph = RenderGraph::<()>::new();
    let invalid = graph.import_buffer_slot(
        "invalid-state",
        ImportBufferContract {
            descriptor: BufferDesc { size: 16 },
            initial_state: ResourceAccessState::ColorAttachmentWrite,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_buffer(
        invalid.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::MissingImportContract,
    );

    let mut graph = RenderGraph::<()>::new();
    let mut invalid_desc = texture();
    invalid_desc.mip_levels = 8;
    let invalid = graph.import_texture_slot(
        "invalid-desc",
        ImportTextureContract {
            descriptor: invalid_desc,
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_texture(
        invalid.version,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );

    let mut graph = RenderGraph::<()>::new();
    let defined = graph.import_texture_slot(
        "color",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_texture(
        defined.version,
        ExportTextureContract {
            final_state: ResourceAccessState::DepthStencilWrite,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );
}

#[test]
fn surface_rejects_unmodeled_texture_semantics_and_non_swapchain_shapes() {
    let caps = capabilities(true, true, true, true);
    let mut graph = RenderGraph::<()>::new();
    let surface = graph.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let initialized = graph.add_raster_pass(
        "surface-color",
        |builder| {
            let next = builder.color_attachment(
                surface.version,
                color_ops(
                    LoadOp::Clear([0.0; 4]),
                    StoreOp::Store,
                    WriteCoverage::Unknown,
                ),
            );
            (next, ())
        },
        |_, _, _, _| Ok(()),
    );
    let sampled = graph.add_compute_pass(
        "sample-surface",
        |builder| {
            let read = builder.read_texture(
                &initialized.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), read)
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(sampled.id, SideEffectReason::Diagnostic("retain".into()));
    graph.present(initialized.output, PresentContract::new());
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );

    let mut graph = RenderGraph::<()>::new();
    let mut invalid_surface = texture();
    invalid_surface.sample_count = 4;
    graph.import_surface_texture_slot(
        "multisampled-surface",
        SurfaceTextureContract {
            descriptor: invalid_surface,
        },
    );
    assert_error(
        graph.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );
}
