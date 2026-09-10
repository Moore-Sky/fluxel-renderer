//! Capability and attachment validation cases.

use super::*;

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

pub(super) fn color_ops(
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
