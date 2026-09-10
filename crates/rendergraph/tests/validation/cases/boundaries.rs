//! Import, export, and surface-boundary validation cases.

use super::*;

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
                super::capabilities::color_ops(
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

#[test]
fn ordinary_imports_reject_surface_ownership_and_duplicate_queue_ids() {
    let mut graph = RenderGraph::<()>::new();
    graph.import_buffer_slot(
        "invalid-owner",
        ImportBufferContract {
            descriptor: BufferDesc { size: 16 },
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Surface,
            initial_contents: InitialContents::Defined,
        },
    );
    assert_error(
        graph.compile(&capabilities(true, true, true, true)),
        CompileErrorKind::MissingImportContract,
    );

    let mut caps = capabilities(true, true, true, true);
    caps.queues.push(QueueDescriptor::new(
        QueueId::new(0),
        QueueCapabilities::new(true, true, true, true),
    ));
    assert_error(
        RenderGraph::<()>::new().compile(&caps),
        CompileErrorKind::UnsupportedSemanticRequirement,
    );
}
