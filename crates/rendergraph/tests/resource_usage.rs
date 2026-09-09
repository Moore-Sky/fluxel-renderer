//! Resource-usage summary regression tests.

use fluxel_rendergraph::{
    test_rhi::{TestBuffer, TestRegistry, TestRhi, TestTexture, TestTraceEvent},
    *,
};

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(true, true, true))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(true, true)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
                .attachments(false, true, vec![1])
                .build(),
        )
        .build()
}

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 64 }
}

#[test]
fn culled_resources_have_no_usage_requirement() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let dead = graph.create_buffer("dead", buffer());
    graph.add_compute_pass(
        "dead",
        |pass| {
            let (out, _) = pass.write_buffer(
                dead,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (out, ())
        },
        |_, _, _, _| Ok(()),
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    assert!(compiled.execution_plan().resource_requirements().is_empty());
}

#[test]
fn retained_accesses_and_export_state_are_unioned_for_textures() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let image = graph.create_texture("image", texture());
    let written = graph.add_compute_pass(
        "write",
        |pass| {
            let (out, handle) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (out, handle)
        },
        |_, _, _, _| Ok(()),
    );
    let sampled = graph.add_compute_pass(
        "sample",
        |pass| {
            let handle = pass.read_texture(
                &written.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), handle)
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(sampled.id, SideEffectReason::Diagnostic("retain".into()));
    graph.export_texture(
        written.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );

    let plan = graph
        .compile(&capabilities())
        .unwrap()
        .graph
        .execution_plan()
        .clone();
    assert_eq!(plan.resource_requirements().len(), 1);
    let ResourceUsageSummary::Texture(usage) = plan.resource_requirements()[0].usage else {
        panic!("expected texture usage");
    };
    assert!(usage.contains(TextureUsageKind::StorageWrite));
    assert!(usage.contains(TextureUsageKind::Sampled));
    assert!(usage.contains(TextureUsageKind::CopySource));
    assert!(!usage.contains(TextureUsageKind::CopyDestination));
}

#[test]
fn import_and_export_states_contribute_buffer_usage() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "read",
        |pass| {
            let handle = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), handle)
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    graph.export_buffer(
        imported.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );

    let plan = graph
        .compile(&capabilities())
        .unwrap()
        .graph
        .execution_plan()
        .clone();
    let ResourceUsageSummary::Buffer(usage) = plan.resource_requirements()[0].usage else {
        panic!("expected buffer usage");
    };
    assert!(usage.contains(BufferUsageKind::Uniform));
    assert!(usage.contains(BufferUsageKind::StorageRead));
    assert!(usage.contains(BufferUsageKind::CopySource));
}

#[test]
fn transient_usage_is_forwarded_to_backend_allocation() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let data = graph.create_buffer("data", buffer());
    let image = graph.create_texture("image", texture());
    let write = graph.add_compute_pass(
        "write",
        |pass| {
            let (buffer_out, buffer_handle) = pass.write_buffer(
                data,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            let (texture_out, texture_handle) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            ((buffer_out, texture_out), (buffer_handle, texture_handle))
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        write.output.0,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    graph.export_texture(
        write.output.1,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let executor = FrameExecutor::new(TestRhi::new(capabilities(), DeviceIdentity::new(9)));
    let registry = TestRegistry::new(DeviceIdentity::new(9));
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    let backend = executor.try_backend().unwrap();
    let trace = backend.trace();
    let usage = trace.iter().find_map(|event| match event {
        TestTraceEvent::CreateBuffer { usage, .. } => Some(*usage),
        _ => None,
    });
    let usage = usage.expect("created transient buffer");
    assert!(usage.contains(BufferUsageKind::StorageWrite));
    assert!(usage.contains(BufferUsageKind::CopySource));
    let usage = trace.iter().find_map(|event| match event {
        TestTraceEvent::CreateTexture { usage, .. } => Some(*usage),
        _ => None,
    });
    let usage = usage.expect("created transient texture");
    assert!(usage.contains(TextureUsageKind::StorageWrite));
    assert!(usage.contains(TextureUsageKind::CopySource));
}

#[test]
fn a_culled_read_does_not_pollute_a_live_resource_requirement() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.add_compute_pass(
        "culled storage reader",
        |pass| {
            let handle = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), handle)
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        imported.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let ResourceUsageSummary::Buffer(usage) =
        compiled.execution_plan().resource_requirements()[0].usage
    else {
        panic!("expected buffer usage");
    };
    assert!(usage.contains(BufferUsageKind::Uniform));
    assert!(usage.contains(BufferUsageKind::CopySource));
    assert!(!usage.contains(BufferUsageKind::StorageRead));
}

#[test]
fn storage_read_write_requires_both_creation_operations() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "storage",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageReadWrite,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let update = graph.add_compute_pass(
        "update",
        |pass| {
            let (output, handle) = pass.read_write_buffer(
                imported.version,
                BufferReadWriteUse::Storage,
                BufferRange::whole(),
            );
            (output, handle)
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        update.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let ResourceUsageSummary::Buffer(usage) =
        compiled.execution_plan().resource_requirements()[0].usage
    else {
        panic!("expected buffer usage");
    };
    assert!(usage.contains(BufferUsageKind::StorageRead));
    assert!(usage.contains(BufferUsageKind::StorageWrite));
    assert!(usage.contains(BufferUsageKind::CopySource));
}

#[test]
fn import_to_export_without_passes_keeps_both_boundary_usages() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "boundary only",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_buffer(
        imported.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let ResourceUsageSummary::Buffer(usage) =
        compiled.execution_plan().resource_requirements()[0].usage
    else {
        panic!("expected buffer usage");
    };
    assert!(usage.contains(BufferUsageKind::Uniform));
    assert!(usage.contains(BufferUsageKind::CopySource));
}

#[test]
fn every_texture_state_maps_to_its_creation_operation() {
    let cases = [
        (
            ResourceAccessState::ShaderSampledRead,
            TextureUsageKind::Sampled,
        ),
        (
            ResourceAccessState::ShaderStorageRead,
            TextureUsageKind::StorageRead,
        ),
        (
            ResourceAccessState::ShaderStorageWrite,
            TextureUsageKind::StorageWrite,
        ),
        (
            ResourceAccessState::ColorAttachmentWrite,
            TextureUsageKind::ColorAttachment,
        ),
        (
            ResourceAccessState::DepthStencilWrite,
            TextureUsageKind::DepthStencilAttachment,
        ),
        (
            ResourceAccessState::CopySource,
            TextureUsageKind::CopySource,
        ),
        (
            ResourceAccessState::CopyDestination,
            TextureUsageKind::CopyDestination,
        ),
    ];

    for (state, expected) in cases {
        let descriptor = if expected == TextureUsageKind::DepthStencilAttachment {
            TextureDesc {
                format: TextureFormat::Depth32Float,
                ..texture()
            }
        } else {
            texture()
        };
        let mut graph: RenderGraph<()> = RenderGraph::new();
        let imported = graph.import_texture_slot(
            "input",
            ImportTextureContract {
                descriptor,
                initial_state: state,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        graph.export_texture(
            imported.version,
            ExportTextureContract { final_state: state },
        );

        let compiled = graph.compile(&capabilities()).unwrap().graph;
        let ResourceUsageSummary::Texture(usage) =
            compiled.execution_plan().resource_requirements()[0].usage
        else {
            panic!("expected texture usage");
        };
        assert!(usage.contains(expected), "missing mapping for {state:?}");
    }
}

#[test]
fn present_root_maps_to_texture_presentation_usage() {
    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let surface = graph.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = graph.add_raster_pass(
        "clear surface",
        |pass| {
            let output = pass.color_attachment(
                surface.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.present(pass.output, PresentContract::new());

    let compiled = graph.compile(&caps).unwrap().graph;
    let ResourceUsageSummary::Texture(usage) =
        compiled.execution_plan().resource_requirements()[0].usage
    else {
        panic!("expected texture usage");
    };
    assert!(usage.contains(TextureUsageKind::Present));
}

#[test]
fn every_buffer_state_maps_to_its_creation_operation() {
    let cases = [
        (ResourceAccessState::UniformRead, BufferUsageKind::Uniform),
        (
            ResourceAccessState::ShaderStorageRead,
            BufferUsageKind::StorageRead,
        ),
        (
            ResourceAccessState::ShaderStorageWrite,
            BufferUsageKind::StorageWrite,
        ),
        (ResourceAccessState::VertexRead, BufferUsageKind::Vertex),
        (ResourceAccessState::IndexRead, BufferUsageKind::Index),
        (ResourceAccessState::IndirectRead, BufferUsageKind::Indirect),
        (ResourceAccessState::CopySource, BufferUsageKind::CopySource),
        (
            ResourceAccessState::CopyDestination,
            BufferUsageKind::CopyDestination,
        ),
    ];

    for (state, expected) in cases {
        let mut graph: RenderGraph<()> = RenderGraph::new();
        let imported = graph.import_buffer_slot(
            "input",
            ImportBufferContract {
                descriptor: buffer(),
                initial_state: state,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        graph.export_buffer(
            imported.version,
            ExportBufferContract { final_state: state },
        );

        let compiled = graph.compile(&capabilities()).unwrap().graph;
        let ResourceUsageSummary::Buffer(usage) =
            compiled.execution_plan().resource_requirements()[0].usage
        else {
            panic!("expected buffer usage");
        };
        assert!(usage.contains(expected), "missing mapping for {state:?}");
    }
}

#[test]
fn usage_sets_cover_exact_and_superset_requirements() {
    let required_texture =
        TextureUsage::from_kinds([TextureUsageKind::Sampled, TextureUsageKind::CopySource]);
    assert!(required_texture.contains_all(required_texture));
    assert!(
        required_texture
            .with(TextureUsageKind::StorageWrite)
            .contains_all(required_texture)
    );
    assert!(!TextureUsage::from_kinds([TextureUsageKind::Sampled]).contains_all(required_texture));

    let required_buffer = BufferUsage::from_kinds([
        BufferUsageKind::StorageRead,
        BufferUsageKind::CopyDestination,
    ]);
    assert!(required_buffer.contains_all(required_buffer));
    assert!(
        required_buffer
            .with(BufferUsageKind::Uniform)
            .contains_all(required_buffer)
    );
    assert!(!BufferUsage::from_kinds([BufferUsageKind::StorageRead]).contains_all(required_buffer));
}

#[test]
fn imported_resource_with_insufficient_allowed_usage_fails_before_recording() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_buffer(
        imported.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let logical = compiled.execution_plan().resource_requirements()[0].resource;
    let device = DeviceIdentity::new(9);
    let executor = FrameExecutor::new(TestRhi::new(capabilities(), device));
    let mut registry = TestRegistry::new(device);
    let binding = BufferBindingId::new(1);
    registry.register_buffer(
        binding,
        TestBuffer::new(1),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::Uniform]),
        ResourceAccessState::UniformRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(imported.slot, binding);

    match executor.execute(
        &compiled,
        compiled.instantiate_local(inputs),
        &registry,
        &registry,
    ) {
        Err(ExecutionError::FrameBinding(error)) => {
            assert_eq!(error.kind, FrameBindingErrorKind::UsageMismatch);
            assert_eq!(error.buffer_slot, Some(imported.slot));
            assert_eq!(error.resource, Some(logical));
            assert!(error.detail.contains("required BufferUsage"));
            assert!(error.detail.contains("actual BufferUsage"));
            assert!(error.detail.contains("CopySource"));
        }
        Err(other) => panic!("unexpected execution error: {other}"),
        Ok(_) => panic!("insufficient imported usage unexpectedly executed"),
    }
    assert!(
        !executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::BeginEncoder { .. }))
    );
}

#[test]
fn imported_resource_allows_exact_and_superset_usage() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_buffer(
        imported.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let exact = BufferUsage::from_kinds([BufferUsageKind::Uniform, BufferUsageKind::CopySource]);

    for allowed in [exact, exact.with(BufferUsageKind::StorageRead)] {
        let device = DeviceIdentity::new(9);
        let executor = FrameExecutor::new(TestRhi::new(capabilities(), device));
        let mut registry = TestRegistry::new(device);
        let binding = BufferBindingId::new(1);
        registry.register_buffer(
            binding,
            TestBuffer::new(1),
            buffer(),
            allowed,
            ResourceAccessState::UniformRead,
        );
        let mut inputs = FrameInputs::new(());
        inputs.bind_buffer(imported.slot, binding);
        executor
            .execute(
                &compiled,
                compiled.instantiate_local(inputs),
                &registry,
                &registry,
            )
            .unwrap();
    }
}

#[test]
fn transient_resource_with_insufficient_allowed_usage_fails_before_recording() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let data = graph.create_buffer("data", buffer());
    let pass = graph.add_compute_pass(
        "write",
        |pass| {
            let (output, _) = pass.write_buffer(
                data,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let logical = compiled.execution_plan().resource_requirements()[0].resource;
    let device = DeviceIdentity::new(9);
    let mut backend = TestRhi::new(capabilities(), device);
    backend.set_transient_buffer_usage(BufferUsage::from_kinds([BufferUsageKind::StorageWrite]));
    let executor = FrameExecutor::new(backend);
    let registry = TestRegistry::new(device);

    match executor.execute(
        &compiled,
        compiled.instantiate_local(FrameInputs::new(())),
        &registry,
        &registry,
    ) {
        Err(ExecutionError::FrameBinding(error)) => {
            assert_eq!(error.kind, FrameBindingErrorKind::UsageMismatch);
            assert_eq!(error.resource, Some(logical));
            assert!(error.buffer_slot.is_none());
        }
        Err(other) => panic!("unexpected execution error: {other}"),
        Ok(_) => panic!("insufficient transient usage unexpectedly executed"),
    }
    assert!(
        !executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::BeginEncoder { .. }))
    );
}

#[test]
fn imported_texture_allows_exact_and_superset_usage_and_rejects_missing_usage() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let imported = graph.import_texture_slot(
        "input",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    graph.export_texture(
        imported.version,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let logical = compiled.execution_plan().resource_requirements()[0].resource;
    let exact = TextureUsage::from_kinds([TextureUsageKind::Sampled, TextureUsageKind::CopySource]);

    for allowed in [exact, exact.with(TextureUsageKind::StorageRead)] {
        let device = DeviceIdentity::new(9);
        let executor = FrameExecutor::new(TestRhi::new(capabilities(), device));
        let mut registry = TestRegistry::new(device);
        let binding = TextureBindingId::new(1);
        registry.register_texture(
            binding,
            TestTexture::new(1),
            texture(),
            allowed,
            ResourceAccessState::ShaderSampledRead,
        );
        let mut inputs = FrameInputs::new(());
        inputs.bind_texture(imported.slot, binding);
        executor
            .execute(
                &compiled,
                compiled.instantiate_local(inputs),
                &registry,
                &registry,
            )
            .unwrap();
    }

    let device = DeviceIdentity::new(9);
    let executor = FrameExecutor::new(TestRhi::new(capabilities(), device));
    let mut registry = TestRegistry::new(device);
    let binding = TextureBindingId::new(2);
    registry.register_texture(
        binding,
        TestTexture::new(2),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(imported.slot, binding);

    match executor.execute(
        &compiled,
        compiled.instantiate_local(inputs),
        &registry,
        &registry,
    ) {
        Err(ExecutionError::FrameBinding(error)) => {
            assert_eq!(error.kind, FrameBindingErrorKind::UsageMismatch);
            assert_eq!(error.texture_slot, Some(imported.slot));
            assert_eq!(error.resource, Some(logical));
            assert!(error.detail.contains("required TextureUsage"));
            assert!(error.detail.contains("actual TextureUsage"));
            assert!(error.detail.contains("CopySource"));
        }
        Err(other) => panic!("unexpected execution error: {other}"),
        Ok(_) => panic!("insufficient imported texture usage unexpectedly executed"),
    }
}

#[test]
fn transient_texture_with_insufficient_allowed_usage_fails_before_recording() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    let image = graph.create_texture("image", texture());
    let pass = graph.add_compute_pass(
        "write",
        |pass| {
            let (output, _) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let logical = compiled.execution_plan().resource_requirements()[0].resource;
    let device = DeviceIdentity::new(9);
    let mut backend = TestRhi::new(capabilities(), device);
    backend.set_transient_texture_usage(TextureUsage::from_kinds([TextureUsageKind::StorageWrite]));
    let executor = FrameExecutor::new(backend);
    let registry = TestRegistry::new(device);

    match executor.execute(
        &compiled,
        compiled.instantiate_local(FrameInputs::new(())),
        &registry,
        &registry,
    ) {
        Err(ExecutionError::FrameBinding(error)) => {
            assert_eq!(error.kind, FrameBindingErrorKind::UsageMismatch);
            assert_eq!(error.resource, Some(logical));
            assert!(error.texture_slot.is_none());
        }
        Err(other) => panic!("unexpected execution error: {other}"),
        Ok(_) => panic!("insufficient transient texture usage unexpectedly executed"),
    }
    assert!(
        !executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::BeginEncoder { .. }))
    );
}

#[test]
fn culled_import_needs_no_binding_or_usage_validation() {
    let mut graph: RenderGraph<()> = RenderGraph::new();
    graph.import_texture_slot(
        "dead import",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let device = DeviceIdentity::new(9);
    let executor = FrameExecutor::new(TestRhi::new(capabilities(), device));
    let registry = TestRegistry::new(device);

    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
}
