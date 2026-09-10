//! Physical-identity and completion contract cases.

use super::*;

#[test]
fn texture_and_buffer_physical_identities_use_separate_namespaces() {
    let mut graph = RenderGraph::new();
    let imported_texture = graph.import_texture_slot(
        "texture",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let imported_buffer = graph.import_buffer_slot(
        "buffer",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "read-both-kinds",
        |pass| {
            pass.read_texture(
                &imported_texture.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            pass.read_buffer(
                &imported_buffer.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let graph = graph.compile(&capabilities()).unwrap().graph;
    let texture_id = TextureBindingId::new(31);
    let buffer_id = BufferBindingId::new(32);
    let mut registry = TestRegistry::new(device());
    registry.register_texture(
        texture_id,
        fluxel_rendergraph::test_rhi::TestTexture::new(1),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    registry.register_buffer(
        buffer_id,
        TestBuffer::new(1),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(imported_texture.slot, texture_id);
    inputs.bind_buffer(imported_buffer.slot, buffer_id);

    executor()
        .execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
}

#[test]
fn resolver_rejects_a_handle_declared_by_a_different_pass() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let foreign = graph.add_compute_pass(
        "source",
        |pass| {
            let read = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            (read, ())
        },
        |_, _, _, _| Ok(()),
    );
    let retained = graph.add_compute_pass(
        "consumer",
        |_| ((), foreign.output),
        |_, resolver, foreign, _| {
            resolver.resolve_bindings(
                BindingSetId::new(91),
                &[BindingResource::BufferRead(foreign)],
                &[],
            )?;
            Ok(())
        },
    );
    graph.mark_side_effect(retained.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());
    assert!(matches!(
        executor().execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Recording(error))
            if error.kind == RecordingErrorKind::ForeignOrUndeclaredPassAccess
    ));
}
#[test]
fn two_pending_completions_are_independent_and_local_send_instantiate() {
    let (graph, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    let executor = executor();
    let mut first_input = FrameInputs::new(1);
    first_input.bind_buffer(slot, BufferBindingId::new(1));
    let mut second_input = FrameInputs::new(1);
    second_input.bind_buffer(slot, BufferBindingId::new(2));
    let mut first = executor
        .execute(
            &graph,
            graph.instantiate_local(first_input),
            &registry,
            &registry,
        )
        .unwrap();
    let mut second = executor
        .execute(
            &graph,
            graph.instantiate_send(second_input),
            &registry,
            &registry,
        )
        .unwrap();
    let completion: TestCompletion = *first.submission.completion();
    assert_eq!(
        first.submission.status().unwrap(),
        CompletionStatus::Pending
    );
    assert_eq!(
        second.submission.status().unwrap(),
        CompletionStatus::Pending
    );
    executor.try_backend().unwrap().complete(completion);
    assert_eq!(
        first.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    assert_eq!(
        second.submission.status().unwrap(),
        CompletionStatus::Pending
    );
}

#[test]
fn wrong_graph_and_surface_execution_are_rejected() {
    let (first, _) = imported_compute_graph();
    let (second, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    let mut inputs = FrameInputs::new(1);
    inputs.bind_buffer(slot, BufferBindingId::new(1));
    assert!(matches!(
        executor().execute(
            &first,
            second.instantiate_local(inputs),
            &registry,
            &registry
        ),
        Err(ExecutionError::WrongCompiledGraph)
    ));

    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut surface = RenderGraph::new();
    let image = surface.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = surface.add_raster_pass(
        "surface",
        |pass| {
            let out = pass.color_attachment(
                image.version,
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
            (out, ())
        },
        |_, _, _, _| Ok(()),
    );
    surface.present(pass.output, PresentContract::new());
    let compiled = surface.compile(&caps).unwrap().graph;
    let executor = FrameExecutor::new(TestRhi::new(caps, device()));
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &TestRegistry::new(device()),
            &TestRegistry::new(device())
        ),
        Err(ExecutionError::UnsupportedExecutionFeature(_))
    ));
}
