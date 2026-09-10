//! Completion and provider-boundary validation cases.

use super::*;

struct CountingProvider {
    registry: TestRegistry,
    calls: Arc<AtomicUsize>,
}

impl FrameResourceProvider<TestRhi> for CountingProvider {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TestTexture, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.registry.texture(id)
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<TestBuffer, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.registry.buffer(id)
    }
}

#[test]
fn missing_import_precheck_has_no_provider_or_transient_side_effects() {
    let mut graph = RenderGraph::new();
    let transient = graph.create_buffer("transient", buffer(8));
    let imported = graph.import_buffer_slot(
        "required",
        ImportBufferContract {
            descriptor: buffer(8),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "write-and-read",
        |pass| {
            let (output, _) = pass.write_buffer(
                transient,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = CountingProvider {
        registry: TestRegistry::new(device()),
        calls: Arc::clone(&calls),
    };
    let executor = executor();

    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &provider,
            &provider.registry,
        ),
        Err(ExecutionError::FrameBinding(error)) if error.kind == FrameBindingErrorKind::MissingBuffer
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        !executor.try_backend().unwrap().trace().iter().any(|event| {
            matches!(
                event,
                TestTraceEvent::CreateTexture { .. } | TestTraceEvent::CreateBuffer { .. }
            )
        })
    );
}

#[test]
fn missing_texture_and_provider_device_mismatch_fail_before_recording() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_texture_slot(
        "input",
        ImportTextureContract {
            descriptor: texture(TextureFormat::Rgba8Unorm),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "read",
        |pass| {
            pass.read_texture(
                &imported.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&caps()).unwrap().graph;
    let empty = TestRegistry::new(device());
    assert!(matches!(
        executor().execute(&compiled, compiled.instantiate_local(FrameInputs::new(())), &empty, &empty),
        Err(ExecutionError::FrameBinding(error)) if error.kind == FrameBindingErrorKind::MissingTexture
    ));
    let mut registry = TestRegistry::new(device());
    registry.register_texture(
        TextureBindingId::new(1),
        TestTexture::new(1),
        texture(TextureFormat::Rgba8Unorm),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    let provider = WrongDeviceProvider(registry);
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(imported.slot, TextureBindingId::new(1));
    assert!(matches!(
        executor().execute(&compiled, compiled.instantiate_local(inputs), &provider, &provider),
        Err(ExecutionError::FrameBinding(error)) if error.kind == FrameBindingErrorKind::DeviceMismatch
    ));
}

#[test]
fn binding_provider_device_mismatch_is_a_recording_error() {
    let recipe = BindingSetId::new(12);
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(8),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "binding",
        |pass| {
            let read =
                pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            ((), read)
        },
        move |_, resolver, read, _| {
            resolver.resolve_bindings(recipe, &[BindingResource::BufferRead(read)], &[])?;
            Ok(())
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&caps()).unwrap().graph;
    let mut registry = TestRegistry::new(device());
    registry.register_buffer(
        BufferBindingId::new(12),
        TestBuffer::new(12),
        buffer(8),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry.register_bindings(recipe, fluxel_rendergraph::test_rhi::TestBindings::new(12));
    let provider = WrongBindingProvider(registry);
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(input.slot, BufferBindingId::new(12));
    assert!(matches!(
        executor().execute(&compiled, compiled.instantiate_local(inputs), &provider, &provider),
        Err(ExecutionError::Recording(error))
            if error.kind == RecordingErrorKind::IncompatibleBindingRecipe
    ));
}

#[test]
fn every_nonempty_frame_finishes_and_submits_once() {
    let mut graph = RenderGraph::new();
    let buffer = graph.create_buffer("data", buffer(8));
    let pass = graph.add_compute_pass(
        "write",
        |pass| {
            let (output, _) = pass.write_buffer(
                buffer,
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
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert_eq!(
        trace
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::BeginEncoder { .. }))
            .count(),
        1
    );
    assert_eq!(
        trace
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::Finish { .. }))
            .count(),
        1
    );
    assert_eq!(
        trace
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::Submit { .. }))
            .count(),
        1
    );
}

#[test]
fn distinct_logical_imports_cannot_alias_one_physical_generation() {
    let mut graph = RenderGraph::new();
    let contract = ImportBufferContract {
        descriptor: buffer(8),
        initial_state: ResourceAccessState::ShaderStorageRead,
        ownership: ExternalOwnership::Caller,
        initial_contents: InitialContents::Defined,
    };
    let first = graph.import_buffer_slot("first", contract);
    let second = graph.import_buffer_slot("second", contract);
    let pass = graph.add_compute_pass(
        "read-both",
        |pass| {
            pass.read_buffer(&first.version, BufferReadUse::Storage, BufferRange::whole());
            pass.read_buffer(
                &second.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&caps()).unwrap().graph;
    let binding = BufferBindingId::new(91);
    let mut registry = TestRegistry::new(device());
    registry.register_buffer(
        binding,
        TestBuffer::new(91),
        buffer(8),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(first.slot, binding);
    inputs.bind_buffer(second.slot, binding);
    assert!(matches!(
        executor().execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::FrameBinding(error))
            if error.kind == FrameBindingErrorKind::AliasedPhysicalResource
    ));
}

#[test]
fn failed_completion_is_terminal_and_trace_collection_can_be_disabled() {
    let mut graph = RenderGraph::new();
    let value = graph.create_buffer("value", buffer(8));
    let pass = graph.add_compute_pass(
        "write",
        |pass| {
            let (value, _) = pass.write_buffer(
                value,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (value, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let mut backend = TestRhi::new(caps(), device());
    backend.set_trace_enabled(false);
    let executor = FrameExecutor::new(backend);
    let mut frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    let completion = *frame.submission.completion();
    executor.try_backend().unwrap().fail(completion);
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Failed(CompletionFailure::ExecutionFailed)
    );
    assert_eq!(frame.submission.retained_lease_count(), 0);
    assert!(executor.try_backend().unwrap().trace().is_empty());
}

#[test]
fn callback_backend_diagnostics_fail_fast_instead_of_deadlocking() {
    let executor = Arc::new(executor());
    let callback_executor = Arc::clone(&executor);
    let observed_busy = Arc::new(AtomicBool::new(false));
    let callback_observed = Arc::clone(&observed_busy);
    let mut graph = RenderGraph::new();
    let pass = graph.add_compute_pass(
        "reentrant-diagnostic",
        |_| ((), ()),
        move |_, _, _, _| {
            callback_observed.store(callback_executor.try_backend().is_none(), Ordering::SeqCst);
            Ok(())
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    assert!(observed_busy.load(Ordering::SeqCst));
}
