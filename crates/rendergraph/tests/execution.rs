use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use fluxel_rendergraph::{
    test_rhi::{
        TestBuffer, TestLeaseProbe, TestRasterPipeline, TestRegistry, TestRhi, TestRhiError,
        TestTraceEvent,
    },
    *,
};

fn device() -> DeviceIdentity {
    DeviceIdentity::new(7)
}

fn caps() -> DeviceCapabilities {
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
        .build()
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 64 }
}

fn executor() -> FrameExecutor<TestRhi> {
    FrameExecutor::new(TestRhi::new(caps(), device()))
}

fn imported_buffer_graph() -> (CompiledGraph, ImportBufferSlot, ExportBufferSlot) {
    let mut graph = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::CopySource,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_buffer("target", buffer());
    let copied = graph.add_copy_pass(
        "copy",
        |pass| {
            let src = pass.read_buffer(&imported.version, BufferRange::whole());
            let (out, dst) = pass.write_buffer(target, BufferRange::whole(), WriteCoverage::Full);
            (out, (src, dst))
        },
        |commands, _, data, _| {
            commands.copy_buffer(
                &data.0,
                &data.1,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 32,
                },
            )
        },
    );
    let compute = graph.add_compute_pass(
        "compute",
        |pass| {
            pass.read_buffer(&copied.output, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |commands, _, _, _| commands.dispatch([2, 1, 1]),
    );
    graph.mark_side_effect(
        compute.id,
        SideEffectReason::Diagnostic("retain compute".into()),
    );
    let export = graph.export_buffer(
        copied.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageRead,
        },
    );
    (graph.compile(&caps()).unwrap().graph, imported.slot, export)
}

fn registry_with_input() -> (TestRegistry, BufferBindingId, TestLeaseProbe) {
    let mut registry = TestRegistry::new(device());
    let id = BufferBindingId::new(1);
    let probe = registry.register_buffer(
        id,
        TestBuffer::new(1),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::CopySource]),
        ResourceAccessState::CopySource,
    );
    (registry, id, probe)
}

#[test]
fn records_single_queue_copy_compute_transitions_and_export() {
    let (graph, slot, export) = imported_buffer_graph();
    let (registry, input, _) = registry_with_input();
    let executor = executor();
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(slot, input);
    let frame = executor
        .execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();

    assert!(frame.exports.buffer(export).is_some());
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
            .filter(|event| matches!(event, TestTraceEvent::Submit { .. }))
            .count(),
        1
    );
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::BeginCopy { label } if label == "copy"))
    );
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::CopyBuffer { .. }))
    );
    assert!(trace.iter().any(
        |event| matches!(event, TestTraceEvent::BeginCompute { label } if label == "compute")
    ));
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::Dispatch { groups: [2, 1, 1] }))
    );
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::TransitionBuffer {
            before: ResourceAccessState::CopyDestination,
            after: ResourceAccessState::ShaderStorageRead,
            ..
        }
    )));
}

#[test]
fn raster_callback_runs_only_when_retained_and_records_pipeline_draw() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let pipeline = RasterPipelineId::new(3);
    let mut graph = RenderGraph::new();
    let image = graph.create_texture(
        "color",
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
        },
    );
    let pass = graph.add_raster_pass(
        "raster",
        |pass| {
            let output = pass.color_attachment(
                image,
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
        move |commands, _, _, _| {
            seen.fetch_add(1, Ordering::SeqCst);
            commands.set_pipeline(pipeline)?;
            commands.draw(0..3, 0..1)
        },
    );
    graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let mut registry = TestRegistry::new(device());
    registry.register_raster_pipeline(pipeline, TestRasterPipeline::new(9));
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::BeginRaster {
            color_attachments: 1,
            ..
        }
    )));
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::SetRasterPipeline { .. }))
    );
    assert!(trace.iter().any(
        |event| matches!(event, TestTraceEvent::Draw { vertices, .. } if vertices == &(0..3))
    ));
}

#[test]
fn capability_mismatch_and_missing_input_fail_before_submission() {
    let (graph, slot, _) = imported_buffer_graph();
    let (registry, input, _) = registry_with_input();
    let mut incompatible_caps = caps();
    incompatible_caps.limits = DeviceLimits::new(5, 256);
    let mismatch = FrameExecutor::new(TestRhi::new(incompatible_caps, device()));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(slot, input);
    assert!(matches!(
        mismatch.execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry
        ),
        Err(ExecutionError::CapabilityMismatch)
    ));

    let executor = executor();
    assert!(
        matches!(executor.execute(&graph, graph.instantiate_local(FrameInputs::new(())), &registry, &registry), Err(ExecutionError::FrameBinding(error)) if error.kind == FrameBindingErrorKind::MissingBuffer)
    );
    assert!(executor.try_backend().unwrap().trace().is_empty());
}

#[test]
fn canonical_capability_reordering_is_execution_compatible() {
    let mut compile_caps = caps();
    compile_caps.queues.push(QueueDescriptor::new(
        QueueId::new(9),
        QueueCapabilities::new(false, true, true, false),
    ));
    compile_caps.texture_formats.push(
        TextureFormatCapabilities::builder(TextureFormat::Rgba16Float)
            .storage(true, true)
            .build(),
    );
    let mut graph = RenderGraph::new();
    let buffer = graph.create_buffer("transient", buffer());
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
    let compiled = graph.compile(&compile_caps).unwrap().graph;
    let mut reordered = compile_caps;
    reordered.queues.reverse();
    reordered.texture_formats.reverse();
    let executor = FrameExecutor::new(TestRhi::new(reordered, device()));
    let registry = TestRegistry::new(device());
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
}

#[test]
fn imported_binding_rejects_descriptor_state_and_device_mismatches() {
    let (graph, slot, _) = imported_buffer_graph();
    let id = BufferBindingId::new(11);
    for (registry_device, descriptor, state, expected) in [
        (
            DeviceIdentity::new(99),
            buffer(),
            ResourceAccessState::CopySource,
            FrameBindingErrorKind::DeviceMismatch,
        ),
        (
            device(),
            BufferDesc { size: 65 },
            ResourceAccessState::CopySource,
            FrameBindingErrorKind::DescriptorMismatch,
        ),
        (
            device(),
            buffer(),
            ResourceAccessState::ShaderStorageRead,
            FrameBindingErrorKind::InitialStateMismatch,
        ),
    ] {
        let mut registry = TestRegistry::new(registry_device);
        registry.register_buffer(
            id,
            TestBuffer::new(11),
            descriptor,
            BufferUsage::from_kinds([BufferUsageKind::StorageRead, BufferUsageKind::CopySource]),
            state,
        );
        let mut inputs = FrameInputs::new(());
        inputs.bind_buffer(slot, id);
        match executor().execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry,
        ) {
            Err(ExecutionError::FrameBinding(error)) => assert_eq!(error.kind, expected),
            Err(other) => panic!("unexpected execution error: {other}"),
            Ok(_) => panic!("mismatched imported binding unexpectedly executed"),
        }
    }
}

#[test]
fn dropped_submission_retires_leases_until_completion_is_collected() {
    let (graph, slot, _) = imported_buffer_graph();
    let (registry, input, probe) = registry_with_input();
    let executor = executor();
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(slot, input);
    let frame = executor
        .execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    let completion = *frame.submission.completion();
    assert!(probe.active_leases() > 1);
    drop(frame);
    assert_eq!(executor.try_backend().unwrap().retired_count(), 0);
    assert_eq!(executor.collect_retired().unwrap(), 0);
    assert_eq!(executor.try_backend().unwrap().retired_count(), 1);
    executor.try_backend().unwrap().complete(completion);
    assert_eq!(executor.collect_retired().unwrap(), 1);
    assert_eq!(
        probe.active_leases(),
        1,
        "registry's own lease remains, frame leases retired"
    );
}

#[test]
fn backend_failure_does_not_submit() {
    let (graph, slot, _) = imported_buffer_graph();
    let (registry, input, _) = registry_with_input();
    let executor = executor();
    executor
        .try_backend()
        .unwrap()
        .fail_next(TestRhiError::new("begin fails"));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(slot, input);
    assert!(matches!(
        executor.execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry
        ),
        Err(ExecutionError::Backend(_))
    ));
    assert!(
        !executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::Submit { .. }))
    );
}
