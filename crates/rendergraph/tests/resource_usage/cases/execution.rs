//! Execution-facing resource-usage requirement cases.

use super::*;

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
