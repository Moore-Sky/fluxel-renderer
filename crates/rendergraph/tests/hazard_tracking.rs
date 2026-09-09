use fluxel_rendergraph::*;

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, true, false),
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
                .storage(true, true)
                .build(),
        )
        .build()
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 64 }
}

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

fn same_state_transitions(plan: &ExecutionPlan, pass: usize) -> Vec<&PlannedTransition> {
    plan.passes()[pass]
        .transitions
        .iter()
        .filter(|transition| transition.before == transition.after)
        .collect()
}

#[test]
fn storage_write_to_storage_write_emits_same_state_memory_barrier() {
    let mut graph = RenderGraph::<()>::new();
    let buffer = graph.create_buffer("buffer", buffer());
    let first = graph.add_compute_pass(
        "first",
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
    let second = graph.add_compute_pass(
        "second",
        |pass| {
            let (output, _) = pass.write_buffer(
                first.output,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        second.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let barriers = same_state_transitions(compiled.execution_plan(), 1);
    assert_eq!(barriers.len(), 1);
    assert_eq!(barriers[0].before, ResourceAccessState::ShaderStorageWrite);
}

#[test]
fn texture_storage_write_to_storage_write_emits_same_state_memory_barrier() {
    let mut graph = RenderGraph::<()>::new();
    let texture = graph.create_texture("texture", texture());
    let first = graph.add_compute_pass(
        "first",
        |pass| {
            let (output, _) = pass.write_texture(
                texture,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    let second = graph.add_compute_pass(
        "second",
        |pass| {
            let (output, _) = pass.write_texture(
                first.output,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        second.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let barriers = same_state_transitions(compiled.execution_plan(), 1);
    assert_eq!(barriers.len(), 1);
    assert_eq!(barriers[0].before, ResourceAccessState::ShaderStorageWrite);
    assert!(matches!(
        barriers[0].range,
        PlannedResourceRange::Texture(_)
    ));
}

#[test]
fn storage_read_write_to_storage_read_write_emits_same_state_memory_barrier() {
    let mut graph = RenderGraph::<()>::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let first = graph.add_compute_pass(
        "first",
        |pass| {
            let (output, _) = pass.read_write_buffer(
                input.version,
                BufferReadWriteUse::Storage,
                BufferRange::whole(),
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    let second = graph.add_compute_pass(
        "second",
        |pass| {
            let (output, _) = pass.read_write_buffer(
                first.output,
                BufferReadWriteUse::Storage,
                BufferRange::whole(),
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        second.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageReadWrite,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let barriers = same_state_transitions(compiled.execution_plan(), 1);
    assert_eq!(barriers.len(), 1);
    assert_eq!(
        barriers[0].before,
        ResourceAccessState::ShaderStorageReadWrite
    );
}

#[test]
fn storage_read_to_storage_read_does_not_emit_same_state_memory_barrier() {
    let mut graph = RenderGraph::<()>::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let first = graph.add_compute_pass(
        "first",
        |pass| {
            pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    let second = graph.add_compute_pass(
        "second",
        |pass| {
            pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(first.id, SideEffectReason::Diagnostic("first".into()));
    graph.mark_side_effect(second.id, SideEffectReason::Diagnostic("second".into()));
    graph.depends_on(
        first.id,
        second.id,
        ExplicitOrderReason::Diagnostic("test ordering".into()),
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    assert!(same_state_transitions(compiled.execution_plan(), 1).is_empty());
}

#[test]
fn disjoint_storage_writes_do_not_emit_a_same_state_memory_barrier() {
    let mut graph = RenderGraph::<()>::new();
    let buffer = graph.create_buffer("buffer", buffer());
    let first = graph.add_compute_pass(
        "first",
        |pass| {
            let (output, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::Bytes {
                    offset: 0,
                    size: 32,
                },
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    let second = graph.add_compute_pass(
        "second",
        |pass| {
            let (output, _) = pass.write_buffer(
                first.output,
                BufferWriteUse::Storage,
                BufferRange::Bytes {
                    offset: 32,
                    size: 32,
                },
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        second.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let compiled = graph.compile(&capabilities()).unwrap().graph;
    assert!(same_state_transitions(compiled.execution_plan(), 1).is_empty());
}

#[test]
fn overlapping_same_pass_reads_with_incompatible_states_are_rejected() {
    let mut graph = RenderGraph::<()>::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::UniformRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "incompatible reads",
        |pass| {
            pass.read_buffer(&input.version, BufferReadUse::Uniform, BufferRange::whole());
            pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));

    assert_eq!(
        graph.compile(&capabilities()).unwrap_err().kind,
        CompileErrorKind::ConflictingAccess
    );
}
