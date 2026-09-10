//! Integration tests for dependency inference, culling, ordering, and plan stability.

use fluxel_rendergraph::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn capabilities() -> DeviceCapabilities {
    let rgba8 = TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
        .sampled(true, true)
        .storage(true, true)
        .attachments(true, false, vec![1])
        .copies(true, true)
        .build();
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, true, true, true),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(true, true, true))
        .texture_format(rgba8)
        .surface(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            true,
            true,
        ))
        .build()
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 256 }
}

fn write_pass(
    graph: &mut RenderGraph,
    name: &'static str,
    input: BufferVersion,
) -> DeclaredPass<BufferVersion> {
    graph.add_compute_pass(
        name,
        |pass| {
            let (output, _) = pass.write_buffer(
                input,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    )
}

fn read_pass(
    graph: &mut RenderGraph,
    name: &'static str,
    input: &BufferVersion,
) -> DeclaredPass<()> {
    graph.add_compute_pass(
        name,
        |pass| {
            pass.read_buffer(input, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    )
}

#[test]
fn raw_dependency_is_inferred_and_topologically_ordered() {
    let mut graph = RenderGraph::new();
    let x = graph.create_buffer("x", buffer());
    let producer = write_pass(&mut graph, "producer", x);
    let consumer = read_pass(&mut graph, "consumer", &producer.output);
    graph.mark_side_effect(consumer.id, SideEffectReason::Diagnostic("test".into()));

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert_eq!(output.graph.execution_order(), &[producer.id, consumer.id]);
    assert!(
        output
            .report
            .inferred_dependencies
            .iter()
            .any(|edge| { edge.producer == producer.id && edge.consumer == consumer.id })
    );
}

#[test]
fn fan_out_keeps_both_readers_after_the_producer() {
    let mut graph = RenderGraph::new();
    let x = graph.create_buffer("x", buffer());
    let producer = write_pass(&mut graph, "producer", x);
    let first = read_pass(&mut graph, "first-reader", &producer.output);
    let second = read_pass(&mut graph, "second-reader", &producer.output);
    graph.mark_side_effect(first.id, SideEffectReason::Diagnostic("test".into()));
    graph.mark_side_effect(second.id, SideEffectReason::Diagnostic("test".into()));

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert_eq!(
        output.graph.execution_order(),
        &[producer.id, first.id, second.id]
    );
    for reader in [first.id, second.id] {
        assert!(
            output
                .report
                .inferred_dependencies
                .iter()
                .any(|edge| { edge.producer == producer.id && edge.consumer == reader })
        );
    }
}

#[test]
fn war_orders_old_version_reader_before_overwrite() {
    let mut graph = RenderGraph::new();
    let x = graph.create_buffer("x", buffer());
    let first = write_pass(&mut graph, "initialize", x);
    let observer = read_pass(&mut graph, "observe-old-version", &first.output);
    let overwrite = write_pass(&mut graph, "overwrite", first.output);
    graph.mark_side_effect(observer.id, SideEffectReason::Diagnostic("test".into()));
    graph.export_buffer(
        overwrite.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert_eq!(
        output.graph.execution_order(),
        &[first.id, observer.id, overwrite.id]
    );
    assert!(
        output
            .report
            .inferred_dependencies
            .iter()
            .any(|edge| { edge.producer == observer.id && edge.consumer == overwrite.id })
    );
}

#[test]
fn disconnected_passes_are_culled() {
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::CopySource,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let live = write_pass(&mut graph, "live", input.version);
    let dead_resource = graph.create_buffer("dead", buffer());
    let dead = write_pass(&mut graph, "dead", dead_resource);
    graph.export_buffer(
        live.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert_eq!(output.graph.execution_order(), &[live.id]);
    assert_eq!(output.report.culled_passes, vec![dead.id]);
}

#[test]
fn a_dead_old_version_reader_does_not_become_live_through_war() {
    let mut graph = RenderGraph::new();
    let x = graph.create_buffer("x", buffer());
    let initial = write_pass(&mut graph, "initialize", x);
    let dead_reader = read_pass(&mut graph, "dead-reader", &initial.output);
    let overwrite = write_pass(&mut graph, "overwrite", initial.output);
    graph.export_buffer(
        overwrite.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert!(output.report.culled_passes.contains(&dead_reader.id));
    assert_eq!(output.graph.execution_order(), &[initial.id, overwrite.id]);
}

#[test]
fn side_effect_retains_an_otherwise_disconnected_pass() {
    let mut graph = RenderGraph::<()>::new();
    let pass = graph.add_copy_pass("publish", |_| ((), ()), |_, _, _, _| Ok(()));
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::ExternalProtocol("publish".into()),
    );

    let output = graph
        .compile(&capabilities())
        .expect("graph should compile");
    assert_eq!(output.graph.execution_order(), &[pass.id]);
    assert!(output.report.culled_passes.is_empty());
    assert_eq!(
        output.report.retained_side_effects,
        vec![RetainedSideEffect {
            pass: pass.id,
            reason: SideEffectReason::ExternalProtocol("publish".into()),
        }]
    );
}

#[test]
fn explicit_order_is_retained_and_cycles_are_rejected() {
    let mut graph = RenderGraph::<()>::new();
    let before = graph.add_copy_pass("before", |_| ((), ()), |_, _, _, _| Ok(()));
    let after = graph.add_copy_pass("after", |_| ((), ()), |_, _, _, _| Ok(()));
    graph.mark_side_effect(before.id, SideEffectReason::ExternalProtocol("test".into()));
    graph.mark_side_effect(after.id, SideEffectReason::ExternalProtocol("test".into()));
    graph.depends_on(
        before.id,
        after.id,
        ExplicitOrderReason::ExternalProtocol("ordered".into()),
    );

    let output = graph
        .compile(&capabilities())
        .expect("ordered graph should compile");
    assert_eq!(output.graph.execution_order(), &[before.id, after.id]);
    assert_eq!(output.report.explicit_orders.len(), 1);

    let mut cyclic = RenderGraph::<()>::new();
    let a = cyclic.add_copy_pass("a", |_| ((), ()), |_, _, _, _| Ok(()));
    let b = cyclic.add_copy_pass("b", |_| ((), ()), |_, _, _, _| Ok(()));
    cyclic.mark_side_effect(a.id, SideEffectReason::Diagnostic("test".into()));
    cyclic.mark_side_effect(b.id, SideEffectReason::Diagnostic("test".into()));
    cyclic.depends_on(
        a.id,
        b.id,
        ExplicitOrderReason::Diagnostic("a before b".into()),
    );
    cyclic.depends_on(
        b.id,
        a.id,
        ExplicitOrderReason::Diagnostic("b before a".into()),
    );
    assert_eq!(
        cyclic.compile(&capabilities()).unwrap_err().kind,
        CompileErrorKind::DependencyCycle
    );
}

#[test]
fn compile_returns_repeatable_immutable_snapshots() {
    let mut graph = RenderGraph::new();
    let x = graph.create_buffer("x", buffer());
    let producer = write_pass(&mut graph, "producer", x);
    let consumer = read_pass(&mut graph, "consumer", &producer.output);
    graph.mark_side_effect(consumer.id, SideEffectReason::Diagnostic("test".into()));

    let first = graph
        .compile(&capabilities())
        .expect("first compile should succeed");
    let second = graph
        .compile(&capabilities())
        .expect("second compile should succeed");
    assert_eq!(
        first.graph.execution_order(),
        second.graph.execution_order()
    );
    assert_eq!(first.report.culled_passes, second.report.culled_passes);
    assert_eq!(
        first.report.inferred_dependencies,
        second.report.inferred_dependencies
    );
}

#[test]
fn setup_runs_once_while_compile_never_records_commands() {
    let setup_calls = Arc::new(AtomicUsize::new(0));
    let execute_calls = Arc::new(AtomicUsize::new(0));
    let setup_counter = Arc::clone(&setup_calls);
    let execute_counter = Arc::clone(&execute_calls);
    let mut graph = RenderGraph::<()>::new();
    let pass = graph.add_copy_pass(
        "retained-recipe",
        move |_| {
            setup_counter.fetch_add(1, Ordering::SeqCst);
            ((), ())
        },
        move |_, _, _, _| {
            execute_counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));

    graph.compile(&capabilities()).expect("first compile");
    graph.compile(&capabilities()).expect("second compile");
    assert_eq!(setup_calls.load(Ordering::SeqCst), 1);
    assert_eq!(execute_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn foreign_versions_and_conflicting_same_pass_accesses_are_rejected() {
    let mut owner = RenderGraph::<()>::new();
    let foreign = owner.create_buffer("foreign", buffer());
    let mut graph = RenderGraph::<()>::new();
    let pass = write_pass(&mut graph, "foreign-writer", foreign);
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_eq!(
        graph.compile(&capabilities()).unwrap_err().kind,
        CompileErrorKind::StaleOrForeignVersion
    );

    let mut graph = RenderGraph::<()>::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "conflict",
        |builder| {
            let _read = builder.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            let (next, _write) = builder.write_buffer(
                imported.version,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_eq!(
        graph.compile(&capabilities()).unwrap_err().kind,
        CompileErrorKind::ConflictingAccess
    );
}

#[test]
fn same_pass_cannot_consume_its_own_successor() {
    let mut graph = RenderGraph::<()>::new();
    let buffer = graph.create_buffer("buffer", buffer());
    let pass = graph.add_compute_pass(
        "write-then-read",
        |builder| {
            let (next, write) = builder.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            let read = builder.read_buffer(&next, BufferReadUse::Storage, BufferRange::whole());
            (next, (write, read))
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageRead,
        },
    );

    assert_eq!(
        graph.compile(&capabilities()).unwrap_err().kind,
        CompileErrorKind::ConflictingAccess
    );
}
