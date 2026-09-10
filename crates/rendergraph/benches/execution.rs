//! Benchmarks immutable-plan traversal and serial test-backend execution overhead.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fluxel_rendergraph::{test_rhi::*, *};

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, false, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .limits(DeviceLimits::new(1, 256))
        .buffers(BufferCapabilities::new(true, true, false))
        .build()
}

fn compiled(pass_count: usize, capabilities: &DeviceCapabilities) -> CompiledGraph {
    let mut graph = RenderGraph::new();
    let mut version = graph.create_buffer("execution", BufferDesc { size: 256 });
    for index in 0..pass_count {
        version = graph
            .add_compute_pass(
                format!("write-{index}"),
                |pass| {
                    let (output, _) = pass.write_buffer(
                        version,
                        BufferWriteUse::Storage,
                        BufferRange::whole(),
                        WriteCoverage::Full,
                    );
                    (output, ())
                },
                |_, _, _, _| Ok(()),
            )
            .output;
    }
    graph.export_buffer(
        version,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageRead,
        },
    );
    graph.compile(capabilities).unwrap().graph
}

fn execute_once(executor: &FrameExecutor<TestRhi>, graph: &CompiledGraph, registry: &TestRegistry) {
    let mut frame = executor
        .execute(
            graph,
            graph.instantiate_local(FrameInputs::new(())),
            registry,
            registry,
        )
        .unwrap();
    let completion = *frame.submission.completion();
    executor.try_backend().unwrap().complete(completion);
    black_box(frame.submission.status().unwrap());
}

fn benchmarks(c: &mut Criterion) {
    let capabilities = capabilities();
    let device = DeviceIdentity::new(1);
    let registry = TestRegistry::new(device);

    let mut group = c.benchmark_group("execution_plan");
    for &pass_count in &[128_usize, 1_024] {
        let graph = compiled(pass_count, &capabilities);
        group.bench_function(format!("traverse/{pass_count}"), |b| {
            b.iter(|| {
                for pass in graph.execution_plan().passes() {
                    black_box(pass);
                }
                black_box(graph.execution_plan().final_transitions());
            });
        });

        for &trace_enabled in &[false, true] {
            let mut backend = TestRhi::new(capabilities.clone(), device);
            backend.set_trace_enabled(trace_enabled);
            let executor = FrameExecutor::new(backend);
            group.bench_function(
                format!("serial_test_rhi/{pass_count}/trace_{trace_enabled}"),
                |b| b.iter(|| execute_once(&executor, &graph, &registry)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
