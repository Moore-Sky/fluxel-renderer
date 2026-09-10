//! Benchmarks graph compilation for representative linear and fan-out topologies.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use fluxel_rendergraph::*;

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
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(1, 256))
        .buffers(BufferCapabilities::new(true, true, false))
        .build()
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 256 }
}

fn write_pass(graph: &mut RenderGraph, input: BufferVersion) -> DeclaredPass<BufferVersion> {
    graph.add_compute_pass(
        "write",
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

fn read_pass(graph: &mut RenderGraph, input: &BufferVersion) -> DeclaredPass<()> {
    graph.add_compute_pass(
        "read",
        |pass| {
            pass.read_buffer(input, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    )
}

/// Builds a retained write-after-write chain with exactly `pass_count` passes.
fn retained_linear(pass_count: usize) -> RenderGraph {
    assert!(pass_count > 0);
    let mut graph = RenderGraph::new();
    let mut version = graph.create_buffer("linear", buffer());
    for _ in 0..pass_count {
        version = write_pass(&mut graph, version).output;
    }
    graph.export_buffer(
        version,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    graph
}

/// Builds a retained one-to-many graph with exactly `pass_count` passes.
fn retained_fan_out(pass_count: usize) -> RenderGraph {
    assert!(pass_count > 1);
    let mut graph = RenderGraph::new();
    let input = graph.create_buffer("fan-out", buffer());
    let producer = write_pass(&mut graph, input);
    for _ in 1..pass_count {
        let reader = read_pass(&mut graph, &producer.output);
        graph.mark_side_effect(reader.id, SideEffectReason::Diagnostic("benchmark".into()));
    }
    graph
}

fn benchmarks(c: &mut Criterion) {
    let capabilities = capabilities();
    let mut group = c.benchmark_group("compile");

    for &pass_count in &[128_usize, 1_024] {
        group.bench_function(format!("retained_linear/{pass_count}"), |b| {
            b.iter_batched(
                || retained_linear(pass_count),
                |graph| black_box(graph.compile(&capabilities).expect("graph should compile")),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(format!("retained_fan_out/{pass_count}"), |b| {
            b.iter_batched(
                || retained_fan_out(pass_count),
                |graph| black_box(graph.compile(&capabilities).expect("graph should compile")),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
