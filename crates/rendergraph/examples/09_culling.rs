//! Compile-only fixture for root-driven pass culling; run with `cargo run --example 09_culling`.

mod common;

use fluxel_rendergraph::*;

struct CopyData {
    source: BufferRead,
    destination: BufferWrite,
}

fn copy_pass(
    graph: &mut RenderGraph,
    name: &'static str,
    source: BufferVersion,
    destination: BufferVersion,
) -> DeclaredPass<BufferVersion> {
    graph.add_copy_pass(
        name,
        |pass| {
            let source = pass.read_buffer(&source, BufferRange::whole());
            let (destination, destination_access) =
                pass.write_buffer(destination, BufferRange::whole(), WriteCoverage::Full);
            (
                destination,
                CopyData {
                    source,
                    destination: destination_access,
                },
            )
        },
        |commands, _resolver, data, _frame| {
            commands.copy_buffer(
                &data.source,
                &data.destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 256,
                },
            )
        },
    )
}

fn main() {
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        common::imported_buffer(256, ResourceAccessState::CopySource),
    );
    let dead_input = graph.import_buffer_slot(
        "dead-input",
        common::imported_buffer(256, ResourceAccessState::CopySource),
    );
    let live_a = graph.create_buffer("live-a", common::buffer(256));
    let live_b = graph.create_buffer("live-b", common::buffer(256));
    let dead_a = graph.create_buffer("dead-a", common::buffer(256));
    let dead_b = graph.create_buffer("dead-b", common::buffer(256));
    let dead_c = graph.create_buffer("dead-c", common::buffer(256));

    let live_a = copy_pass(&mut graph, "live-a", input.version, live_a);
    let live_b = copy_pass(&mut graph, "live-b", live_a.output, live_b);

    // This disconnected chain has no export, presentation, or side effect root.
    // The compiler lists all three disconnected passes in CompileReport::culled_passes.
    let dead_a = copy_pass(&mut graph, "dead-a", dead_input.version, dead_a);
    let dead_b = copy_pass(&mut graph, "dead-b", dead_a.output, dead_b);
    let _dead_c = copy_pass(&mut graph, "dead-c", dead_b.output, dead_c);

    graph.export_buffer(
        live_b.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
