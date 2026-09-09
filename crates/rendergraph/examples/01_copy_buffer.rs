//! CPU-only execution example; run with `cargo run --example 01_copy_buffer`.

mod common;

use fluxel_rendergraph::test_rhi::{TestBuffer, TestRegistry, TestRhi};
use fluxel_rendergraph::*;

struct CopyData {
    source: BufferRead,
    destination: BufferWrite,
}

fn main() {
    let mut graph = RenderGraph::new();
    let source = graph.import_buffer_slot(
        "upload-buffer",
        common::imported_buffer(256, ResourceAccessState::CopySource),
    );
    let destination = graph.create_buffer("copied-buffer", common::buffer(256));

    let copy = graph.add_copy_pass(
        "copy-buffer",
        |pass| {
            let source = pass.read_buffer(&source.version, BufferRange::whole());
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
            )?;
            Ok(())
        },
    );

    graph.export_buffer(
        copy.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let capabilities = common::single_queue_capabilities();
    let compiled = graph.compile(&capabilities).unwrap().graph;
    let device = DeviceIdentity::new(1);
    let mut registry = TestRegistry::new(device);
    let input = BufferBindingId::new(1);
    registry.register_buffer(
        input,
        TestBuffer::new(1),
        common::buffer(256),
        BufferUsage::from_kinds([BufferUsageKind::CopySource]),
        ResourceAccessState::CopySource,
    );
    let executor = FrameExecutor::new(TestRhi::new(capabilities, device));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(source.slot, input);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    let event_count = executor.try_backend().unwrap().trace().len();
    println!(
        "TestRhi executed copy: exports={}, trace_events={event_count}",
        frame.exports.buffers().count(),
    );
}
