//! CPU-only execution example; run with `cargo run --example 02_compute_buffer`.

mod common;

use fluxel_rendergraph::pass::BindingResource;
use fluxel_rendergraph::test_rhi::{
    TestBindings, TestBuffer, TestComputePipeline, TestRegistry, TestRhi,
};
use fluxel_rendergraph::*;

struct ComputeData {
    input: BufferRead,
    output: BufferWrite,
}

fn main() {
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        common::imported_buffer(1024, ResourceAccessState::ShaderStorageRead),
    );
    let output = graph.create_buffer("output", common::buffer(1024));

    let compute = graph.add_compute_pass(
        "transform",
        |pass| {
            let input =
                pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            let (output, output_access) = pass.write_buffer(
                output,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (
                output,
                ComputeData {
                    input,
                    output: output_access,
                },
            )
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(1),
                &[
                    BindingResource::BufferRead(&data.input),
                    BindingResource::BufferWrite(&data.output),
                ],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(1))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([16, 1, 1])?;
            Ok(())
        },
    );

    graph.export_buffer(
        compute.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let capabilities = common::single_queue_capabilities();
    let compiled = graph.compile(&capabilities).unwrap().graph;
    let device = DeviceIdentity::new(2);
    let mut registry = TestRegistry::new(device);
    let input_id = BufferBindingId::new(1);
    registry.register_buffer(
        input_id,
        TestBuffer::new(1),
        common::buffer(1024),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry.register_compute_pipeline(ComputePipelineId::new(1), TestComputePipeline::new(1));
    registry.register_bindings(BindingSetId::new(1), TestBindings::new(1));
    let executor = FrameExecutor::new(TestRhi::new(capabilities, device));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(input.slot, input_id);
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
        "TestRhi executed compute: exports={}, trace_events={event_count}",
        frame.exports.buffers().count(),
    );
}
