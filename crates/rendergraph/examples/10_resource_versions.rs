//! Compile-only fixture for a linear logical buffer-version chain; run with `cargo run --example 10_resource_versions`.

mod common;

use fluxel_rendergraph::*;

struct WriteData {
    output: BufferWrite,
}

struct ReadWriteData {
    value: BufferReadWrite,
}

struct ReadData {
    value: BufferRead,
}

fn main() {
    let mut graph = RenderGraph::new();
    let x0 = graph.create_buffer("x", common::buffer(1024));

    let x1 = graph.add_compute_pass(
        "initialize-x1",
        |pass| {
            let (output, access) = pass.write_buffer(
                x0,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, WriteData { output: access })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(10))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(10),
                &[BindingResource::BufferWrite(&data.output)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([16, 1, 1])
        },
    );

    // The old-version reader must be declared before X1 is consumed into X2.
    // The derived dependency keeps this reader before a future physical overwrite.
    let x1_observer = graph.add_compute_pass(
        "observe-x1-before-overwrite",
        |pass| {
            let value = pass.read_buffer(&x1.output, BufferReadUse::Storage, BufferRange::whole());
            ((), ReadData { value })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(101))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(101),
                &[BindingResource::BufferRead(&data.value)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    graph.mark_side_effect(
        x1_observer.id,
        SideEffectReason::Diagnostic("old-version WAR fixture".into()),
    );

    let x2 = graph.add_compute_pass(
        "modify-x2",
        |pass| {
            let (output, access) = pass.read_write_buffer(
                x1.output,
                BufferReadWriteUse::Storage,
                BufferRange::whole(),
            );
            (output, ReadWriteData { value: access })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(11))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(11),
                &[BindingResource::BufferReadWrite(&data.value)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([16, 1, 1])
        },
    );

    let inspect = graph.add_compute_pass(
        "inspect-x2",
        |pass| {
            let value = pass.read_buffer(&x2.output, BufferReadUse::Storage, BufferRange::whole());
            ((), ReadData { value })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(12))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(12),
                &[BindingResource::BufferRead(&data.value)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    graph.mark_side_effect(
        inspect.id,
        SideEffectReason::Diagnostic("version fixture".into()),
    );

    graph.export_buffer(
        x2.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
