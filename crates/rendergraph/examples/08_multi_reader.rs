//! Compile-only fixture for fan-out from one logical resource version.
//! Run with `cargo run --example 08_multi_reader`; compile never calls execute callbacks.

mod common;

use fluxel_rendergraph::pass::BindingResource;
use fluxel_rendergraph::*;

struct ProduceData {
    output: TextureWrite,
}

struct ReadData {
    input: TextureRead,
}

struct ModifyData {
    input: TextureWrite,
}

fn main() {
    let mut graph = RenderGraph::new();
    let texture = graph.create_texture("shared", common::rgba8_texture(1280, 720));

    let produced = graph.add_compute_pass(
        "produce",
        |pass| {
            let (output, access) = pass.write_texture(
                texture,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ProduceData { output: access })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(1))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(1),
                &[BindingResource::TextureWrite(&data.output)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );

    // Both readers borrow the same X1. Their declaration order does not create
    // a writer branch, and either may later be recorded independently.
    let shadow = graph.add_compute_pass(
        "shadow-reader",
        |pass| {
            let input = pass.read_texture(
                &produced.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), ReadData { input })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(2))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(2),
                &[BindingResource::TextureRead(&data.input)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );
    graph.mark_side_effect(
        shadow.id,
        SideEffectReason::Diagnostic("fan-out fixture".into()),
    );

    let ssao = graph.add_compute_pass(
        "ssao-reader",
        |pass| {
            let input = pass.read_texture(
                &produced.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), ReadData { input })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(3))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(3),
                &[BindingResource::TextureRead(&data.input)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );
    graph.mark_side_effect(
        ssao.id,
        SideEffectReason::Diagnostic("fan-out fixture".into()),
    );

    // Consuming X1 only happens after every intended reader has been declared.
    let modified = graph.add_compute_pass(
        "modify",
        |pass| {
            let (output, input) = pass.write_texture(
                produced.output,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ModifyData { input })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(4))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(4),
                &[BindingResource::TextureWrite(&data.input)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );

    graph.export_texture(
        modified.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
