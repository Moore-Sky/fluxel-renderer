//! Compile-only fixture for composing helpers around one texture version.
//! Run with `cargo run --example 13_helper_composition`.
//!
//! The move model deliberately makes the ordering cost visible: all helpers
//! that read X1 must receive `&X1` before `ModifyHelper` consumes X1 into X2.
//! This fixture is the acceptance test for deciding whether that authoring rule
//! remains natural once real renderer helpers are written.

mod common;

use fluxel_rendergraph::*;

struct ReadData {
    source: TextureRead,
}

struct ModifyData {
    target: TextureWrite,
}

fn shadow_helper(graph: &mut RenderGraph, input: &TextureVersion) -> PassId {
    let pass = graph.add_compute_pass(
        "shadow-helper",
        |builder| {
            let source =
                builder.read_texture(input, TextureReadUse::Sampled, TextureRange::whole());
            ((), ReadData { source })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(40))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(40),
                &[BindingResource::TextureRead(&data.source)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );
    pass.id
}

fn ssao_helper(graph: &mut RenderGraph, input: &TextureVersion) -> PassId {
    let pass = graph.add_compute_pass(
        "ssao-helper",
        |builder| {
            let source =
                builder.read_texture(input, TextureReadUse::Sampled, TextureRange::whole());
            ((), ReadData { source })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(41))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(41),
                &[BindingResource::TextureRead(&data.source)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );
    pass.id
}

fn modify_helper(graph: &mut RenderGraph, input: TextureVersion) -> TextureVersion {
    graph
        .add_compute_pass(
            "modify-helper",
            |builder| {
                let (output, target) = builder.write_texture(
                    input,
                    TextureWriteUse::Storage,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                (output, ModifyData { target })
            },
            |commands, resolver, data, _frame| {
                commands.set_pipeline(ComputePipelineId::new(42))?;
                let bindings = resolver.resolve_bindings(
                    BindingSetId::new(42),
                    &[BindingResource::TextureWrite(&data.target)],
                    &[],
                )?;
                commands.set_bindings(&bindings)?;
                commands.dispatch([80, 45, 1])
            },
        )
        .output
}

fn main() {
    let mut graph = RenderGraph::new();
    let x0 = graph.create_texture("scene", common::rgba8_texture(1280, 720));
    let x1 = modify_helper(&mut graph, x0);

    // Calling either read helper after this next line would be rejected by Rust:
    // `x1` is consumed to create X2. The constraint is intentional for now and
    // must be judged with real helpers before freezing the public API.
    let shadow = shadow_helper(&mut graph, &x1);
    let ssao = ssao_helper(&mut graph, &x1);
    graph.mark_side_effect(
        shadow,
        SideEffectReason::Diagnostic("helper fixture".into()),
    );
    graph.mark_side_effect(ssao, SideEffectReason::Diagnostic("helper fixture".into()));
    let x2 = modify_helper(&mut graph, x1);

    graph.export_texture(
        x2,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
