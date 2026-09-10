//! Compile-only example; run with `cargo run --example 05_import_export_history`.
//! Execute callbacks are retained but are not invoked by `compile`.
//! This is a generic persistent storage read-modify-write fixture, not a claim
//! that a complete TAA algorithm may update one history texture in place.

mod common;

use fluxel_rendergraph::*;

struct HistoryData {
    history: TextureReadWrite,
}

fn main() {
    let mut graph = RenderGraph::new();
    let history = graph.import_texture_slot(
        "persistent-storage-state",
        common::imported_texture(1920, 1080, ResourceAccessState::ShaderSampledRead),
    );

    let update = graph.add_compute_pass(
        "update-history",
        |pass| {
            let (history, history_access) = pass.read_write_texture(
                history.version,
                TextureReadWriteUse::Storage,
                TextureRange::whole(),
            );
            (
                history,
                HistoryData {
                    history: history_access,
                },
            )
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(3),
                &[BindingResource::TextureReadWrite(&data.history)],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(3))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([120, 68, 1])?;
            Ok(())
        },
    );

    graph.export_texture(
        update.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
