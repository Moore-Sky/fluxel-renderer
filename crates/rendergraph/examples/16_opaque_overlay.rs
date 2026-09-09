//! Compile-only fixture showing that attachment content operations are setup facts.
//! Run with `cargo run --example 16_opaque_overlay`.
//! Opaque initializes the target; overlay explicitly loads that produced content.

mod common;

use fluxel_rendergraph::pass::{ColorAttachmentDesc, LoadOp, StoreOp};
use fluxel_rendergraph::*;

fn main() {
    let mut graph = RenderGraph::new();
    let scene = graph.create_texture("scene", common::rgba8_texture(1280, 720));

    let opaque = graph.add_raster_pass(
        "opaque",
        |pass| {
            let scene = pass.color_attachment(
                scene,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.02, 0.03, 0.05, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            (scene, ())
        },
        |commands, _resolver, _data, _frame| {
            commands.set_pipeline(RasterPipelineId::new(16))?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );

    let overlay = graph.add_raster_pass(
        "overlay",
        |pass| {
            let scene = pass.color_attachment(
                opaque.output,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            (scene, ())
        },
        |commands, _resolver, _data, _frame| {
            commands.set_pipeline(RasterPipelineId::new(160))?;
            commands.draw(0..6, 0..1)?;
            Ok(())
        },
    );
    graph.export_texture(
        overlay.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
