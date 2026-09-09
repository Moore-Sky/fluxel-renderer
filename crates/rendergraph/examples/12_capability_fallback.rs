//! Compile-only fixture for explicit native-compute and WebGL2 raster variants.
//! Run with `cargo run --example 12_capability_fallback`.
//!
//! A compute algorithm is not automatically portable to WebGL2. The renderer
//! selects a semantically equivalent raster variant before compiling the graph;
//! capability validation then reports whether each selected variant is legal.

mod common;

use fluxel_rendergraph::pass::{BindingResource, ColorAttachmentDesc, LoadOp, StoreOp};
use fluxel_rendergraph::*;

struct ComputeData {
    output: TextureWrite,
}

fn native_compute_variant() -> CompileResult {
    let mut graph = RenderGraph::new();
    let output = graph.create_texture("effect", common::rgba8_texture(1280, 720));
    let effect = graph.add_compute_pass(
        "native-compute-effect",
        |pass| {
            let (output, access) = pass.write_texture(
                output,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, ComputeData { output: access })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(30))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(30),
                &[BindingResource::TextureWrite(&data.output)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );
    graph.export_texture(
        effect.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    graph.compile(&common::single_queue_capabilities())
}

fn webgl2_raster_variant() -> CompileResult {
    let mut graph = RenderGraph::new();
    let output = graph.create_texture("effect", common::rgba8_texture(1280, 720));
    let effect = graph.add_raster_pass(
        "webgl2-raster-effect",
        |pass| {
            let output = pass.color_attachment(
                output,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (output, ())
        },
        |commands, _resolver, _data, _frame| {
            commands.set_pipeline(RasterPipelineId::new(31))?;
            commands.draw(0..3, 0..1)
        },
    );
    graph.export_texture(
        effect.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ColorAttachmentWrite,
        },
    );
    graph.compile(&common::webgl2_capabilities())
}

fn main() {
    let _native: CompileResult = native_compute_variant();
    let _webgl2: CompileResult = webgl2_raster_variant();
    let _unsupported_storage: CompileResult = unsupported_webgl_storage_buffer();
}

fn unsupported_webgl_storage_buffer() -> CompileResult {
    let mut graph = RenderGraph::new();
    let buffer = graph.import_buffer_slot(
        "storage-buffer",
        common::imported_buffer(1024, ResourceAccessState::ShaderStorageRead),
    );
    let pass = graph.add_raster_pass(
        "storage-buffer-on-webgl2",
        |builder| {
            let read = builder.read_buffer(
                &buffer.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), read)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain unsupported buffer fixture".into()),
    );
    graph.compile(&common::webgl2_capabilities())
}
