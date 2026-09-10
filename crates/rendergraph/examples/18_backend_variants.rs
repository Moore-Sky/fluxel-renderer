//! Compile-only fixture for explicit renderer-selected algorithm variants.
//! Run with `cargo run --example 18_backend_variants`.
//! Native uses a compute semantic; WebGL2 selects a raster implementation of the
//! same effect rather than pretending a compute pass can be lowered there.

mod common;

use fluxel_rendergraph::*;

struct ComputeData {
    source: TextureRead,
    destination: TextureWrite,
}

struct RasterData {
    source: TextureRead,
}

struct PresentData {
    source: TextureRead,
}

fn native_compute_variant() -> CompileResult {
    let mut graph = RenderGraph::new();
    let source = graph.import_texture_slot(
        "source",
        common::imported_texture(1280, 720, ResourceAccessState::ShaderSampledRead),
    );
    let output = graph.create_texture("effect-output", common::rgba8_texture(1280, 720));
    let effect = graph.add_compute_pass(
        "native-compute-effect",
        |pass| {
            let source = pass.read_texture(
                &source.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let (output, destination) = pass.write_texture(
                output,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (
                output,
                ComputeData {
                    source,
                    destination,
                },
            )
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(18),
                &[
                    BindingResource::TextureRead(&data.source),
                    BindingResource::TextureWrite(&data.destination),
                ],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(18))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])?;
            Ok(())
        },
    );
    graph.export_texture(
        effect.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    graph.compile(&common::single_queue_capabilities())
}

fn webgl2_raster_variant() -> CompileResult {
    let mut graph = RenderGraph::new();
    let source = graph.import_texture_slot(
        "source",
        common::imported_texture(1280, 720, ResourceAccessState::ShaderSampledRead),
    );
    let output = graph.create_texture("effect-output", common::rgba8_texture(1280, 720));
    let surface = graph.import_surface_texture_slot("surface", common::imported_surface(1280, 720));
    let effect = graph.add_raster_pass(
        "webgl2-fullscreen-effect",
        |pass| {
            let source = pass.read_texture(
                &source.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
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
            (output, RasterData { source })
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(180),
                &[BindingResource::TextureRead(&data.source)],
                &[],
            )?;
            commands.set_pipeline(RasterPipelineId::new(180))?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );
    let composite = graph.add_raster_pass(
        "webgl2-surface-composite",
        |pass| {
            let source = pass.read_texture(
                &effect.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let surface = pass.color_attachment(
                surface.version,
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
            (surface, PresentData { source })
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(181),
                &[BindingResource::TextureRead(&data.source)],
                &[],
            )?;
            commands.set_pipeline(RasterPipelineId::new(181))?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );
    graph.present(composite.output, PresentContract::new());
    graph.compile(&common::webgl2_capabilities())
}

fn main() {
    let _native = native_compute_variant();
    let _webgl2 = webgl2_raster_variant();
}
