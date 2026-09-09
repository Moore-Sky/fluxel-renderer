//! Compile-only fixture for MRT and fan-out readers of one attachment version.
//! Run with `cargo run --example 17_mrt_multi_reader`.

mod common;

use fluxel_rendergraph::pass::{BindingResource, ColorAttachmentDesc, LoadOp, StoreOp};
use fluxel_rendergraph::*;

struct GBuffer {
    albedo: TextureVersion,
    normal: TextureVersion,
}

struct LightingData {
    albedo: TextureRead,
    normal: TextureRead,
}

struct DebugData {
    albedo: TextureRead,
}

fn main() {
    let mut graph = RenderGraph::new();
    let albedo = graph.create_texture("gbuffer-albedo", common::rgba8_texture(1280, 720));
    let normal = graph.create_texture("gbuffer-normal", common::rgba8_texture(1280, 720));
    let lit = graph.create_texture("lit", common::rgba8_texture(1280, 720));

    let gbuffer = graph.add_raster_pass(
        "gbuffer",
        |pass| {
            let albedo = pass.color_attachment(
                albedo,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            let normal = pass.color_attachment(
                normal,
                ColorAttachmentDesc {
                    index: 1,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            (GBuffer { albedo, normal }, ())
        },
        |commands, _resolver, _data, _frame| {
            commands.set_pipeline(RasterPipelineId::new(17))?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );

    let lighting = graph.add_compute_pass(
        "lighting",
        |pass| {
            let albedo = pass.read_texture(
                &gbuffer.output.albedo,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let normal = pass.read_texture(
                &gbuffer.output.normal,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let (lit, write) = pass.write_texture(
                lit,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (lit, (LightingData { albedo, normal }, write))
        },
        |commands, resolver, data, _frame| {
            let (reads, write) = data;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(17),
                &[
                    BindingResource::TextureRead(&reads.albedo),
                    BindingResource::TextureRead(&reads.normal),
                    BindingResource::TextureWrite(write),
                ],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(17))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])?;
            Ok(())
        },
    );

    let debug = graph.add_raster_pass(
        "albedo-debug-reader",
        |pass| {
            let albedo = pass.read_texture(
                &gbuffer.output.albedo,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), DebugData { albedo })
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(170),
                &[BindingResource::TextureRead(&data.albedo)],
                &[],
            )?;
            commands.set_pipeline(RasterPipelineId::new(170))?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );
    graph.mark_side_effect(
        debug.id,
        SideEffectReason::Diagnostic("retain MRT fan-out reader".into()),
    );
    graph.export_texture(
        lighting.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
