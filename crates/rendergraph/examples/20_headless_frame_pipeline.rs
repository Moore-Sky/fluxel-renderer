//! A complete CPU-only frame pipeline executed through `TestRhi`.
//!
//! Run with `cargo run --example 20_headless_frame_pipeline`.
//! Unlike the surface-oriented frame-pipeline example, this ends by exporting
//! an ordinary texture, so it is executable without a window or GPU.

mod common;

use fluxel_rendergraph::{
    test_rhi::{TestBindings, TestComputePipeline, TestRasterPipeline, TestRegistry, TestRhi},
    *,
};

struct PostData {
    scene: TextureRead,
    post: TextureWrite,
}

struct CopyData {
    source: TextureRead,
    destination: TextureWrite,
}

fn main() {
    let mut graph = RenderGraph::new();
    let scene = graph.create_texture("scene", common::rgba8_texture(1280, 720));
    let post = graph.create_texture("post", common::rgba8_texture(1280, 720));
    let output = graph.create_texture("output", common::rgba8_texture(1280, 720));

    let opaque = graph.add_raster_pass(
        "opaque",
        |pass| {
            let scene = pass.color_attachment(
                scene,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.02, 0.02, 0.04, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (scene, ())
        },
        |commands, _, _, _| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            commands.draw(0..3, 0..1)
        },
    );

    let post = graph.add_compute_pass(
        "post",
        |pass| {
            let scene = pass.read_texture(
                &opaque.output,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let (post, post_access) = pass.write_texture(
                post,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (
                post,
                PostData {
                    scene,
                    post: post_access,
                },
            )
        },
        |commands, resolver, data, _| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(1),
                &[
                    BindingResource::TextureRead(&data.scene),
                    BindingResource::TextureWrite(&data.post),
                ],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(1))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])
        },
    );

    let copy = graph.add_copy_pass(
        "copy-output",
        |pass| {
            let source = pass.read_texture(&post.output, TextureRange::whole());
            let (output, destination) =
                pass.write_texture(output, TextureRange::whole(), WriteCoverage::Full);
            (
                output,
                CopyData {
                    source,
                    destination,
                },
            )
        },
        |commands, _, data, _| {
            commands.copy_texture(
                &data.source,
                &data.destination,
                TextureCopyRegion {
                    source_origin: [0, 0, 0],
                    destination_origin: [0, 0, 0],
                    extent: [1280, 720, 1],
                    source_mip_level: 0,
                    destination_mip_level: 0,
                },
            )
        },
    );

    let export = graph.export_texture(
        copy.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let capabilities = common::single_queue_capabilities();
    let compiled = graph.compile(&capabilities).unwrap().graph;
    let device = DeviceIdentity::new(20);
    let mut registry = TestRegistry::new(device);
    registry.register_raster_pipeline(RasterPipelineId::new(1), TestRasterPipeline::new(1));
    registry.register_compute_pipeline(ComputePipelineId::new(1), TestComputePipeline::new(1));
    registry.register_bindings(BindingSetId::new(1), TestBindings::new(1));
    let executor = FrameExecutor::new(TestRhi::new(capabilities, device));
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    let event_count = executor.try_backend().unwrap().trace().len();
    println!(
        "TestRhi executed headless frame pipeline: exported={}, trace_events={event_count}",
        frame.exports.texture(export).is_some(),
    );
}
