//! Compile-only example; run with `cargo run --example 04_frame_pipeline`.
//! Execute callbacks are retained but are not invoked by `compile`.

mod common;

use fluxel_rendergraph::pass::{
    BindingResource, ColorAttachmentDesc, DepthStencilAttachmentDesc, LoadOp, StoreOp,
};
use fluxel_rendergraph::*;

struct OpaqueData {
    albedo: TextureRead,
}

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
    let albedo = graph.import_texture_slot(
        "albedo",
        common::imported_texture(1280, 720, ResourceAccessState::ShaderSampledRead),
    );
    let surface = graph.import_surface_texture_slot("surface", common::imported_surface(1280, 720));
    let scene = graph.create_texture("scene", common::rgba8_texture(1280, 720));
    let depth = graph.create_texture("depth", common::depth_texture(1280, 720));
    let post = graph.create_texture("post", common::rgba8_texture(1280, 720));

    let opaque = graph.add_raster_pass(
        "opaque",
        |pass| {
            let albedo = pass.read_texture(
                &albedo.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let scene = pass.color_attachment(
                scene,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            let _depth = pass.depth_stencil_attachment(
                depth,
                DepthStencilAttachmentDesc {
                    range: TextureRange::whole(),
                    depth: Some(AttachmentOps {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Discard,
                        write_coverage: WriteCoverage::Unknown,
                    }),
                    stencil: None,
                },
            );
            (scene, OpaqueData { albedo })
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(1),
                &[BindingResource::TextureRead(&data.albedo)],
                &[],
            )?;
            commands.set_pipeline(RasterPipelineId::new(1))?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
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
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(2),
                &[
                    BindingResource::TextureRead(&data.scene),
                    BindingResource::TextureWrite(&data.post),
                ],
                &[],
            )?;
            commands.set_pipeline(ComputePipelineId::new(2))?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([80, 45, 1])?;
            Ok(())
        },
    );

    let copy = graph.add_copy_pass(
        "copy-to-surface",
        |pass| {
            let source = pass.read_texture(&post.output, TextureRange::whole());
            let (surface, destination) =
                pass.write_texture(surface.version, TextureRange::whole(), WriteCoverage::Full);
            (
                surface,
                CopyData {
                    source,
                    destination,
                },
            )
        },
        |commands, _resolver, data, _frame| {
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
            )?;
            Ok(())
        },
    );

    graph.present(copy.output, PresentContract::new());
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
