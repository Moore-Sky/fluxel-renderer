//! Shared closed legacy-unlit graph declaration for native and browser paths.

use fluxel_rendergraph::{
    AttachmentOps, BindingResource, BufferRange, BufferRead, BufferReadUse, BufferVersion,
    ColorAttachmentDesc, DeclaredPass, IndexFormat, LoadOp, RenderGraph, StoreOp, TextureRange,
    TextureVersion, Viewport, WriteCoverage,
};

use super::{camera_bindings, camera_pipeline};

/// One already-declared set of portable inputs for a fixed unlit draw.
pub(crate) struct FixedGraphDraw<'a> {
    pub(crate) positions: &'a BufferVersion,
    pub(crate) indices: &'a BufferVersion,
    pub(crate) uniform: &'a BufferVersion,
    pub(crate) index_count: u32,
}

struct DeclaredDraw {
    position: BufferRead,
    index: BufferRead,
    uniform: BufferRead,
    index_count: u32,
}

/// Declares the single clear and insertion-ordered indexed draws used by both targets.
pub(crate) fn declare_fixed_unlit_pass(
    graph: &mut RenderGraph,
    target: TextureVersion,
    draws: &[FixedGraphDraw<'_>],
    extent: [u32; 2],
) -> DeclaredPass<TextureVersion> {
    graph.add_raster_pass(
        "packet-legacy-unlit-indexed",
        |pass| {
            let output = pass.color_attachment(
                target,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let declared = draws
                .iter()
                .map(|draw| DeclaredDraw {
                    position: pass.read_buffer(
                        draw.positions,
                        BufferReadUse::Vertex,
                        BufferRange::Whole,
                    ),
                    index: pass.read_buffer(draw.indices, BufferReadUse::Index, BufferRange::Whole),
                    uniform: pass.read_buffer(
                        draw.uniform,
                        BufferReadUse::Uniform,
                        BufferRange::Whole,
                    ),
                    index_count: draw.index_count,
                })
                .collect::<Vec<_>>();
            (output, declared)
        },
        move |commands, resolver, draws, _| {
            commands.set_pipeline(camera_pipeline())?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            for draw in draws {
                let bindings = resolver.resolve_bindings(
                    camera_bindings(),
                    &[BindingResource::BufferRead(&draw.uniform)],
                    &[],
                )?;
                commands.set_bindings(&bindings)?;
                commands.set_vertex_buffer(0, &draw.position)?;
                commands.set_index_buffer(&draw.index, IndexFormat::Uint32)?;
                commands.draw_indexed(0..draw.index_count, 0, 0..1)?;
            }
            Ok(())
        },
    )
}
