//! Compile-only fixture for renderer-owned binding recipes and indexed raster work.
//! Run with `cargo run --example 15_sampled_indexed_draw`; compile does not record commands.
//! The sampler and binding layout live behind `BindingSetId`; only graph resources
//! and dynamic offsets are validated by the pass-local resolver.

mod common;

use fluxel_rendergraph::*;

struct MeshData {
    albedo: TextureRead,
    camera: BufferRead,
    vertices: BufferRead,
    indices: BufferRead,
}

struct FrameData {
    camera_offset: u32,
    instance_count: u32,
}

fn main() {
    let mut graph = RenderGraph::<FrameData>::new();
    let albedo = graph.import_texture_slot(
        "albedo",
        common::imported_texture(1024, 1024, ResourceAccessState::ShaderSampledRead),
    );
    let camera = graph.import_buffer_slot(
        "camera-ring",
        common::imported_buffer(4096, ResourceAccessState::UniformRead),
    );
    let vertices = graph.import_buffer_slot(
        "mesh-vertices",
        common::imported_buffer(4096, ResourceAccessState::VertexRead),
    );
    let indices = graph.import_buffer_slot(
        "mesh-indices",
        common::imported_buffer(2048, ResourceAccessState::IndexRead),
    );
    let target = graph.create_texture("target", common::rgba8_texture(1280, 720));

    let mesh = graph.add_raster_pass(
        "sampled-indexed-mesh",
        |pass| {
            let albedo = pass.read_texture(
                &albedo.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            let camera = pass.read_buffer(
                &camera.version,
                BufferReadUse::Uniform,
                BufferRange::whole(),
            );
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let indices =
                pass.read_buffer(&indices.version, BufferReadUse::Index, BufferRange::whole());
            let target = pass.color_attachment(
                target,
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
            (
                target,
                MeshData {
                    albedo,
                    camera,
                    vertices,
                    indices,
                },
            )
        },
        |commands, resolver, data, frame| {
            // Recipe 15 owns its sampler, group/set layout, and pipeline compatibility metadata.
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(15),
                &[
                    BindingResource::TextureRead(&data.albedo),
                    BindingResource::BufferRead(&data.camera),
                ],
                &[frame.camera_offset],
            )?;
            commands.set_pipeline(RasterPipelineId::new(15))?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.vertices)?;
            commands.set_index_buffer(&data.indices, IndexFormat::Uint32)?;
            commands.draw_indexed(0..36, 0, 0..frame.instance_count)?;
            Ok(())
        },
    );
    graph.export_texture(
        mesh.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let _result: CompileResult<FrameData> = graph.compile(&common::single_queue_capabilities());
}
