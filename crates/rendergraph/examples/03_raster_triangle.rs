//! CPU-only execution example; run with `cargo run --example 03_raster_triangle`.

mod common;

use fluxel_rendergraph::pass::{ColorAttachmentDesc, LoadOp, StoreOp};
use fluxel_rendergraph::test_rhi::{TestBuffer, TestRasterPipeline, TestRegistry, TestRhi};
use fluxel_rendergraph::*;

struct TriangleData {
    vertices: BufferRead,
}

fn main() {
    let mut graph = RenderGraph::new();
    let vertices = graph.import_buffer_slot(
        "triangle-vertices",
        common::imported_buffer(3 * 8 * 4, ResourceAccessState::VertexRead),
    );
    let color = graph.create_texture("triangle-color", common::rgba8_texture(1280, 720));

    let raster = graph.add_raster_pass(
        "triangle",
        |pass| {
            let vertices = pass.read_buffer(
                &vertices.version,
                BufferReadUse::Vertex,
                BufferRange::whole(),
            );
            let color = pass.color_attachment(
                color,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.02, 0.02, 0.04, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            (color, TriangleData { vertices })
        },
        |commands, _resolver, data, _frame| {
            commands.set_pipeline(RasterPipelineId::new(1))?;
            commands.set_vertex_buffer(0, &data.vertices)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );

    graph.export_texture(
        raster.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let capabilities = common::single_queue_capabilities();
    let compiled = graph.compile(&capabilities).unwrap().graph;
    let device = DeviceIdentity::new(3);
    let mut registry = TestRegistry::new(device);
    let vertices_id = BufferBindingId::new(1);
    registry.register_buffer(
        vertices_id,
        TestBuffer::new(1),
        common::buffer(3 * 8 * 4),
        BufferUsage::from_kinds([BufferUsageKind::Vertex]),
        ResourceAccessState::VertexRead,
    );
    registry.register_raster_pipeline(RasterPipelineId::new(1), TestRasterPipeline::new(1));
    let executor = FrameExecutor::new(TestRhi::new(capabilities, device));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(vertices.slot, vertices_id);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    let event_count = executor.try_backend().unwrap().trace().len();
    println!(
        "TestRhi executed raster: exports={}, trace_events={event_count}",
        frame.exports.textures().count(),
    );
}
