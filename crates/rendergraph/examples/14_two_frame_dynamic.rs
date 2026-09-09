//! Compile-only fixture for retained graphs with owned per-frame inputs.
//! Run with `cargo run --example 14_two_frame_dynamic`; instantiation does not execute work.

mod common;

use std::rc::Rc;

use fluxel_rendergraph::pass::{BindingResource, ColorAttachmentDesc, LoadOp, StoreOp};
use fluxel_rendergraph::*;

/// Dynamic values change each frame without rebuilding the graph.
struct FrameData {
    camera_uniform_offset: u32,
    visible_instances: u32,
    viewport: Viewport,
}

struct DrawData {
    camera: BufferRead,
}

fn main() {
    let mut graph = RenderGraph::<FrameData>::new();
    let camera = graph.import_buffer_slot(
        "camera-ring-buffer",
        common::imported_buffer(64 * 1024, ResourceAccessState::UniformRead),
    );
    let camera_slot = camera.slot;
    let color = graph.create_texture("color", common::rgba8_texture(1280, 720));

    let draw = graph.add_raster_pass(
        "dynamic-draw",
        |pass| {
            let camera = pass.read_buffer(
                &camera.version,
                BufferReadUse::Uniform,
                BufferRange::whole(),
            );
            let color = pass.color_attachment(
                color,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.01, 0.01, 0.02, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Unknown,
                    },
                },
            );
            (color, DrawData { camera })
        },
        |commands, resolver, data, frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(14),
                &[BindingResource::BufferRead(&data.camera)],
                &[frame.camera_uniform_offset],
            )?;
            commands.set_pipeline(RasterPipelineId::new(14))?;
            commands.set_bindings(&bindings)?;
            commands.set_viewport(frame.viewport)?;
            commands.draw(0..3, 0..frame.visible_instances)?;
            Ok(())
        },
    );
    graph.export_texture(
        draw.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );

    // One compiled plan can create independent owner-thread and Send-mode frame runs.
    if let Ok(output) = graph.compile(&common::single_queue_capabilities()) {
        let mut frame_zero = FrameInputs::new(FrameData {
            camera_uniform_offset: 0,
            visible_instances: 300,
            viewport: frame_viewport(1280.0, 720.0),
        });
        let mut frame_one = FrameInputs::new(FrameData {
            camera_uniform_offset: 256,
            visible_instances: 412,
            viewport: frame_viewport(960.0, 540.0),
        });

        frame_zero.bind_buffer(camera_slot, BufferBindingId::new(100));
        frame_one.bind_buffer(camera_slot, BufferBindingId::new(101));
        let mut send_frame = FrameInputs::new(FrameData {
            camera_uniform_offset: 512,
            visible_instances: 270,
            viewport: frame_viewport(1280.0, 720.0),
        });
        send_frame.bind_buffer(camera_slot, BufferBindingId::new(102));
        let _local_zero = output.graph.instantiate_local(frame_zero);
        let _local_one = output.graph.instantiate_local(frame_one);
        let _send = output.graph.instantiate_send(send_frame);
    }

    local_non_send_frame_data();
}

fn frame_viewport(width: f32, height: f32) -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width,
        height,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

struct LocalFrameData {
    _ui_label: Rc<String>,
}

fn local_non_send_frame_data() {
    let graph = RenderGraph::<LocalFrameData>::new();
    if let Ok(output) = graph.compile(&common::single_queue_capabilities()) {
        let inputs = FrameInputs::new(LocalFrameData {
            _ui_label: Rc::new("owner-thread UI".into()),
        });
        let _local = output.graph.instantiate_local(inputs);

        // `instantiate_send(inputs)` intentionally would not type-check because
        // Rc-backed frame data is owner-thread-only.
    }
}
