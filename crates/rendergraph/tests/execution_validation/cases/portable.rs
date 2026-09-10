//! Portable command and copy validation cases.

use super::*;

#[test]
fn begin_raster_preserves_complete_color_and_depth_attachment_operations() {
    let mut graph = RenderGraph::new();
    let color = graph.create_texture("color", texture(TextureFormat::Rgba8Unorm));
    let depth = graph.create_texture("depth", texture(TextureFormat::Depth32Float));
    let pass = graph.add_raster_pass(
        "attachments",
        |pass| {
            let color = pass.color_attachment(
                color,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.25, 0.5, 0.75, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            let depth = pass.depth_stencil_attachment(
                depth,
                DepthStencilAttachmentDesc {
                    range: TextureRange::whole(),
                    depth: Some(AttachmentOps {
                        load: LoadOp::Clear(0.5),
                        store: StoreOp::Discard,
                        write_coverage: WriteCoverage::Full,
                    }),
                    stencil: None,
                },
            );
            ((color, depth), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        pass.output.0,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    let Some(TestTraceEvent::BeginRaster {
        colors,
        depth_stencil,
        ..
    }) = trace
        .iter()
        .find(|event| matches!(event, TestTraceEvent::BeginRaster { .. }))
    else {
        panic!("missing raster begin")
    };
    assert_eq!(colors.len(), 1);
    assert_eq!(
        colors[0].operations.load,
        LoadOp::Clear([0.25, 0.5, 0.75, 1.0])
    );
    assert_eq!(colors[0].operations.store, StoreOp::Store);
    let depth = depth_stencil.as_ref().expect("depth attachment");
    assert_eq!(depth.depth.expect("depth ops").load, LoadOp::Clear(0.5));
    assert_eq!(depth.depth.expect("depth ops").store, StoreOp::Discard);
}

#[test]
fn raster_commands_reject_invalid_portable_arguments_before_draw_or_submit() {
    enum InvalidCommand {
        Viewport(Viewport),
        Scissor(ScissorRect),
        Draw {
            vertices: std::ops::Range<u32>,
            instances: std::ops::Range<u32>,
        },
        DrawIndexed {
            indices: std::ops::Range<u32>,
            instances: std::ops::Range<u32>,
        },
    }

    let valid_viewport = Viewport {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let cases = [
        InvalidCommand::Viewport(Viewport {
            x: f32::NAN,
            ..valid_viewport
        }),
        InvalidCommand::Viewport(Viewport {
            width: 0.0,
            ..valid_viewport
        }),
        InvalidCommand::Viewport(Viewport {
            height: -1.0,
            ..valid_viewport
        }),
        InvalidCommand::Viewport(Viewport {
            min_depth: -0.1,
            ..valid_viewport
        }),
        InvalidCommand::Viewport(Viewport {
            max_depth: 1.1,
            ..valid_viewport
        }),
        InvalidCommand::Viewport(Viewport {
            min_depth: 0.75,
            max_depth: 0.25,
            ..valid_viewport
        }),
        InvalidCommand::Scissor(ScissorRect {
            x: u32::MAX,
            y: 0,
            width: 1,
            height: 1,
        }),
        InvalidCommand::Scissor(ScissorRect {
            x: 0,
            y: u32::MAX,
            width: 1,
            height: 1,
        }),
        InvalidCommand::Draw {
            vertices: 0..0,
            instances: 0..1,
        },
        InvalidCommand::Draw {
            vertices: 0..3,
            instances: 0..0,
        },
        InvalidCommand::DrawIndexed {
            indices: 0..0,
            instances: 0..1,
        },
        InvalidCommand::DrawIndexed {
            indices: 0..3,
            instances: 0..0,
        },
    ];

    for command in cases {
        let mut graph = RenderGraph::new();
        let color = graph.create_texture("color", texture(TextureFormat::Rgba8Unorm));
        let pass = graph.add_raster_pass(
            "invalid raster command",
            |pass| {
                let output = pass.color_attachment(
                    color,
                    ColorAttachmentDesc {
                        index: 0,
                        range: TextureRange::whole(),
                        operations: AttachmentOps {
                            load: LoadOp::Clear([0.0; 4]),
                            store: StoreOp::Store,
                            write_coverage: WriteCoverage::Full,
                        },
                    },
                );
                (output, ())
            },
            move |commands, _, _, _| match &command {
                InvalidCommand::Viewport(viewport) => commands.set_viewport(*viewport),
                InvalidCommand::Scissor(scissor) => commands.set_scissor(*scissor),
                InvalidCommand::Draw {
                    vertices,
                    instances,
                } => commands.draw(vertices.clone(), instances.clone()),
                InvalidCommand::DrawIndexed { indices, instances } => {
                    commands.draw_indexed(indices.clone(), 0, instances.clone())
                }
            },
        );
        graph.export_texture(
            pass.output,
            ExportTextureContract {
                final_state: ResourceAccessState::ShaderSampledRead,
            },
        );
        let compiled = graph.compile(&caps()).unwrap().graph;
        let registry = TestRegistry::new(device());
        let executor = executor();
        assert!(matches!(
            executor.execute(
                &compiled,
                compiled.instantiate_local(FrameInputs::new(())),
                &registry,
                &registry,
            ),
            Err(ExecutionError::Recording(error))
                if error.kind == RecordingErrorKind::InvalidCommandArgument
        ));
        let trace = executor.try_backend().unwrap().trace().to_vec();
        assert!(!trace.iter().any(|event| matches!(
            event,
            TestTraceEvent::SetViewport { .. }
                | TestTraceEvent::SetScissor { .. }
                | TestTraceEvent::Draw { .. }
                | TestTraceEvent::DrawIndexed { .. }
                | TestTraceEvent::Submit { .. }
        )));
    }
}

#[test]
fn whole_buffer_copy_cannot_exceed_the_physical_descriptor() {
    let mut graph = RenderGraph::new();
    let source = graph.create_buffer("source", buffer(64));
    let destination = graph.create_buffer("destination", buffer(64));
    let initialized = graph.add_compute_pass(
        "init",
        |pass| {
            let (source, _) = pass.write_buffer(
                source,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            let (destination, _) = pass.write_buffer(
                destination,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            ((source, destination), ())
        },
        |_, _, _, _| Ok(()),
    );
    let copy = graph.add_copy_pass(
        "copy",
        |pass| {
            let source = pass.read_buffer(&initialized.output.0, BufferRange::whole());
            let (output, destination) = pass.write_buffer(
                initialized.output.1,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, (source, destination))
        },
        |commands, _, handles, _| {
            commands.copy_buffer(
                &handles.0,
                &handles.1,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 65,
                },
            )
        },
    );
    graph.export_buffer(
        copy.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    assert!(matches!(
        executor().execute(&compiled, compiled.instantiate_local(FrameInputs::new(())), &registry, &registry),
        Err(ExecutionError::Recording(error)) if error.kind == RecordingErrorKind::InvalidCommandArgument
    ));
}

#[test]
fn buffer_copy_rejects_misaligned_arguments_before_copy_or_submit() {
    for region in [
        BufferCopyRegion {
            source_offset: 2,
            destination_offset: 0,
            size: 4,
        },
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 2,
            size: 4,
        },
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: 6,
        },
    ] {
        let mut graph = RenderGraph::new();
        let source = graph.create_buffer("source", buffer(64));
        let destination = graph.create_buffer("destination", buffer(64));
        let initialized = graph.add_compute_pass(
            "init",
            |pass| {
                let (source, _) = pass.write_buffer(
                    source,
                    BufferWriteUse::Storage,
                    BufferRange::whole(),
                    WriteCoverage::Full,
                );
                let (destination, _) = pass.write_buffer(
                    destination,
                    BufferWriteUse::Storage,
                    BufferRange::whole(),
                    WriteCoverage::Full,
                );
                ((source, destination), ())
            },
            |_, _, _, _| Ok(()),
        );
        let copy = graph.add_copy_pass(
            "copy",
            |pass| {
                let source = pass.read_buffer(&initialized.output.0, BufferRange::whole());
                let (output, destination) = pass.write_buffer(
                    initialized.output.1,
                    BufferRange::whole(),
                    WriteCoverage::Full,
                );
                (output, (source, destination))
            },
            move |commands, _, handles, _| commands.copy_buffer(&handles.0, &handles.1, region),
        );
        graph.export_buffer(
            copy.output,
            ExportBufferContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph.compile(&caps()).unwrap().graph;
        let registry = TestRegistry::new(device());
        let executor = executor();
        assert!(matches!(
            executor.execute(
                &compiled, compiled.instantiate_local(FrameInputs::new(())), &registry, &registry,
            ),
            Err(ExecutionError::Recording(error)) if error.kind == RecordingErrorKind::InvalidCommandArgument
        ));
        let trace = executor.try_backend().unwrap().trace().to_vec();
        assert!(
            !trace
                .iter()
                .any(|event| matches!(event, TestTraceEvent::CopyBuffer { .. }))
        );
        assert!(
            !trace
                .iter()
                .any(|event| matches!(event, TestTraceEvent::Submit { .. }))
        );
    }
}

#[test]
fn texture_copy_rejects_out_of_bounds_and_unsupported_shapes() {
    for descriptor in [
        TextureDesc {
            dimension: TextureDimension::D3,
            ..texture(TextureFormat::Rgba8Unorm)
        },
        texture(TextureFormat::Rgba8Unorm),
    ] {
        let mut graph = RenderGraph::new();
        let source = graph.create_texture("source", descriptor);
        let destination = graph.create_texture("destination", descriptor);
        let initialized = graph.add_compute_pass(
            "init",
            |pass| {
                let (source, _) = pass.write_texture(
                    source,
                    TextureWriteUse::Storage,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                let (destination, _) = pass.write_texture(
                    destination,
                    TextureWriteUse::Storage,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                ((source, destination), ())
            },
            |_, _, _, _| Ok(()),
        );
        let region = if descriptor.dimension == TextureDimension::D3 {
            TextureCopyRegion {
                source_origin: [0; 3],
                destination_origin: [0; 3],
                extent: [1, 1, 1],
                source_mip_level: 0,
                destination_mip_level: 0,
            }
        } else {
            TextureCopyRegion {
                source_origin: [7, 0, 0],
                destination_origin: [0; 3],
                extent: [2, 1, 1],
                source_mip_level: 0,
                destination_mip_level: 0,
            }
        };
        let copy = graph.add_copy_pass(
            "copy",
            |pass| {
                let source = pass.read_texture(&initialized.output.0, TextureRange::whole());
                let (output, destination) = pass.write_texture(
                    initialized.output.1,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                (output, (source, destination))
            },
            move |commands, _, handles, _| commands.copy_texture(&handles.0, &handles.1, region),
        );
        graph.export_texture(
            copy.output,
            ExportTextureContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph.compile(&caps()).unwrap().graph;
        let registry = TestRegistry::new(device());
        assert!(matches!(
            executor().execute(&compiled, compiled.instantiate_local(FrameInputs::new(())), &registry, &registry),
            Err(ExecutionError::Recording(error)) if error.kind == RecordingErrorKind::InvalidCommandArgument
        ));
    }
}

#[test]
fn texture_copy_rejects_incompatible_formats() {
    for destination_format in [TextureFormat::Rgba8UnormSrgb, TextureFormat::Rgba16Float] {
        let mut graph = RenderGraph::new();
        let source = graph.create_texture("source", texture(TextureFormat::Rgba8Unorm));
        let destination = graph.create_texture("destination", texture(destination_format));
        let initialized = graph.add_compute_pass(
            "init",
            |pass| {
                let (source, _) = pass.write_texture(
                    source,
                    TextureWriteUse::Storage,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                let (destination, _) = pass.write_texture(
                    destination,
                    TextureWriteUse::Storage,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                ((source, destination), ())
            },
            |_, _, _, _| Ok(()),
        );
        let copy = graph.add_copy_pass(
            "copy",
            |pass| {
                let source = pass.read_texture(&initialized.output.0, TextureRange::whole());
                let (output, destination) = pass.write_texture(
                    initialized.output.1,
                    TextureRange::whole(),
                    WriteCoverage::Full,
                );
                (output, (source, destination))
            },
            |commands, _, handles, _| {
                commands.copy_texture(
                    &handles.0,
                    &handles.1,
                    TextureCopyRegion {
                        source_origin: [0; 3],
                        destination_origin: [0; 3],
                        extent: [1, 1, 1],
                        source_mip_level: 0,
                        destination_mip_level: 0,
                    },
                )
            },
        );
        graph.export_texture(
            copy.output,
            ExportTextureContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph.compile(&caps()).unwrap().graph;
        let registry = TestRegistry::new(device());
        assert!(matches!(
            executor().execute(&compiled, compiled.instantiate_local(FrameInputs::new(())), &registry, &registry),
            Err(ExecutionError::Recording(error)) if error.kind == RecordingErrorKind::InvalidCommandArgument
        ));
    }
}

#[test]
fn texture_copy_accepts_matching_srgb_formats() {
    let mut graph = RenderGraph::new();
    let source = graph.create_texture("source", texture(TextureFormat::Rgba8UnormSrgb));
    let destination = graph.create_texture("destination", texture(TextureFormat::Rgba8UnormSrgb));
    let initialized = graph.add_compute_pass(
        "init",
        |pass| {
            let (source, _) = pass.write_texture(
                source,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            let (destination, _) = pass.write_texture(
                destination,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            ((source, destination), ())
        },
        |_, _, _, _| Ok(()),
    );
    let copy = graph.add_copy_pass(
        "copy",
        |pass| {
            let source = pass.read_texture(&initialized.output.0, TextureRange::whole());
            let (output, destination) = pass.write_texture(
                initialized.output.1,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            (output, (source, destination))
        },
        |commands, _, handles, _| {
            commands.copy_texture(
                &handles.0,
                &handles.1,
                TextureCopyRegion {
                    source_origin: [0; 3],
                    destination_origin: [0; 3],
                    extent: [1, 1, 1],
                    source_mip_level: 0,
                    destination_mip_level: 0,
                },
            )
        },
    );
    graph.export_texture(
        copy.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .expect("matching sRGB copies are legal");
    assert!(
        executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::CopyTexture { .. }))
    );
}

#[test]
fn transient_export_retains_submission_lease_until_completion() {
    let mut graph = RenderGraph::new();
    let transient = graph.create_buffer("export", buffer(16));
    let pass = graph.add_compute_pass(
        "write",
        |pass| {
            let (output, _) = pass.write_buffer(
                transient,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    let export = graph.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    let mut frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(
        frame.exports.buffer(export).expect("export").descriptor,
        buffer(16)
    );
    assert!(frame.submission.retained_lease_count() > 0);
    let completion = *frame.submission.completion();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Pending
    );
    executor.try_backend().unwrap().complete(completion);
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    assert_eq!(frame.submission.retained_lease_count(), 0);
}
