//! Declares the one-pass legacy-unlit graph owned by a multi-draw packet.
//!
//! The graph sees only portable buffer imports, their semantic uses, and one
//! transient target.  Snapshot generation reuse is resolved here before graph
//! declaration: repeated draws share one position/index import pair, while
//! every draw retains an independent uniform import and binding identity.

use core::fmt;
use std::{collections::HashMap, sync::Arc};

use fluxel_rendergraph::{
    AttachmentOps, BindingResource, BufferBindingId, BufferRange, BufferRead, ColorAttachmentDesc,
    DeviceCapabilities, ExportBufferContract, ExportBufferSlot, ExportTextureContract,
    ExportTextureSlot, Extent3d, ExternalOwnership, ImportBufferContract, ImportBufferSlot,
    IndexFormat, InitialContents, LoadOp, RenderGraph, ResourceAccessState, StoreOp, TextureDesc,
    TextureDimension, TextureFormat, TextureRange, Viewport, WriteCoverage,
};

use crate::frame_uniform::FRAME_UNIFORM_BYTES;

use super::{RenderPacket, ids::PacketIdIssuer};
use crate::fixed_frame::{camera_bindings, camera_pipeline};

/// A compiled graph plus its private frame-binding topology.
#[derive(Clone)]
pub(in crate::fixed_frame) struct PacketGraph {
    pub(in crate::fixed_frame) compiled: Arc<fluxel_rendergraph::CompiledGraph<()>>,
    pub(in crate::fixed_frame) target_export: ExportTextureSlot,
    pub(in crate::fixed_frame) snapshots: Vec<SnapshotImport>,
    pub(in crate::fixed_frame) draws: Vec<DrawImport>,
}

/// The one imported pair that represents a unique immutable snapshot generation.
#[derive(Clone)]
pub(in crate::fixed_frame) struct SnapshotImport {
    pub(in crate::fixed_frame) source_draw: usize,
    #[allow(dead_code, reason = "packet conformance checks generation reuse")]
    pub(in crate::fixed_frame) generation: u64,
    pub(in crate::fixed_frame) position_slot: ImportBufferSlot,
    pub(in crate::fixed_frame) index_slot: ImportBufferSlot,
    pub(in crate::fixed_frame) position_binding: BufferBindingId,
    pub(in crate::fixed_frame) index_binding: BufferBindingId,
    #[allow(
        dead_code,
        reason = "packet conformance checks restored snapshot states"
    )]
    pub(in crate::fixed_frame) position_export: ExportBufferSlot,
    #[allow(
        dead_code,
        reason = "packet conformance checks restored snapshot states"
    )]
    pub(in crate::fixed_frame) index_export: ExportBufferSlot,
}

/// The independent uniform import and draw count for one insertion-ordered draw.
#[derive(Clone)]
pub(in crate::fixed_frame) struct DrawImport {
    pub(in crate::fixed_frame) uniform_slot: ImportBufferSlot,
    pub(in crate::fixed_frame) uniform_binding: BufferBindingId,
}

struct ImportedSnapshot {
    source_draw: usize,
    generation: u64,
    positions: fluxel_rendergraph::ImportedBuffer,
    indices: fluxel_rendergraph::ImportedBuffer,
    position_binding: BufferBindingId,
    index_binding: BufferBindingId,
}

struct DeclaredDraw {
    position: BufferRead,
    index: BufferRead,
    uniform: BufferRead,
    index_count: u32,
}

/// Why a packet graph could not be declared before any GPU work was accepted.
#[derive(Clone, Debug)]
pub(in crate::fixed_frame) enum PacketGraphBuildError {
    /// The packet-local buffer identity namespace was exhausted.
    IdentityExhausted,
    /// Portable graph compilation rejected the closed declaration.
    Compile(fluxel_rendergraph::CompileError),
}

impl fmt::Display for PacketGraphBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdentityExhausted => {
                formatter.write_str("packet graph identity allocation exhausted")
            }
            Self::Compile(error) => write!(formatter, "packet graph compilation failed: {error}"),
        }
    }
}

impl std::error::Error for PacketGraphBuildError {}

/// Builds the fixed legacy-unlit declaration for one already-validated packet.
pub(in crate::fixed_frame) fn build_packet_graph(
    packet: &RenderPacket,
    capabilities: &DeviceCapabilities,
) -> Result<PacketGraph, PacketGraphBuildError> {
    let mut graph = RenderGraph::new();
    let mut ids = PacketIdIssuer::new();
    let mut snapshots = Vec::<ImportedSnapshot>::new();
    let mut snapshot_indices = HashMap::<u64, usize>::new();
    let mut draw_snapshot_indices = Vec::with_capacity(packet.draws().len());
    let mut uniforms = Vec::with_capacity(packet.draws().len());
    let mut uniform_bindings = Vec::with_capacity(packet.draws().len());

    for (draw_index, draw) in packet.draws().iter().enumerate() {
        let snapshot_index = match snapshot_indices.get(&draw.snapshot().generation()) {
            Some(index) => *index,
            None => {
                let snapshot = draw.snapshot();
                let positions = graph.import_buffer_slot(
                    format!("packet-snapshot-{draw_index}-positions"),
                    snapshot_buffer_contract(snapshot.positions()),
                );
                let indices = graph.import_buffer_slot(
                    format!("packet-snapshot-{draw_index}-indices"),
                    snapshot_buffer_contract(snapshot.indices()),
                );
                let index = snapshots.len();
                snapshots.push(ImportedSnapshot {
                    source_draw: draw_index,
                    generation: snapshot.generation(),
                    positions,
                    indices,
                    position_binding: ids
                        .buffer()
                        .map_err(|_| PacketGraphBuildError::IdentityExhausted)?,
                    index_binding: ids
                        .buffer()
                        .map_err(|_| PacketGraphBuildError::IdentityExhausted)?,
                });
                snapshot_indices.insert(snapshot.generation(), index);
                index
            }
        };
        let uniform = graph.import_buffer_slot(
            format!("packet-draw-{draw_index}-uniform"),
            ImportBufferContract {
                descriptor: fluxel_rendergraph::BufferDesc {
                    size: FRAME_UNIFORM_BYTES as u64,
                },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        draw_snapshot_indices.push(snapshot_index);
        uniforms.push(uniform);
        uniform_bindings.push(
            ids.buffer()
                .map_err(|_| PacketGraphBuildError::IdentityExhausted)?,
        );
    }

    let target = graph.create_texture(
        "packet-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: packet.extent()[0],
                height: packet.extent()[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let extent = packet.extent();
    let pass = graph.add_raster_pass(
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
            let draws = packet
                .draws()
                .iter()
                .enumerate()
                .map(|(draw_index, draw)| {
                    let snapshot = &snapshots[draw_snapshot_indices[draw_index]];
                    DeclaredDraw {
                        position: pass.read_buffer(
                            &snapshot.positions.version,
                            fluxel_rendergraph::BufferReadUse::Vertex,
                            BufferRange::Whole,
                        ),
                        index: pass.read_buffer(
                            &snapshot.indices.version,
                            fluxel_rendergraph::BufferReadUse::Index,
                            BufferRange::Whole,
                        ),
                        uniform: pass.read_buffer(
                            &uniforms[draw_index].version,
                            fluxel_rendergraph::BufferReadUse::Uniform,
                            BufferRange::Whole,
                        ),
                        index_count: draw.snapshot().index_count(),
                    }
                })
                .collect::<Vec<_>>();
            (output, draws)
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
    );

    let snapshots = snapshots
        .into_iter()
        .map(|snapshot| {
            let position_export = graph.export_buffer(
                snapshot.positions.version,
                ExportBufferContract {
                    final_state: ResourceAccessState::CopyDestination,
                },
            );
            let index_export = graph.export_buffer(
                snapshot.indices.version,
                ExportBufferContract {
                    final_state: ResourceAccessState::CopyDestination,
                },
            );
            SnapshotImport {
                source_draw: snapshot.source_draw,
                generation: snapshot.generation,
                position_slot: snapshot.positions.slot,
                index_slot: snapshot.indices.slot,
                position_binding: snapshot.position_binding,
                index_binding: snapshot.index_binding,
                position_export,
                index_export,
            }
        })
        .collect();
    let draws = uniforms
        .into_iter()
        .enumerate()
        .map(|(draw_index, uniform)| DrawImport {
            uniform_slot: uniform.slot,
            uniform_binding: uniform_bindings[draw_index],
        })
        .collect();
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(capabilities)
        .map_err(PacketGraphBuildError::Compile)?
        .graph;
    Ok(PacketGraph {
        compiled: Arc::new(compiled),
        target_export,
        snapshots,
        draws,
    })
}

fn snapshot_buffer_contract(buffer: &fluxel_rhi::UploadedBuffer) -> ImportBufferContract {
    ImportBufferContract {
        descriptor: buffer.buffer().descriptor().buffer,
        initial_state: buffer.outgoing_state(),
        ownership: ExternalOwnership::Caller,
        initial_contents: InitialContents::Defined,
    }
}
