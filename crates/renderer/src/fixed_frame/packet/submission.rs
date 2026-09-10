//! Advances packet uniform uploads and one raster execution without blocking.
//!
//! A packet owns all snapshot reservations as one transaction.  Uniform copies
//! never read those snapshots, so every failure before raster acceptance may
//! release the whole transaction.  Once raster work is accepted, completion is
//! the only proof that every imported snapshot returned to its declared state;
//! any unproven accepted outcome therefore poisons every reservation.

use std::{collections::HashSet, sync::Arc};

use fluxel_rendergraph::{CompletionStatus, ExecutionError, FrameInputs};
use fluxel_rhi::experimental::fixed_artifacts::{
    RasterBackend, RasterKernel, RasterObjectProvider,
};
use fluxel_rhi::{
    BufferDescriptor, MemoryPolicy, NativeExecutionError, PendingBufferUpload, UploadedBuffer,
};

use crate::fixed_frame::{
    FixedFrameExecutionError, FixedFrameRasterObservationError, FixedFrameUniformObservationError,
};

use super::*;
use crate::upload::{SnapshotDrawReservation, SnapshotUseError};

/// Internal progress state.  Only one uniform copy can be pending: starting
/// draw `n + 1` is allowed only after draw `n` has been finalized, which makes
/// later-start failures safe to publish without a host wait.
enum PacketPhase {
    UniformUploading {
        draw_index: usize,
        pending: PendingBufferUpload,
    },
    RasterReady,
    RasterAccepted(fluxel_rendergraph::ExecutedFrame<RasterBackend>),
    Terminal,
}

/// A non-blocking multi-draw packet operation.
pub struct RenderPacketSubmission {
    phase: PacketPhase,
    device: fluxel_rhi::Device,
    executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
    packet: RenderPacket,
    graph: Option<PacketGraph>,
    objects: Option<RasterObjectProvider>,
    ready_uniforms: Vec<UploadedBuffer>,
    reservations: Vec<SnapshotDrawReservation>,
    pub(super) target_export: Option<fluxel_rendergraph::ExportTextureSlot>,
    image: Option<super::super::FrameImage>,
    failure: Option<RenderPacketFailure>,
    #[cfg(all(test, windows))]
    pub(super) completed: Option<fluxel_rendergraph::ExecutedFrame<RasterBackend>>,
}

impl core::fmt::Debug for RenderPacketSubmission {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let phase = match &self.phase {
            PacketPhase::UniformUploading { .. } => "uniform-uploading",
            PacketPhase::RasterReady => "raster-ready",
            PacketPhase::RasterAccepted(_) => "raster-accepted",
            PacketPhase::Terminal => "terminal",
        };
        formatter
            .debug_struct("RenderPacketSubmission")
            .field("draw_count", &self.packet.draws().len())
            .field("phase", &phase)
            .field("complete", &self.image.is_some())
            .field("failed", &self.failure.is_some())
            .finish_non_exhaustive()
    }
}

impl super::super::FixedFrameRenderer {
    /// Starts the owned packet's first uniform upload without waiting.
    ///
    /// All unique snapshot generations are reserved before any native work is
    /// accepted.  The first uniform is special because its rejection can still
    /// return a start error; all later starts happen from [`RenderPacketSubmission::poll`].
    pub fn submit_packet(
        &self,
        packet: RenderPacket,
    ) -> Result<RenderPacketSubmission, RenderPacketStartError> {
        if packet.device() != self.device.identity() {
            return Err(RenderPacketStartError::ForeignPacketDevice);
        }
        let mut reservations = self.reserve_packet_snapshots(&packet)?;
        let graph = match super::graph::build_packet_graph(&packet, &self.capabilities) {
            Ok(graph) => graph,
            Err(super::PacketGraphBuildError::Compile(error)) => {
                release_reservations(&mut reservations);
                return Err(RenderPacketStartError::Graph(error));
            }
            Err(super::PacketGraphBuildError::IdentityExhausted) => {
                release_reservations(&mut reservations);
                return Err(RenderPacketStartError::GraphIdentityExhausted);
            }
        };
        self.start_reserved_packet(packet, graph, reservations)
    }

    /// Starts a packet with a graph allocation supplied by a paired native
    /// fixture.  Production callers cannot select or observe this graph.
    #[cfg(all(test, windows))]
    pub(in crate::fixed_frame) fn submit_packet_with_graph_for_test(
        &self,
        packet: RenderPacket,
        graph: PacketGraph,
    ) -> Result<RenderPacketSubmission, RenderPacketStartError> {
        if packet.device() != self.device.identity() {
            return Err(RenderPacketStartError::ForeignPacketDevice);
        }
        let reservations = self.reserve_packet_snapshots(&packet)?;
        self.start_reserved_packet(packet, graph, reservations)
    }

    /// Reserves every generation once, preserving first-draw error identity.
    fn reserve_packet_snapshots(
        &self,
        packet: &RenderPacket,
    ) -> Result<Vec<SnapshotDrawReservation>, RenderPacketStartError> {
        let mut reservations = Vec::new();
        let mut generations = HashSet::with_capacity(packet.draws().len());
        for (draw_index, draw) in packet.draws().iter().enumerate() {
            if !generations.insert(draw.snapshot().generation()) {
                continue;
            }
            match draw.snapshot().reserve_for_draw() {
                Ok(reservation) => reservations.push(reservation),
                Err(error) => {
                    release_reservations(&mut reservations);
                    return Err(RenderPacketStartError::Reservation {
                        draw_index,
                        cause: reservation_error(error),
                    });
                }
            }
        }
        Ok(reservations)
    }

    /// Shares the one native starter after either production compilation or
    /// test-only graph injection has acquired the same reservation transaction.
    fn start_reserved_packet(
        &self,
        packet: RenderPacket,
        graph: PacketGraph,
        mut reservations: Vec<SnapshotDrawReservation>,
    ) -> Result<RenderPacketSubmission, RenderPacketStartError> {
        let pipeline = match self
            .device
            .create_raster_pipeline(RasterKernel::IndexedPositionFloat32x3CameraMaterial)
        {
            Ok(pipeline) => pipeline,
            Err(error) => {
                release_reservations(&mut reservations);
                return Err(RenderPacketStartError::Pipeline(error));
            }
        };
        let mut objects = RasterObjectProvider::new(&self.device);
        if let Err(error) =
            objects.register_raster_pipeline(super::super::camera_pipeline(), pipeline)
        {
            release_reservations(&mut reservations);
            return Err(RenderPacketStartError::Provider(error));
        }
        if let Err(error) = objects.register_raster_uniform_bindings(
            super::super::camera_bindings(),
            super::super::camera_pipeline(),
        ) {
            release_reservations(&mut reservations);
            return Err(RenderPacketStartError::Provider(error));
        }

        let pending = match start_uniform(&self.device, packet.draws()[0].uniform()) {
            Ok(pending) => pending,
            Err(error) => {
                release_reservations(&mut reservations);
                return Err(RenderPacketStartError::FirstUniformStart(error));
            }
        };

        Ok(RenderPacketSubmission {
            phase: PacketPhase::UniformUploading {
                draw_index: 0,
                pending,
            },
            device: self.device.clone(),
            executor: Arc::clone(&self.executor),
            packet,
            graph: Some(graph),
            objects: Some(objects),
            ready_uniforms: Vec::new(),
            reservations,
            target_export: None,
            image: None,
            failure: None,
            #[cfg(all(test, windows))]
            completed: None,
        })
    }
}

impl RenderPacketSubmission {
    #[cfg(all(test, windows))]
    pub(super) fn has_accepted_raster(&self) -> bool {
        matches!(&self.phase, PacketPhase::RasterAccepted(_))
    }

    #[cfg(all(test, windows))]
    pub(super) fn packet_graph(&self) -> &PacketGraph {
        self.graph
            .as_ref()
            .expect("active packet submission retains its graph")
    }

    /// Polls one serial uniform upload or the one accepted raster submission.
    pub fn poll(&mut self) -> RenderPacketStatus {
        if let Some(image) = &self.image {
            return RenderPacketStatus::Complete(image.clone());
        }
        if let Some(failure) = &self.failure {
            return RenderPacketStatus::Failed(failure.clone());
        }

        // Install Terminal while moving the state out.  Each branch below must
        // restore a retryable phase or publish exactly one terminal outcome.
        let phase = core::mem::replace(&mut self.phase, PacketPhase::Terminal);
        match phase {
            PacketPhase::UniformUploading {
                draw_index,
                pending,
            } => self.poll_uniform(draw_index, pending),
            PacketPhase::RasterReady => self.submit_raster(),
            PacketPhase::RasterAccepted(frame) => self.poll_raster(frame),
            PacketPhase::Terminal => unreachable!("terminal packet has image or failure"),
        }
    }

    fn poll_uniform(
        &mut self,
        draw_index: usize,
        pending: PendingBufferUpload,
    ) -> RenderPacketStatus {
        match pending.status() {
            Ok(CompletionStatus::Pending) => {
                self.phase = PacketPhase::UniformUploading {
                    draw_index,
                    pending,
                };
                RenderPacketStatus::Pending
            }
            Ok(CompletionStatus::Complete) => match pending.finalize() {
                Ok(uniform) => {
                    debug_assert_eq!(self.ready_uniforms.len(), draw_index);
                    self.ready_uniforms.push(uniform);
                    self.start_next_uniform_or_raster(draw_index + 1)
                }
                Err(incomplete) => {
                    self.finish_pre_raster(RenderPacketFailure::UniformObservation {
                        draw_index,
                        cause: FixedFrameUniformObservationError::Finalize(incomplete.status()),
                    })
                }
            },
            Ok(CompletionStatus::Failed(cause)) => {
                self.finish_pre_raster(RenderPacketFailure::UniformCompletion { draw_index, cause })
            }
            Ok(_) => self.finish_pre_raster(RenderPacketFailure::UniformObservation {
                draw_index,
                cause: FixedFrameUniformObservationError::UnknownCompletionStatus,
            }),
            Err(cause) => self.finish_pre_raster(RenderPacketFailure::UniformObservation {
                draw_index,
                cause: FixedFrameUniformObservationError::Status(cause),
            }),
        }
    }

    fn start_next_uniform_or_raster(&mut self, next_index: usize) -> RenderPacketStatus {
        if next_index == self.packet.draws().len() {
            self.phase = PacketPhase::RasterReady;
            return self.submit_raster();
        }
        match start_uniform(&self.device, self.packet.draws()[next_index].uniform()) {
            Ok(pending) => {
                self.phase = PacketPhase::UniformUploading {
                    draw_index: next_index,
                    pending,
                };
                RenderPacketStatus::Pending
            }
            Err(cause) => self.finish_pre_raster(RenderPacketFailure::UniformStart {
                draw_index: next_index,
                cause,
            }),
        }
    }

    fn submit_raster(&mut self) -> RenderPacketStatus {
        let graph = self
            .graph
            .as_ref()
            .expect("pre-raster packet retains graph");
        let objects = self
            .objects
            .as_ref()
            .expect("pre-raster packet retains object provider");
        debug_assert_eq!(self.ready_uniforms.len(), self.packet.draws().len());
        let resources = PacketResources::new(&self.packet, graph, &self.ready_uniforms);
        let mut inputs = FrameInputs::new(());
        for snapshot in &graph.snapshots {
            inputs
                .bind_buffer(snapshot.position_slot, snapshot.position_binding)
                .bind_buffer(snapshot.index_slot, snapshot.index_binding);
        }
        for draw in &graph.draws {
            inputs.bind_buffer(draw.uniform_slot, draw.uniform_binding);
        }
        match self.executor.execute(
            &graph.compiled,
            graph.compiled.instantiate_local(inputs),
            &resources,
            objects,
        ) {
            Ok(frame) => {
                self.target_export = Some(graph.target_export);
                self.phase = PacketPhase::RasterAccepted(frame);
                RenderPacketStatus::Pending
            }
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = PacketPhase::RasterReady;
                RenderPacketStatus::Busy
            }
            Err(error) => self.finish_pre_raster(RenderPacketFailure::RasterStart {
                cause: execution_error(error),
            }),
        }
    }

    fn poll_raster(
        &mut self,
        mut frame: fluxel_rendergraph::ExecutedFrame<RasterBackend>,
    ) -> RenderPacketStatus {
        match frame.submission.status() {
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = PacketPhase::RasterAccepted(frame);
                RenderPacketStatus::Busy
            }
            Err(error) => self.finish_accepted(RenderPacketFailure::RasterObservation {
                cause: FixedFrameRasterObservationError::Execution(execution_error(error)),
            }),
            Ok(CompletionStatus::Pending) => {
                self.phase = PacketPhase::RasterAccepted(frame);
                RenderPacketStatus::Pending
            }
            Ok(CompletionStatus::Complete) => {
                let Some(exported) = frame.exports.texture(
                    self.target_export
                        .expect("accepted packet has a target export"),
                ) else {
                    return self.finish_accepted(RenderPacketFailure::RasterObservation {
                        cause: FixedFrameRasterObservationError::MissingTargetExport,
                    });
                };
                let image = super::super::FrameImage {
                    extent: [
                        exported.descriptor.extent.width,
                        exported.descriptor.extent.height,
                    ],
                    format: exported.descriptor.format,
                    _texture: exported.physical.clone(),
                };
                release_reservations_complete(&mut self.reservations);
                self.image = Some(image.clone());
                #[cfg(all(test, windows))]
                {
                    self.completed = Some(frame);
                }
                #[cfg(not(all(test, windows)))]
                self.graph.take();
                self.objects.take();
                self.phase = PacketPhase::Terminal;
                RenderPacketStatus::Complete(image)
            }
            Ok(CompletionStatus::Failed(cause)) => {
                self.finish_accepted(RenderPacketFailure::RasterCompletion { cause })
            }
            Ok(_) => self.finish_accepted(RenderPacketFailure::RasterObservation {
                cause: FixedFrameRasterObservationError::UnknownCompletionStatus,
            }),
        }
    }

    fn finish_pre_raster(&mut self, failure: RenderPacketFailure) -> RenderPacketStatus {
        self.graph.take();
        self.objects.take();
        self.ready_uniforms.clear();
        release_reservations(&mut self.reservations);
        self.phase = PacketPhase::Terminal;
        self.failure = Some(failure.clone());
        RenderPacketStatus::Failed(failure)
    }

    fn finish_accepted(&mut self, failure: RenderPacketFailure) -> RenderPacketStatus {
        self.graph.take();
        self.objects.take();
        self.ready_uniforms.clear();
        poison_reservations(&mut self.reservations);
        self.phase = PacketPhase::Terminal;
        self.failure = Some(failure.clone());
        RenderPacketStatus::Failed(failure)
    }
}

impl Drop for RenderPacketSubmission {
    fn drop(&mut self) {
        if matches!(self.phase, PacketPhase::RasterAccepted(_)) {
            poison_reservations(&mut self.reservations);
        } else {
            release_reservations(&mut self.reservations);
        }
    }
}

fn start_uniform(
    device: &fluxel_rhi::Device,
    uniform: &crate::frame_uniform::FrameUniform,
) -> Result<PendingBufferUpload, fluxel_rhi::BufferUploadError> {
    device.upload_immutable_buffer(
        BufferDescriptor {
            buffer: fluxel_rendergraph::BufferDesc {
                size: crate::frame_uniform::FRAME_UNIFORM_BYTES as u64,
            },
            usage: fluxel_rendergraph::BufferUsage::from_kinds([
                fluxel_rendergraph::BufferUsageKind::CopyDestination,
                fluxel_rendergraph::BufferUsageKind::Uniform,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        },
        uniform.bytes(),
    )
}

fn reservation_error(error: SnapshotUseError) -> RenderPacketReservationError {
    match error {
        SnapshotUseError::InFlight => RenderPacketReservationError::Busy,
        SnapshotUseError::Poisoned => RenderPacketReservationError::Poisoned,
    }
}

fn release_reservations(reservations: &mut Vec<SnapshotDrawReservation>) {
    for reservation in reservations.drain(..) {
        reservation.release_before_submit();
    }
}

fn release_reservations_complete(reservations: &mut Vec<SnapshotDrawReservation>) {
    for mut reservation in reservations.drain(..) {
        reservation.release_complete();
    }
}

fn poison_reservations(reservations: &mut Vec<SnapshotDrawReservation>) {
    for mut reservation in reservations.drain(..) {
        reservation.poison();
    }
}

pub(super) fn execution_error(
    error: ExecutionError<NativeExecutionError>,
) -> FixedFrameExecutionError {
    match error {
        ExecutionError::FrameBinding(error) => FixedFrameExecutionError::FrameBinding(error),
        ExecutionError::Recording(error) => FixedFrameExecutionError::Recording(error),
        ExecutionError::CapabilityMismatch => FixedFrameExecutionError::CapabilityMismatch,
        ExecutionError::WrongCompiledGraph => FixedFrameExecutionError::WrongCompiledGraph,
        ExecutionError::UnsupportedExecutionFeature(feature) => {
            FixedFrameExecutionError::UnsupportedExecutionFeature(feature)
        }
        ExecutionError::Backend(error) => FixedFrameExecutionError::Backend(error),
        ExecutionError::ExecutorBusy => unreachable!("busy is handled before failure mapping"),
        _ => FixedFrameExecutionError::Unknown,
    }
}
