//! The deliberately closed headless indexed-frame coordinator.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    AttachmentOps, BindingResource, BindingSetId, BoundBuffer, BufferBindingId, BufferRange,
    ColorAttachmentDesc, CompletionFailure, CompletionStatus, DeviceIdentity, ExecutionError,
    ExportBufferContract, ExportTextureContract, Extent3d, ExternalOwnership, FrameBindingError,
    FrameBindingErrorKind, FrameInputs, FrameResourceProvider, ImportBufferContract, IndexFormat,
    InitialContents, LoadOp, RasterPipelineId, RenderGraph, ResourceAccessState, StoreOp,
    TextureDesc, TextureDimension, TextureFormat, TextureRange, Viewport, WriteCoverage,
};
use fluxel_rhi::{
    Buffer, BufferDescriptor, BufferUploadError, Device, MemoryPolicy, PendingBufferUpload,
    RasterBackend, RasterKernel, RasterObjectProvider, ResourceLease, Texture, UploadedBuffer,
};

use crate::frame_uniform::{FRAME_UNIFORM_BYTES, FrameUniform};

#[cfg(all(test, windows))]
fn native_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
use crate::upload::{IndexedMeshSnapshot, SnapshotUseError};

fn position_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0001)
}
fn index_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0002)
}
#[cfg(test)]
fn fixed_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0210_0001)
}
fn uniform_binding() -> BufferBindingId {
    BufferBindingId::new(0x0220_0003)
}
fn camera_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0220_0001)
}
fn camera_bindings() -> BindingSetId {
    BindingSetId::new(0x0220_0001)
}

/// Coordinates the fixed headless `f32x3/u32` indexed draw slice.
///
/// This type owns no application-visible native resource handles.  It accepts
/// only a completed [`IndexedMeshSnapshot`] and produces opaque image metadata
/// after non-blocking completion observation.
pub struct FixedFrameRenderer {
    device: Device,
    executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
}

impl FixedFrameRenderer {
    /// Creates a renderer over an already-opened headless device.
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            executor: Arc::new(fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(
                device.clone(),
            ))),
            device,
        }
    }

    /// Starts one fixed indexed draw without waiting for the GPU.
    ///
    /// The snapshot generation is reserved until this returned operation proves
    /// completion.  A graph rejection before native submission releases the
    /// reservation; only accepted but unproven Raster work poisons it on drop.
    pub fn draw(
        &self,
        snapshot: &IndexedMeshSnapshot,
        camera: &crate::Camera,
        material: &crate::BasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(DrawStartError::InvalidExtent);
        }
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
            return Err(DrawStartError::InvalidIndexCount);
        }
        let uniform = FrameUniform::new(camera, material)
            .map_err(|_| DrawStartError::InvalidCameraMaterial)?;
        let graph = Arc::new(
            build_camera_graph(snapshot, extent)
                .map_err(|error| DrawStartError::Graph(error.to_string()))?,
        );
        self.start_camera(snapshot, graph, uniform)
    }

    fn start_camera(
        &self,
        snapshot: &IndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        let reservation = snapshot.reserve_for_draw().map_err(|error| match error {
            SnapshotUseError::InFlight => DrawStartError::SnapshotInFlight,
            SnapshotUseError::Poisoned => DrawStartError::SnapshotPoisoned,
        })?;
        let pipeline = match self
            .device
            .create_raster_pipeline(RasterKernel::IndexedPositionFloat32x3CameraMaterial)
        {
            Ok(pipeline) => pipeline,
            Err(error) => {
                reservation.release_before_submit();
                return Err(DrawStartError::Pipeline(error.to_string()));
            }
        };
        let mut objects = RasterObjectProvider::new(&self.device);
        if let Err(error) = objects.register_raster_pipeline(camera_pipeline(), pipeline) {
            reservation.release_before_submit();
            return Err(DrawStartError::Pipeline(error.to_string()));
        }
        if let Err(error) =
            objects.register_raster_uniform_bindings(camera_bindings(), camera_pipeline())
        {
            reservation.release_before_submit();
            return Err(DrawStartError::Pipeline(error.to_string()));
        }
        let pending = match self.device.upload_immutable_buffer(
            BufferDescriptor {
                buffer: fluxel_rendergraph::BufferDesc {
                    size: FRAME_UNIFORM_BYTES as u64,
                },
                usage: fluxel_rendergraph::BufferUsage::from_kinds([
                    fluxel_rendergraph::BufferUsageKind::CopyDestination,
                    fluxel_rendergraph::BufferUsageKind::Uniform,
                ]),
                memory: MemoryPolicy::DeviceOnly,
            },
            uniform.bytes(),
        ) {
            Ok(pending) => pending,
            Err(error) => {
                reservation.release_before_submit();
                return Err(DrawStartError::UniformStart(error));
            }
        };
        Ok(FixedFrameSubmission {
            phase: CameraPhase::Uploading(pending),
            executor: Arc::clone(&self.executor),
            snapshot: snapshot.clone(),
            graph: Some(graph),
            objects: Some(objects),
            reservation: Some(reservation),
            target_export: None,
            image: None,
            failure: None,
            #[cfg(test)]
            completed: None,
        })
    }
}

impl fmt::Debug for FixedFrameRenderer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixedFrameRenderer")
            .finish_non_exhaustive()
    }
}

/// Why a fixed frame could not be accepted for submission.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DrawStartError {
    /// The requested target dimensions contain zero.
    InvalidExtent,
    /// The ready snapshot belongs to another native device.
    ForeignSnapshotDevice,
    /// This fixed triangle-list recipe requires a nonzero multiple of three indices.
    InvalidIndexCount,
    /// Another draw is active for this snapshot generation.
    SnapshotInFlight,
    /// A prior accepted draw did not prove its terminal state.
    SnapshotPoisoned,
    /// Camera/material values cannot produce the closed uniform ABI.
    InvalidCameraMaterial,
    /// The immutable uniform upload was rejected before work was accepted.
    UniformStart(BufferUploadError),
    /// Graph compilation rejected the fixed declaration.
    Graph(String),
    /// The device could not create the closed raster artifact.
    Pipeline(String),
}

impl fmt::Display for DrawStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent => formatter.write_str("fixed frame extent must be nonzero"),
            Self::ForeignSnapshotDevice => {
                formatter.write_str("snapshot belongs to another device")
            }
            Self::InvalidIndexCount => {
                formatter.write_str("fixed frame needs a nonzero triangle-list index count")
            }
            Self::SnapshotInFlight => formatter.write_str("snapshot draw is already in flight"),
            Self::SnapshotPoisoned => formatter.write_str("snapshot draw state is poisoned"),
            Self::InvalidCameraMaterial => {
                formatter.write_str("invalid camera or material uniform input")
            }
            Self::UniformStart(error) => {
                write!(formatter, "frame uniform upload did not start: {error}")
            }
            Self::Graph(error) => write!(formatter, "fixed frame graph rejected: {error}"),
            Self::Pipeline(error) => write!(formatter, "fixed frame pipeline rejected: {error}"),
        }
    }
}

impl std::error::Error for DrawStartError {}

/// A terminal failure of the two-phase fixed frame operation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FixedFrameFailure {
    /// The uniform upload reached a terminal GPU failure.
    UniformCompletion(CompletionFailure),
    /// Uniform completion could not be observed or finalized.
    UniformObservation(String),
    /// Raster recording or submit was rejected before raster work was accepted.
    RasterStart(String),
    /// An accepted raster submission did not complete successfully.
    RasterCompletion(CompletionFailure),
}

impl fmt::Display for FixedFrameFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UniformCompletion(error) => {
                write!(formatter, "uniform completion failed: {error:?}")
            }
            Self::UniformObservation(error) => {
                write!(formatter, "uniform completion observation failed: {error}")
            }
            Self::RasterStart(error) => write!(formatter, "raster did not start: {error}"),
            Self::RasterCompletion(error) => {
                write!(formatter, "raster completion failed: {error:?}")
            }
        }
    }
}
impl std::error::Error for FixedFrameFailure {}

/// A non-blocking two-phase fixed frame operation.
pub struct FixedFrameSubmission {
    phase: CameraPhase,
    executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
    snapshot: IndexedMeshSnapshot,
    graph: Option<Arc<CameraGraph>>,
    objects: Option<RasterObjectProvider>,
    reservation: Option<crate::upload::SnapshotDrawReservation>,
    target_export: Option<fluxel_rendergraph::ExportTextureSlot>,
    image: Option<FrameImage>,
    failure: Option<FixedFrameFailure>,
    #[cfg(test)]
    completed: Option<fluxel_rendergraph::ExecutedFrame<RasterBackend>>,
}

enum CameraPhase {
    Uploading(PendingBufferUpload),
    RasterReady(UploadedBuffer),
    RasterAccepted(fluxel_rendergraph::ExecutedFrame<RasterBackend>),
    Terminal,
}

impl FixedFrameSubmission {
    /// Polls GPU completion without waiting.
    pub fn poll(&mut self) -> FixedFrameStatus {
        if let Some(image) = &self.image {
            return FixedFrameStatus::Complete(image.clone());
        }
        if let Some(failure) = &self.failure {
            return FixedFrameStatus::Failed(failure.clone());
        }
        let phase = core::mem::replace(&mut self.phase, CameraPhase::Terminal);
        match phase {
            CameraPhase::Uploading(upload) => self.poll_upload(upload),
            CameraPhase::RasterReady(uniform) => self.submit_raster(uniform),
            CameraPhase::RasterAccepted(frame) => self.poll_raster(frame),
            CameraPhase::Terminal => unreachable!("terminal operation has image or failure"),
        }
    }

    fn poll_upload(&mut self, upload: PendingBufferUpload) -> FixedFrameStatus {
        match upload.status() {
            Ok(CompletionStatus::Pending) => {
                self.phase = CameraPhase::Uploading(upload);
                FixedFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => match upload.finalize() {
                Ok(uniform) => self.submit_raster(uniform),
                Err(incomplete) => {
                    self.finish_pre_accept(FixedFrameFailure::UniformObservation(format!(
                        "finalize observed {:?}",
                        incomplete.status()
                    )));
                    FixedFrameStatus::Failed(self.failure.clone().unwrap())
                }
            },
            Ok(CompletionStatus::Failed(error)) => {
                self.finish_pre_accept(FixedFrameFailure::UniformCompletion(error));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
            Ok(_) => {
                self.finish_pre_accept(FixedFrameFailure::UniformObservation(
                    "unknown completion state".into(),
                ));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
            Err(error) => {
                self.finish_pre_accept(FixedFrameFailure::UniformObservation(error.to_string()));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
        }
    }

    fn submit_raster(&mut self, uniform: UploadedBuffer) -> FixedFrameStatus {
        let graph = self.graph.as_ref().expect("pre-accept retains graph");
        let objects = self.objects.as_ref().expect("pre-accept retains objects");
        let resources = CameraResources {
            device: self.snapshot.positions().buffer().device_identity(),
            positions: self.snapshot.positions().buffer().clone(),
            indices: self.snapshot.indices().buffer().clone(),
            uniform: uniform.buffer().clone(),
            position_state: self.snapshot.positions().outgoing_state(),
            index_state: self.snapshot.indices().outgoing_state(),
        };
        let mut inputs = FrameInputs::new(());
        inputs
            .bind_buffer(graph.position_slot, position_binding())
            .bind_buffer(graph.index_slot, index_binding())
            .bind_buffer(graph.uniform_slot, uniform_binding());
        match self.executor.execute(
            &graph.compiled,
            graph.compiled.instantiate_local(inputs),
            &resources,
            objects,
        ) {
            Ok(frame) => {
                self.target_export = Some(graph.target_export);
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Pending
            }
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = CameraPhase::RasterReady(uniform);
                FixedFrameStatus::Busy
            }
            Err(error) => {
                self.finish_pre_accept(FixedFrameFailure::RasterStart(error.to_string()));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
        }
    }

    fn poll_raster(
        &mut self,
        mut frame: fluxel_rendergraph::ExecutedFrame<RasterBackend>,
    ) -> FixedFrameStatus {
        match frame.submission.status() {
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Busy
            }
            Err(_) => self.finish_accepted(FixedFrameFailure::RasterCompletion(
                CompletionFailure::DeviceLost,
            )),
            Ok(CompletionStatus::Pending) => {
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => {
                let exported = frame
                    .exports
                    .texture(
                        self.target_export
                            .expect("accepted frame has target export"),
                    )
                    .expect("fixed graph exports its target");
                let image = FrameImage {
                    extent: [
                        exported.descriptor.extent.width,
                        exported.descriptor.extent.height,
                    ],
                    format: exported.descriptor.format,
                    _texture: exported.physical.clone(),
                };
                self.reservation
                    .take()
                    .expect("accepted fixed frame owns its reservation")
                    .release_complete();
                self.image = Some(image.clone());
                #[cfg(test)]
                {
                    self.completed = Some(frame);
                }
                #[cfg(not(test))]
                {
                    // Completion has copied the only application-visible
                    // ownership into FrameImage; graph/provider artifacts
                    // are no longer needed by the public operation.
                    self.graph.take();
                    self.objects.take();
                }
                self.phase = CameraPhase::Terminal;
                FixedFrameStatus::Complete(image)
            }
            Ok(CompletionStatus::Failed(failure)) => {
                self.finish_accepted(FixedFrameFailure::RasterCompletion(failure))
            }
            Ok(_) => self.finish_accepted(FixedFrameFailure::RasterCompletion(
                CompletionFailure::DeviceLost,
            )),
        }
    }

    fn finish_pre_accept(&mut self, failure: FixedFrameFailure) {
        self.graph.take();
        self.objects.take();
        if let Some(reservation) = self.reservation.take() {
            reservation.release_before_submit();
        }
        self.phase = CameraPhase::Terminal;
        self.failure = Some(failure);
    }
    fn finish_accepted(&mut self, failure: FixedFrameFailure) -> FixedFrameStatus {
        if let Some(mut reservation) = self.reservation.take() {
            reservation.poison();
        }
        self.phase = CameraPhase::Terminal;
        self.failure = Some(failure.clone());
        FixedFrameStatus::Failed(failure)
    }
}

impl Drop for FixedFrameSubmission {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            if matches!(self.phase, CameraPhase::RasterAccepted(_)) {
                let mut reservation = reservation;
                reservation.poison();
            } else {
                reservation.release_before_submit();
            }
        }
    }
}

/// The observed state of a [`FixedFrameSubmission`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FixedFrameStatus {
    /// Uniform upload or an accepted raster submission is still incomplete.
    Pending,
    /// The executor is momentarily held by another safe operation; retry poll.
    Busy,
    /// The fixed frame completed and its opaque output is available.
    Complete(FrameImage),
    /// The GPU did not establish the promised terminal state.
    Failed(FixedFrameFailure),
}

/// Opaque ownership plus metadata for one completed fixed target.
#[derive(Clone, Debug)]
pub struct FrameImage {
    extent: [u32; 2],
    format: TextureFormat,
    // Retained privately so application code cannot observe or use a native object.
    _texture: Texture,
}

impl FrameImage {
    /// Returns the pixel dimensions of this headless image.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the fixed image format.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.format
    }
}

#[cfg(test)]
struct SnapshotResources {
    device: DeviceIdentity,
    positions: Buffer,
    indices: Buffer,
    position_state: ResourceAccessState,
    index_state: ResourceAccessState,
}

#[cfg(test)]
impl FrameResourceProvider<RasterBackend> for SnapshotResources {
    fn texture(
        &self,
        _: fluxel_rendergraph::TextureBindingId,
    ) -> Result<fluxel_rendergraph::BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        Err(missing_binding(
            FrameBindingErrorKind::MissingTexture,
            "fixed frame has no texture imports",
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let (buffer, initial_state) = match id {
            id if id == position_binding() => (&self.positions, self.position_state),
            id if id == index_binding() => (&self.indices, self.index_state),
            _ => {
                return Err(missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "unknown fixed frame buffer",
                ));
            }
        };
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor: buffer.descriptor().buffer,
            usage: buffer.allowed_usage(),
            initial_state,
            lease: buffer.lease().into(),
        })
    }
}

fn missing_binding(kind: FrameBindingErrorKind, detail: &str) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    }
}

#[cfg(test)]
struct FixedGraph {
    compiled: fluxel_rendergraph::CompiledGraph<()>,
    position_slot: fluxel_rendergraph::ImportBufferSlot,
    index_slot: fluxel_rendergraph::ImportBufferSlot,
    target_export: fluxel_rendergraph::ExportTextureSlot,
    #[allow(
        dead_code,
        reason = "U02 validates restored imported states through these exports"
    )]
    position_export: fluxel_rendergraph::ExportBufferSlot,
    #[allow(
        dead_code,
        reason = "U02 validates restored imported states through these exports"
    )]
    index_export: fluxel_rendergraph::ExportBufferSlot,
}

struct CameraGraph {
    compiled: fluxel_rendergraph::CompiledGraph<()>,
    position_slot: fluxel_rendergraph::ImportBufferSlot,
    index_slot: fluxel_rendergraph::ImportBufferSlot,
    uniform_slot: fluxel_rendergraph::ImportBufferSlot,
    target_export: fluxel_rendergraph::ExportTextureSlot,
    #[allow(
        dead_code,
        reason = "U03 test-only evidence checks restored snapshot states"
    )]
    position_export: fluxel_rendergraph::ExportBufferSlot,
    #[allow(
        dead_code,
        reason = "U03 test-only evidence checks restored snapshot states"
    )]
    index_export: fluxel_rendergraph::ExportBufferSlot,
}

struct CameraResources {
    device: DeviceIdentity,
    positions: Buffer,
    indices: Buffer,
    uniform: Buffer,
    position_state: ResourceAccessState,
    index_state: ResourceAccessState,
}
impl FrameResourceProvider<RasterBackend> for CameraResources {
    fn texture(
        &self,
        _: fluxel_rendergraph::TextureBindingId,
    ) -> Result<fluxel_rendergraph::BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        Err(missing_binding(
            FrameBindingErrorKind::MissingTexture,
            "camera frame has no texture imports",
        ))
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let buffer = if id == position_binding() {
            &self.positions
        } else if id == index_binding() {
            &self.indices
        } else if id == uniform_binding() {
            &self.uniform
        } else {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingBuffer,
                "unknown camera frame buffer",
            ));
        };
        let initial_state = if id == position_binding() {
            self.position_state
        } else if id == index_binding() {
            self.index_state
        } else {
            ResourceAccessState::CopyDestination
        };
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor: buffer.descriptor().buffer,
            usage: buffer.allowed_usage(),
            initial_state,
            lease: buffer.lease().into(),
        })
    }
}

fn build_camera_graph(
    snapshot: &IndexedMeshSnapshot,
    extent: [u32; 2],
) -> Result<CameraGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot(
        "camera-frame-positions",
        ImportBufferContract {
            descriptor: snapshot.positions().buffer().descriptor().buffer,
            initial_state: snapshot.positions().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let indices = graph.import_buffer_slot(
        "camera-frame-indices",
        ImportBufferContract {
            descriptor: snapshot.indices().buffer().descriptor().buffer,
            initial_state: snapshot.indices().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let uniform = graph.import_buffer_slot(
        "camera-frame-uniform",
        ImportBufferContract {
            descriptor: fluxel_rendergraph::BufferDesc {
                size: FRAME_UNIFORM_BYTES as u64,
            },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "camera-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "camera-material-indexed",
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
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let indices_read = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            let uniform_read = pass.read_buffer(
                &uniform.version,
                fluxel_rendergraph::BufferReadUse::Uniform,
                BufferRange::Whole,
            );
            (output, (vertices, indices_read, uniform_read))
        },
        move |commands, resolver, data, _| {
            commands.set_pipeline(camera_pipeline())?;
            let bindings = resolver.resolve_bindings(
                camera_bindings(),
                &[BindingResource::BufferRead(&data.2)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())?
        .graph;
    Ok(CameraGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        uniform_slot: uniform.slot,
        target_export,
        position_export,
        index_export,
    })
}

#[cfg(test)]
fn build_graph(
    snapshot: &IndexedMeshSnapshot,
    extent: [u32; 2],
) -> Result<FixedGraph, fluxel_rendergraph::CompileError> {
    let mut graph = RenderGraph::new();
    let positions = graph.import_buffer_slot(
        "fixed-frame-positions",
        ImportBufferContract {
            descriptor: snapshot.positions().buffer().descriptor().buffer,
            initial_state: snapshot.positions().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let indices = graph.import_buffer_slot(
        "fixed-frame-indices",
        ImportBufferContract {
            descriptor: snapshot.indices().buffer().descriptor().buffer,
            initial_state: snapshot.indices().outgoing_state(),
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let target = graph.create_texture(
        "fixed-frame-target",
        TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
    );
    let index_count = snapshot.index_count();
    let pass = graph.add_raster_pass(
        "fixed-frame-indexed-unlit",
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
            let vertices = pass.read_buffer(
                &positions.version,
                fluxel_rendergraph::BufferReadUse::Vertex,
                BufferRange::Whole,
            );
            let index = pass.read_buffer(
                &indices.version,
                fluxel_rendergraph::BufferReadUse::Index,
                BufferRange::Whole,
            );
            (output, (vertices, index))
        },
        move |commands, _, data, _| {
            commands.set_pipeline(fixed_pipeline())?;
            commands.set_vertex_buffer(0, &data.0)?;
            commands.set_index_buffer(&data.1, IndexFormat::Uint32)?;
            commands.set_viewport(Viewport {
                x: 0.0,
                y: 0.0,
                width: extent[0] as f32,
                height: extent[1] as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            })?;
            commands.draw_indexed(0..index_count, 0, 0..1)
        },
    );
    let position_export = graph.export_buffer(
        positions.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let index_export = graph.export_buffer(
        indices.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let target_export = graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())?
        .graph;
    Ok(FixedGraph {
        compiled,
        position_slot: positions.slot,
        index_slot: indices.slot,
        target_export,
        position_export,
        index_export,
    })
}

#[cfg(test)]
mod legacy_u02 {
    use super::*;
    use crate::{Geometry, IndexedMeshUpload, IndexedMeshUploadStatus};
    use fluxel_rhi::{
        Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test,
    };
    use std::{
        thread,
        time::{Duration, Instant},
    };

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn u02_legacy_fixed_color_dx12() {
        run(Backend::Dx12);
    }
    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn u02_legacy_fixed_color_vulkan() {
        run(Backend::Vulkan);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires DX12 and Vulkan devices with required validation"]
    fn u02_legacy_paired_same_compiled_graph() {
        let _guard = native_fixture_guard();
        let dx12 = open(Backend::Dx12);
        let vulkan = open(Backend::Vulkan);
        let geometry = geometry();
        let dx_snapshot = ready_snapshot(&dx12, &geometry);
        let vk_snapshot = ready_snapshot(&vulkan, &geometry);
        let graph = build_graph(&dx_snapshot, [8, 8]).unwrap();
        let expected = oracle();
        let dx_actual = execute(&dx12, &dx_snapshot, &graph);
        assert_eq!(dx_actual, expected);
        artifact(
            Backend::Dx12,
            &dx12,
            &dx_snapshot,
            &graph,
            &expected,
            &dx_actual,
            "paired",
        );
        let vk_actual = execute(&vulkan, &vk_snapshot, &graph);
        assert_eq!(vk_actual, expected);
        artifact(
            Backend::Vulkan,
            &vulkan,
            &vk_snapshot,
            &graph,
            &expected,
            &vk_actual,
            "paired",
        );
    }

    #[cfg(windows)]
    fn run(backend: Backend) {
        let _guard = native_fixture_guard();
        let device = open(backend);
        let geometry = geometry();
        let snapshot = ready_snapshot(&device, &geometry);
        let graph = build_graph(&snapshot, [8, 8]).unwrap();
        let actual = execute(&device, &snapshot, &graph);
        let expected = oracle();
        assert_eq!(
            actual,
            expected,
            "U02 {backend:?} first difference: {:?}",
            first_difference(&actual, &expected)
        );
        artifact(
            backend,
            &device,
            &snapshot,
            &graph,
            &expected,
            &actual,
            "independent",
        );
    }

    #[cfg(windows)]
    fn open(backend: Backend) -> Device {
        Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap()
    }

    #[cfg(windows)]
    fn geometry() -> Geometry {
        Geometry::from_positions(vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap()
    }

    #[cfg(windows)]
    fn ready_snapshot(device: &Device, geometry: &Geometry) -> IndexedMeshSnapshot {
        let mut upload = IndexedMeshUpload::begin(device, geometry).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match upload.poll() {
                IndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
                IndexedMeshUploadStatus::Pending => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
                IndexedMeshUploadStatus::Failed(e) => panic!("legacy upload {e:?}"),
            }
        }
    }

    #[cfg(windows)]
    fn execute(device: &Device, snapshot: &IndexedMeshSnapshot, graph: &FixedGraph) -> Vec<u8> {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let mut objects = RasterObjectProvider::new(device);
        objects
            .register_raster_pipeline(
                fixed_pipeline(),
                device
                    .create_raster_pipeline(RasterKernel::IndexedPositionFloat32x3)
                    .unwrap(),
            )
            .unwrap();
        let resources = SnapshotResources {
            device: device.identity(),
            positions: snapshot.positions().buffer().clone(),
            indices: snapshot.indices().buffer().clone(),
            position_state: snapshot.positions().outgoing_state(),
            index_state: snapshot.indices().outgoing_state(),
        };
        let mut inputs = FrameInputs::new(());
        inputs
            .bind_buffer(graph.position_slot, position_binding())
            .bind_buffer(graph.index_slot, index_binding());
        let executor = fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone()));
        let mut frame = executor
            .execute(
                &graph.compiled,
                graph.compiled.instantiate_local(inputs),
                &resources,
                &objects,
            )
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match frame.submission.status().unwrap() {
                CompletionStatus::Pending => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
                CompletionStatus::Complete => break,
                s => panic!("legacy completion {s:?}"),
            }
        }
        let exported = frame.exports.texture(graph.target_export).unwrap();
        assert_eq!(exported.outgoing_state, ResourceAccessState::CopySource);
        assert_eq!(
            frame
                .exports
                .buffer(graph.position_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            frame
                .exports
                .buffer(graph.index_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        let readback = readback_exported_raster_texture_for_test(device, exported).unwrap();
        assert_eq!(readback.bytes_per_row, 256);
        assert!(
            readback
                .padded
                .chunks_exact(256)
                .all(|row| row[32..].iter().all(|byte| *byte == 0))
        );
        assert!(fluxel_rhi::test_support::validation_diagnostics(device).is_empty());
        readback.tight
    }

    #[cfg(windows)]
    fn oracle() -> Vec<u8> {
        let mut output = vec![0; 8 * 8 * 4];
        for pixel in output.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0, 0, 0, 255]);
        }
        let points = [(2.0, 6.0), (6.0, 6.0), (4.0, 2.0)];
        let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
            (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
        };
        for y in 0..8 {
            for x in 0..8 {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let edges = [
                    edge(points[0], points[1], p),
                    edge(points[1], points[2], p),
                    edge(points[2], points[0], p),
                ];
                if edges.iter().all(|e| *e <= 0.0) || edges.iter().all(|e| *e >= 0.0) {
                    output[(y * 8 + x) * 4..(y * 8 + x + 1) * 4]
                        .copy_from_slice(&[48, 176, 112, 255]);
                }
            }
        }
        output
    }

    #[cfg(windows)]
    fn artifact(
        backend: Backend,
        device: &Device,
        snapshot: &IndexedMeshSnapshot,
        graph: &FixedGraph,
        expected: &[u8],
        actual: &[u8],
        mode: &str,
    ) {
        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "unrecorded".into());
        assert!(
            commit == "unrecorded"
                || (commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()))
        );
        let geometry = geometry();
        let imported_states = [
            snapshot.positions().outgoing_state(),
            snapshot.indices().outgoing_state(),
        ];
        println!(
            "artifact schema=fluxel-u02-v1; case=U02; mode={mode}; commit={commit}; os={}; backend={backend:?}; hardware={:?}; driver={}; execution_plan={:?}; raster_identity={:?}; geometry_positions={:?}; geometry_indices={:?}; expected={expected:?}; actual={actual:?}; first_difference={:?}; imported_states={imported_states:?}; final_states=[CopyDestination,CopyDestination]; target_outgoing=CopySource; completion=Complete; diagnostics={:?}",
            std::env::consts::OS,
            device.hardware(),
            device.hardware().driver,
            graph.compiled.execution_plan(),
            RasterKernel::IndexedPositionFloat32x3.portable_identity(),
            geometry.positions(),
            geometry.indices(),
            first_difference(actual, expected),
            fluxel_rhi::test_support::validation_diagnostics(device)
        );
    }

    fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
        actual.iter().zip(expected).position(|(a, b)| a != b)
    }
}

#[cfg(test)]
mod u03 {
    use super::*;
    use crate::{BasicMaterial, Camera, Geometry, IndexedMeshUpload, IndexedMeshUploadStatus};
    use fluxel_rhi::{
        Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test,
    };
    use std::{
        thread,
        time::{Duration, Instant},
    };

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn u03_camera_material_a_dx12() {
        run(Backend::Dx12, CameraCase::MatrixOrder);
    }
    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn u03_camera_material_a_vulkan() {
        run(Backend::Vulkan, CameraCase::MatrixOrder);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn u03_camera_material_b_dx12() {
        run(Backend::Dx12, CameraCase::MaterialColor);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn u03_camera_material_b_vulkan() {
        run(Backend::Vulkan, CameraCase::MaterialColor);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires DX12 and Vulkan devices with required validation"]
    fn u03_camera_material_paired_same_compiled_graph() {
        let _guard = native_fixture_guard();
        let case = CameraCase::MatrixOrder;
        let dx12 = open(Backend::Dx12);
        let vulkan = open(Backend::Vulkan);
        let geometry = geometry();
        let dx_snapshot = ready_snapshot(&dx12, &geometry);
        let vk_snapshot = ready_snapshot(&vulkan, &geometry);
        // The Arc makes this the exact same compiled graph allocation, not a
        // backend-specific recompilation with equivalent contents.
        let graph = Arc::new(build_camera_graph(&dx_snapshot, [8, 8]).unwrap());
        fluxel_rhi::test_support::clear_validation_diagnostics(&dx12);
        let mut dx = start(&dx12, &dx_snapshot, Arc::clone(&graph), case);
        let expected = oracle(case);
        let dx_actual = complete(&dx12, &mut dx);
        assert_eq!(
            dx_actual,
            expected,
            "DX12 first difference: {:?}",
            first_difference(&dx_actual, &expected)
        );
        artifact(
            Backend::Dx12,
            &dx12,
            &graph,
            case,
            &expected,
            &dx_actual,
            "paired",
        );
        fluxel_rhi::test_support::clear_validation_diagnostics(&vulkan);
        let mut vk = start(&vulkan, &vk_snapshot, Arc::clone(&graph), case);
        let vk_actual = complete(&vulkan, &mut vk);
        assert_eq!(
            vk_actual,
            expected,
            "Vulkan first difference: {:?}",
            first_difference(&vk_actual, &expected)
        );
        artifact(
            Backend::Vulkan,
            &vulkan,
            &graph,
            case,
            &expected,
            &vk_actual,
            "paired",
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with test-support fault injection"]
    fn two_stage_gate_busy_and_drop_contract() {
        let _guard = native_fixture_guard();
        let device = open(Backend::Dx12);
        let snapshot = ready_snapshot(&device, &geometry());
        let renderer = FixedFrameRenderer::new(device);
        let camera = camera(CameraCase::MatrixOrder);
        let material = material(CameraCase::MatrixOrder);

        // Invalid input is rejected before it can reserve this generation.
        assert!(matches!(
            renderer.draw(&snapshot, &camera, &material, [0, 8]),
            Err(DrawStartError::InvalidExtent)
        ));

        // Rejection of the uniform upload is pre-raster, so retry stays valid.
        fluxel_rhi::test_support::inject_submit_rejected_once();
        assert!(matches!(
            renderer.draw(&snapshot, &camera, &material, [8, 8]),
            Err(DrawStartError::UniformStart(_))
        ));
        let uploading = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(uploading);
        let retry = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(retry);

        // An accepted-unknown uniform upload never reaches Raster and must
        // release, rather than poison, the snapshot reservation.
        fluxel_rhi::test_support::inject_submit_accepted_unknown_once();
        let mut unknown_upload = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match unknown_upload.poll() {
                FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                    assert!(Instant::now() < deadline)
                }
                FixedFrameStatus::Failed(FixedFrameFailure::UniformCompletion(_)) => break,
                other => panic!("expected unknown uniform failure, got {other:?}"),
            }
            thread::sleep(Duration::from_millis(1));
        }
        drop(unknown_upload);
        let retry = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(retry);

        // A rejected delayed Raster submit is still pre-accept and likewise
        // leaves the generation reusable.
        let mut rejected_raster = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        fluxel_rhi::test_support::inject_submit_rejected_once();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match rejected_raster.poll() {
                FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                    assert!(Instant::now() < deadline)
                }
                FixedFrameStatus::Failed(FixedFrameFailure::RasterStart(_)) => break,
                other => panic!("expected rejected Raster start, got {other:?}"),
            }
            thread::sleep(Duration::from_millis(1));
        }
        drop(rejected_raster);
        let retry = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(retry);

        // While the shared executor is held, a completed upload cannot begin
        // raster work; its ready uniform and reservation survive Busy.
        let mut busy = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        let guard = renderer.executor.try_backend().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match busy.poll() {
                FixedFrameStatus::Pending => assert!(Instant::now() < deadline),
                FixedFrameStatus::Busy => break,
                other => panic!("expected raster-ready Busy, got {other:?}"),
            }
            thread::sleep(Duration::from_millis(1));
        }
        drop(guard);
        complete(&renderer.device, &mut busy);
        let reusable = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(reusable);

        // Dropping in RasterReady (upload complete, Raster not accepted) also
        // releases the reservation.
        let mut ready_drop = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        let guard = renderer.executor.try_backend().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match ready_drop.poll() {
                FixedFrameStatus::Pending => assert!(Instant::now() < deadline),
                FixedFrameStatus::Busy => break,
                other => panic!("expected RasterReady Busy, got {other:?}"),
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(ready_drop.phase, CameraPhase::RasterReady(_)));
        drop(ready_drop);
        drop(guard);
        let retry = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        drop(retry);

        // Busy while observing an accepted Raster frame retains that exact
        // frame and reservation for a later successful poll.
        let mut accepted_busy = renderer
            .draw(&snapshot, &camera, &material, [8, 8])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !matches!(accepted_busy.phase, CameraPhase::RasterAccepted(_)) {
            assert!(Instant::now() < deadline);
            assert!(matches!(
                accepted_busy.poll(),
                FixedFrameStatus::Pending | FixedFrameStatus::Busy
            ));
            thread::sleep(Duration::from_millis(1));
        }
        let guard = renderer.executor.try_backend().unwrap();
        assert!(matches!(accepted_busy.poll(), FixedFrameStatus::Busy));
        assert!(matches!(
            accepted_busy.phase,
            CameraPhase::RasterAccepted(_)
        ));
        drop(guard);
        complete(&renderer.device, &mut accepted_busy);

        // Accepted-unknown Raster work reaches the accepted failure path and
        // poisons the snapshot generation.
        let failed = ready_snapshot(&renderer.device, &geometry());
        let mut accepted_failure = renderer.draw(&failed, &camera, &material, [8, 8]).unwrap();
        fluxel_rhi::test_support::inject_submit_accepted_unknown_once();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match accepted_failure.poll() {
                FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                    assert!(Instant::now() < deadline)
                }
                FixedFrameStatus::Failed(FixedFrameFailure::RasterCompletion(_)) => break,
                other => panic!("expected accepted Raster failure, got {other:?}"),
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(
            renderer.draw(&failed, &camera, &material, [8, 8]),
            Err(DrawStartError::SnapshotPoisoned)
        ));

        // Once the raster submit was accepted, early Drop conservatively poisons
        // the snapshot rather than allowing an unproven reuse.
        let poisoned = ready_snapshot(&renderer.device, &geometry());
        let mut accepted = renderer
            .draw(&poisoned, &camera, &material, [8, 8])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !matches!(accepted.phase, CameraPhase::RasterAccepted(_)) {
            assert!(Instant::now() < deadline);
            assert!(matches!(
                accepted.poll(),
                FixedFrameStatus::Pending | FixedFrameStatus::Busy
            ));
            thread::sleep(Duration::from_millis(1));
        }
        drop(accepted);
        assert!(matches!(
            renderer.draw(&poisoned, &camera, &material, [8, 8]),
            Err(DrawStartError::SnapshotPoisoned)
        ));
    }

    #[derive(Clone, Copy, Debug)]
    enum CameraCase {
        MatrixOrder,
        MaterialColor,
    }

    #[cfg(windows)]
    fn open(backend: Backend) -> Device {
        Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap()
    }

    #[cfg(windows)]
    fn run(backend: Backend, case: CameraCase) {
        let _guard = native_fixture_guard();
        let device = open(backend);
        let snapshot = ready_snapshot(&device, &geometry());
        fluxel_rhi::test_support::clear_validation_diagnostics(&device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let mut submission = renderer
            .draw(&snapshot, &camera(case), &material(case), [8, 8])
            .unwrap();
        let actual = complete(&device, &mut submission);
        let expected = oracle(case);
        assert_eq!(
            actual,
            expected,
            "U03 {backend:?} {case:?} first difference: {:?}",
            first_difference(&actual, &expected)
        );
        artifact(
            backend,
            &device,
            submission.graph.as_ref().unwrap(),
            case,
            &expected,
            &actual,
            "independent-public",
        );
    }

    #[cfg(windows)]
    fn geometry() -> Geometry {
        Geometry::from_positions(vec![[-0.9, -0.7, 0.0], [0.3, -0.7, 0.0], [-0.3, 0.7, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap()
    }

    #[cfg(windows)]
    fn ready_snapshot(device: &Device, geometry: &Geometry) -> IndexedMeshSnapshot {
        let mut upload = IndexedMeshUpload::begin(device, geometry).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match upload.poll() {
                IndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
                IndexedMeshUploadStatus::Pending => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
                IndexedMeshUploadStatus::Failed(e) => panic!("U03 upload {e:?}"),
            }
        }
    }

    #[cfg(windows)]
    fn camera(case: CameraCase) -> Camera {
        let mut view = *Camera::default().view();
        let mut projection = *Camera::default().projection();
        match case {
            // P * V yields x = .5(x + .4), y = .8(y - .1); reversing
            // the multiplication visibly produces different pixel coverage.
            CameraCase::MatrixOrder => {
                view[3][0] = 0.4;
                view[3][1] = -0.1;
                projection[0][0] = 0.5;
                projection[1][1] = 0.8;
            }
            CameraCase::MaterialColor => {
                view[3][0] = 0.125;
                projection[0][0] = 0.75;
                projection[1][1] = 0.75;
            }
        }
        Camera::new(view, projection)
    }

    #[cfg(windows)]
    fn material(case: CameraCase) -> BasicMaterial {
        match case {
            CameraCase::MatrixOrder => {
                BasicMaterial::new([48.0 / 255.0, 176.0 / 255.0, 112.0 / 255.0, 1.0])
            }
            CameraCase::MaterialColor => {
                BasicMaterial::new([17.0 / 255.0, 93.0 / 255.0, 201.0 / 255.0, 1.0])
            }
        }
    }

    #[cfg(windows)]
    fn start(
        device: &Device,
        snapshot: &IndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        case: CameraCase,
    ) -> FixedFrameSubmission {
        let renderer = FixedFrameRenderer::new(device.clone());
        renderer
            .start_camera(
                snapshot,
                graph,
                FrameUniform::new(&camera(case), &material(case)).unwrap(),
            )
            .unwrap()
    }

    #[cfg(windows)]
    fn complete(device: &Device, submission: &mut FixedFrameSubmission) -> Vec<u8> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match submission.poll() {
                FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                    assert!(Instant::now() < deadline);
                    thread::sleep(Duration::from_millis(1));
                }
                FixedFrameStatus::Complete(_) => break,
                FixedFrameStatus::Failed(e) => panic!("U03 failed {e:?}"),
            }
        }
        let frame = submission.completed.as_ref().unwrap();
        let graph = submission.graph.as_ref().unwrap();
        assert_eq!(
            frame
                .exports
                .buffer(graph.position_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            frame
                .exports
                .buffer(graph.index_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        let readback = readback_exported_raster_texture_for_test(
            device,
            frame
                .exports
                .texture(submission.target_export.unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_eq!(readback.bytes_per_row, 256);
        assert!(
            readback
                .padded
                .chunks_exact(256)
                .all(|row| row[32..].iter().all(|b| *b == 0))
        );
        assert!(fluxel_rhi::test_support::validation_diagnostics(device).is_empty());
        readback.tight
    }

    #[cfg(windows)]
    fn oracle(case: CameraCase) -> Vec<u8> {
        let camera = camera(case);
        let view = camera.view();
        let projection = camera.projection();
        let mut m = [[0.0; 4]; 4];
        for (c, column) in m.iter_mut().enumerate() {
            for (r, value) in column.iter_mut().enumerate() {
                // Deliberately independent from FrameUniform serialization:
                // this oracle computes P * V directly from public inputs.
                *value = projection[0][r] * view[c][0]
                    + projection[1][r] * view[c][1]
                    + projection[2][r] * view[c][2]
                    + projection[3][r] * view[c][3];
            }
        }
        let points = geometry()
            .positions()
            .iter()
            .map(|p| {
                let v = [p[0], p[1], p[2], 1.0];
                let mut clip = [0.0; 4];
                for (r, value) in clip.iter_mut().enumerate() {
                    *value = (0..4).map(|c| m[c][r] * v[c]).sum();
                }
                // U03 isolates transform order and material transport; its
                // fixtures intentionally require no geometric clipping.
                assert!(clip[3] > 0.0);
                assert!(clip[0].abs() <= clip[3] && clip[1].abs() <= clip[3]);
                let ndc = [clip[0] / clip[3], clip[1] / clip[3]];
                ((ndc[0] + 1.0) * 4.0, (1.0 - ndc[1]) * 4.0)
            })
            .collect::<Vec<_>>();
        // Inputs are exact n/255 fixtures; use their defining integer oracle
        // instead of duplicating an underspecified float-to-UNORM cast.
        let color = match case {
            CameraCase::MatrixOrder => [48, 176, 112, 255],
            CameraCase::MaterialColor => [17, 93, 201, 255],
        };
        let mut out = vec![0; 8 * 8 * 4];
        for pixel in out.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0, 0, 0, 255]);
        }
        let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
            (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
        };
        // Screen coordinates have a downward Y axis.  Normalize the three
        // edge tests to the triangle winding, then use the API's top-left
        // inclusion rule for exact-on-edge sample points.
        let top_left = |a: (f32, f32), b: (f32, f32)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
        let winding = edge(points[0], points[1], points[2]);
        for y in 0..8 {
            for x in 0..8 {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let edges = [
                    (points[0], points[1]),
                    (points[1], points[2]),
                    (points[2], points[0]),
                ];
                if edges.iter().all(|&(a, b)| {
                    let value = edge(a, b, p);
                    if winding > 0.0 {
                        value > 0.0 || (value == 0.0 && top_left(a, b))
                    } else {
                        value < 0.0 || (value == 0.0 && top_left(b, a))
                    }
                }) {
                    out[(y * 8 + x) * 4..(y * 8 + x + 1) * 4].copy_from_slice(&color);
                }
            }
        }
        out
    }

    #[cfg(windows)]
    fn artifact(
        backend: Backend,
        device: &Device,
        graph: &CameraGraph,
        case: CameraCase,
        expected: &[u8],
        actual: &[u8],
        mode: &str,
    ) {
        let commit = std::env::var("FLUXEL_TEST_COMMIT")
            .expect("U03 evidence requires FLUXEL_TEST_COMMIT at the exact tested SHA");
        assert!(commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()));
        let uniform = FrameUniform::new(&camera(case), &material(case)).unwrap();
        println!(
            "artifact schema=fluxel-u03-v1; case={case:?}; mode={mode}; commit={commit}; os={}; backend={backend:?}; hardware={:?}; driver={}; uniform={:?}; matrix=column-major_projection_times_view; execution_plan={:?}; raster_identity={:?}; binding=group0_binding0_static_80; expected={expected:?}; actual={actual:?}; first_difference={:?}; imported_states=[CopyDestination,CopyDestination,CopyDestination]; final_states=[CopyDestination,CopyDestination]; target_outgoing=CopySource; phase=Terminal; completion=Complete; diagnostics={:?}",
            std::env::consts::OS,
            device.hardware(),
            device.hardware().driver,
            uniform.bytes(),
            graph.compiled.execution_plan(),
            RasterKernel::IndexedPositionFloat32x3CameraMaterial.portable_identity(),
            first_difference(actual, expected),
            fluxel_rhi::test_support::validation_diagnostics(device)
        );
    }

    fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
        actual.iter().zip(expected).position(|(a, b)| a != b)
    }
}
