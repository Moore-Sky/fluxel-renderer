//! The deliberately closed headless indexed-frame coordinator.

use core::fmt;

use fluxel_rendergraph::{
    AttachmentOps, BoundBuffer, BufferBindingId, BufferRange, ColorAttachmentDesc,
    CompletionFailure, CompletionStatus, DeviceIdentity, ExecutionError, ExportBufferContract,
    ExportTextureContract, Extent3d, ExternalOwnership, FrameBindingError, FrameBindingErrorKind,
    FrameInputs, FrameResourceProvider, ImportBufferContract, IndexFormat, InitialContents, LoadOp,
    RasterPipelineId, RenderGraph, ResourceAccessState, StoreOp, TextureDesc, TextureDimension,
    TextureFormat, TextureRange, Viewport, WriteCoverage,
};
use fluxel_rhi::{
    Buffer, Device, RasterBackend, RasterKernel, RasterObjectProvider, ResourceLease, Texture,
};

use crate::upload::{IndexedMeshSnapshot, SnapshotUseError};

fn position_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0001)
}
fn index_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0002)
}
fn fixed_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0210_0001)
}

/// Coordinates the fixed headless `f32x3/u32` indexed draw slice.
///
/// This type owns no application-visible native resource handles.  It accepts
/// only a completed [`IndexedMeshSnapshot`] and produces opaque image metadata
/// after non-blocking completion observation.
pub struct FixedFrameRenderer {
    device: Device,
    executor: fluxel_rendergraph::FrameExecutor<RasterBackend>,
}

impl FixedFrameRenderer {
    /// Creates a renderer over an already-opened headless device.
    #[must_use]
    pub fn new(device: Device) -> Self {
        Self {
            executor: fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone())),
            device,
        }
    }

    /// Starts one fixed indexed draw without waiting for the GPU.
    ///
    /// The snapshot generation is reserved until this returned operation proves
    /// completion.  A graph rejection before native submission releases the
    /// reservation; an accepted but unproven operation poisons it on drop.
    pub fn draw(
        &self,
        snapshot: &IndexedMeshSnapshot,
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
        let graph = build_graph(snapshot, extent)
            .map_err(|error| DrawStartError::Graph(error.to_string()))?;
        self.draw_compiled(snapshot, &graph)
    }

    fn draw_compiled(
        &self,
        snapshot: &IndexedMeshSnapshot,
        graph: &FixedGraph,
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        if snapshot.positions().buffer().device_identity() != self.device.identity()
            || snapshot.indices().buffer().device_identity() != self.device.identity()
        {
            return Err(DrawStartError::ForeignSnapshotDevice);
        }
        let reservation = snapshot.reserve_for_draw().map_err(|error| match error {
            SnapshotUseError::InFlight => DrawStartError::SnapshotInFlight,
            SnapshotUseError::Poisoned => DrawStartError::SnapshotPoisoned,
        })?;
        let pipeline = match self
            .device
            .create_raster_pipeline(RasterKernel::IndexedPositionFloat32x3)
        {
            Ok(pipeline) => pipeline,
            Err(error) => {
                reservation.release_before_submit();
                return Err(DrawStartError::Pipeline(error.to_string()));
            }
        };
        let mut objects = RasterObjectProvider::new(&self.device);
        if let Err(error) = objects.register_raster_pipeline(fixed_pipeline(), pipeline) {
            reservation.release_before_submit();
            return Err(DrawStartError::Pipeline(error.to_string()));
        }
        let resources = SnapshotResources {
            device: self.device.identity(),
            positions: snapshot.positions().buffer().clone(),
            indices: snapshot.indices().buffer().clone(),
            position_state: snapshot.positions().outgoing_state(),
            index_state: snapshot.indices().outgoing_state(),
        };
        let mut inputs = FrameInputs::new(());
        inputs.bind_buffer(graph.position_slot, position_binding());
        inputs.bind_buffer(graph.index_slot, index_binding());
        let execution = graph.compiled.instantiate_local(inputs);
        match self
            .executor
            .execute(&graph.compiled, execution, &resources, &objects)
        {
            Ok(frame) => Ok(FixedFrameSubmission {
                frame: Some(frame),
                reservation: Some(reservation),
                target_export: graph.target_export,
                image: None,
                failure: None,
            }),
            Err(error) => {
                reservation.release_before_submit();
                Err(DrawStartError::Execution(error.to_string()))
            }
        }
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
    /// Graph compilation rejected the fixed declaration.
    Graph(String),
    /// The device could not create the closed raster artifact.
    Pipeline(String),
    /// Recording or submission was rejected before native work was accepted.
    Execution(String),
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
            Self::Graph(error) => write!(formatter, "fixed frame graph rejected: {error}"),
            Self::Pipeline(error) => write!(formatter, "fixed frame pipeline rejected: {error}"),
            Self::Execution(error) => write!(formatter, "fixed frame execution rejected: {error}"),
        }
    }
}

impl std::error::Error for DrawStartError {}

/// A non-blocking accepted fixed frame.
pub struct FixedFrameSubmission {
    frame: Option<fluxel_rendergraph::ExecutedFrame<RasterBackend>>,
    reservation: Option<crate::upload::SnapshotDrawReservation>,
    target_export: fluxel_rendergraph::ExportTextureSlot,
    image: Option<FrameImage>,
    failure: Option<CompletionFailure>,
}

impl FixedFrameSubmission {
    /// Polls GPU completion without waiting.
    pub fn poll(&mut self) -> FixedFrameStatus {
        if let Some(image) = &self.image {
            return FixedFrameStatus::Complete(image.clone());
        }
        if let Some(failure) = &self.failure {
            return FixedFrameStatus::Failed(*failure);
        }
        let Some(frame) = self.frame.as_mut() else {
            unreachable!("a non-terminal fixed frame retains its accepted submission")
        };
        match frame.submission.status() {
            Err(ExecutionError::ExecutorBusy) => FixedFrameStatus::Busy,
            Err(_) => self.finish_failed(CompletionFailure::DeviceLost),
            Ok(CompletionStatus::Pending) => FixedFrameStatus::Pending,
            Ok(CompletionStatus::Complete) => {
                let exported = frame
                    .exports
                    .texture(self.target_export)
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
                FixedFrameStatus::Complete(image)
            }
            Ok(CompletionStatus::Failed(failure)) => self.finish_failed(failure),
            Ok(_) => self.finish_failed(CompletionFailure::DeviceLost),
        }
    }

    fn finish_failed(&mut self, failure: CompletionFailure) -> FixedFrameStatus {
        self.frame.take();
        if let Some(mut reservation) = self.reservation.take() {
            reservation.poison();
        }
        self.failure = Some(failure);
        FixedFrameStatus::Failed(failure)
    }
}

impl Drop for FixedFrameSubmission {
    fn drop(&mut self) {
        if let Some(mut reservation) = self.reservation.take() {
            reservation.poison();
        }
    }
}

/// The observed state of a [`FixedFrameSubmission`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FixedFrameStatus {
    /// The GPU has not completed the accepted submission.
    Pending,
    /// The executor is momentarily held by another safe operation; retry poll.
    Busy,
    /// The fixed frame completed and its opaque output is available.
    Complete(FrameImage),
    /// The GPU did not establish the promised terminal state.
    Failed(CompletionFailure),
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

struct SnapshotResources {
    device: DeviceIdentity,
    positions: Buffer,
    indices: Buffer,
    position_state: ResourceAccessState,
    index_state: ResourceAccessState,
}

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
mod tests {
    use super::*;
    use std::{
        sync::{Mutex, OnceLock},
        thread,
        time::{Duration, Instant},
    };

    use crate::{Geometry, IndexedMeshUpload, IndexedMeshUploadStatus};
    use fluxel_rhi::{
        Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test,
    };

    #[cfg(windows)]
    static HARDWARE_SERIAL: OnceLock<Mutex<()>> = OnceLock::new();

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn u02_fixed_frame_dx12() {
        run_u02(Backend::Dx12);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn u02_fixed_frame_vulkan() {
        run_u02(Backend::Vulkan);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with test-support fault injection"]
    fn fixed_frame_public_operation_state_contract() {
        let _guard = HARDWARE_SERIAL
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("hardware fixture mutex must not be poisoned");
        let device = Device::open(
            Backend::Dx12,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .expect("state contract device");
        let renderer = FixedFrameRenderer::new(device.clone());
        let deadline = Instant::now() + Duration::from_secs(10);

        let retryable = ready_snapshot(&device);
        fluxel_rhi::test_support::inject_submit_rejected_once();
        assert!(matches!(
            renderer.draw(&retryable, [8, 8]),
            Err(DrawStartError::Execution(_))
        ));
        let mut normal = renderer.draw(&retryable, [8, 8]).expect("pre-submit retry");
        assert!(
            normal
                .frame
                .as_ref()
                .expect("accepted operation retained")
                .submission
                .retained_lease_count()
                > 0
        );
        let backend_guard = renderer.executor.try_backend().expect("backend guard");
        assert!(matches!(normal.poll(), FixedFrameStatus::Busy));
        drop(backend_guard);
        fluxel_rhi::test_support::inject_completion_pending_once();
        assert!(matches!(normal.poll(), FixedFrameStatus::Pending));
        poll_complete(&mut normal, deadline);
        let mut retry_after_complete = renderer
            .draw(&retryable, [8, 8])
            .expect("complete releases gate");
        poll_complete(&mut retry_after_complete, deadline);

        let uncertain = ready_snapshot(&device);
        fluxel_rhi::test_support::inject_submit_accepted_unknown_once();
        let mut unknown = renderer
            .draw(&uncertain, [8, 8])
            .expect("unknown submit accepted");
        assert!(matches!(unknown.poll(), FixedFrameStatus::Failed(_)));
        assert!(matches!(
            renderer.draw(&uncertain, [8, 8]),
            Err(DrawStartError::SnapshotPoisoned)
        ));

        let dropped = ready_snapshot(&device);
        let early = renderer
            .draw(&dropped, [8, 8])
            .expect("accepted early-drop frame");
        drop(early);
        assert!(matches!(
            renderer.draw(&dropped, [8, 8]),
            Err(DrawStartError::SnapshotPoisoned)
        ));
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires DX12 and Vulkan devices with required validation"]
    fn u02_paired_same_compiled_graph() {
        let _guard = HARDWARE_SERIAL
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let open = |backend| {
            Device::open(
                backend,
                DeviceOptions {
                    validation: Validation::Required,
                    ..DeviceOptions::default()
                },
            )
            .unwrap()
        };
        let dx12 = open(Backend::Dx12);
        let vulkan = open(Backend::Vulkan);
        fluxel_rhi::test_support::clear_validation_diagnostics(&dx12);
        fluxel_rhi::test_support::clear_validation_diagnostics(&vulkan);
        let geometry = fixed_geometry();
        let dx_snapshot = ready_snapshot_from(&dx12, &geometry);
        let vk_snapshot = ready_snapshot_from(&vulkan, &geometry);
        let dx_imported_states = uploaded_incoming_states(&dx_snapshot);
        let vk_imported_states = uploaded_incoming_states(&vk_snapshot);
        // Both submissions borrow this exact CompiledGraph allocation.
        let graph = build_graph(&dx_snapshot, [8, 8]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut dx_submission = FixedFrameRenderer::new(dx12.clone())
            .draw_compiled(&dx_snapshot, &graph)
            .unwrap();
        let mut vk_submission = FixedFrameRenderer::new(vulkan.clone())
            .draw_compiled(&vk_snapshot, &graph)
            .unwrap();
        let expected = fixed_oracle();
        let dx_actual = completed_tight(&dx12, &mut dx_submission, deadline);
        let vk_actual = completed_tight(&vulkan, &mut vk_submission, deadline);
        assert_eq!(dx_actual, expected);
        assert_eq!(vk_actual, expected);
        let dx_diagnostics = fluxel_rhi::test_support::validation_diagnostics(&dx12);
        let vk_diagnostics = fluxel_rhi::test_support::validation_diagnostics(&vulkan);
        assert!(dx_diagnostics.is_empty());
        assert!(vk_diagnostics.is_empty());
        emit_u02_artifact(
            Backend::Dx12,
            &dx12,
            &graph,
            &geometry,
            &expected,
            &dx_actual,
            &dx_diagnostics,
            "paired",
            dx_imported_states,
        );
        emit_u02_artifact(
            Backend::Vulkan,
            &vulkan,
            &graph,
            &geometry,
            &expected,
            &vk_actual,
            &vk_diagnostics,
            "paired",
            vk_imported_states,
        );
    }

    #[cfg(windows)]
    fn run_u02(backend: Backend) {
        let _guard = HARDWARE_SERIAL
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let device = Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("U02 {backend:?} device open failed: {error}"));
        fluxel_rhi::test_support::clear_validation_diagnostics(&device);

        let geometry =
            Geometry::from_positions(vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
                .with_indices(vec![0, 1, 2])
                .unwrap();
        let mut upload = IndexedMeshUpload::begin(&device, &geometry).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let snapshot = loop {
            match upload.poll() {
                IndexedMeshUploadStatus::Ready => break upload.ready_snapshot().unwrap(),
                IndexedMeshUploadStatus::Pending => {
                    assert!(Instant::now() < deadline, "U02 upload timed out");
                    thread::sleep(Duration::from_millis(1));
                }
                IndexedMeshUploadStatus::Failed(error) => panic!("U02 upload failed: {error:?}"),
            }
        };
        let imported_states = uploaded_incoming_states(&snapshot);
        let renderer = FixedFrameRenderer::new(device.clone());
        let graph = build_graph(&snapshot, [8, 8]).unwrap();
        let mut submission = renderer.draw_compiled(&snapshot, &graph).unwrap();
        let image = poll_complete(&mut submission, deadline);
        assert_eq!(image.extent(), [8, 8]);
        assert_eq!(image.format(), TextureFormat::Rgba8Unorm);
        let exported = submission
            .frame
            .as_ref()
            .expect("U02 completed frame retained")
            .exports
            .texture(submission.target_export)
            .expect("U02 target export exists");
        assert_eq!(exported.outgoing_state, ResourceAccessState::CopySource);
        let readback = readback_exported_raster_texture_for_test(&device, exported).unwrap();
        assert_eq!(readback.bytes_per_row, 256);
        assert!(
            readback
                .padded
                .chunks_exact(256)
                .all(|row| row[32..].iter().all(|byte| *byte == 0)),
            "U02 {backend:?} staging padding was not zero"
        );
        let actual = readback.tight;
        let expected = fixed_oracle();
        assert_eq!(
            actual,
            expected,
            "U02 {backend:?} first difference: {:?}",
            first_difference(&actual, &expected)
        );
        assert_eq!(
            submission
                .frame
                .as_ref()
                .expect("completed public submission retains exports")
                .exports
                .buffer(graph.position_export)
                .expect("positions export")
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            submission
                .frame
                .as_ref()
                .expect("completed public submission retains exports")
                .exports
                .buffer(graph.index_export)
                .expect("indices export")
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        let diagnostics = fluxel_rhi::test_support::validation_diagnostics(&device);
        assert!(
            diagnostics.is_empty(),
            "U02 {backend:?} diagnostics: {diagnostics:?}"
        );
        emit_u02_artifact(
            backend,
            &device,
            &graph,
            &geometry,
            &expected,
            &actual,
            &diagnostics,
            "independent",
            imported_states,
        );
    }

    #[cfg(windows)]
    fn ready_snapshot(device: &Device) -> IndexedMeshSnapshot {
        ready_snapshot_from(device, &fixed_geometry())
    }

    #[cfg(windows)]
    fn fixed_geometry() -> Geometry {
        Geometry::from_positions(vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap()
    }

    #[cfg(windows)]
    fn ready_snapshot_from(device: &Device, geometry: &Geometry) -> IndexedMeshSnapshot {
        let mut upload = IndexedMeshUpload::begin(device, geometry).expect("state upload starts");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match upload.poll() {
                IndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
                IndexedMeshUploadStatus::Pending => {
                    assert!(Instant::now() < deadline, "state upload timed out");
                    thread::sleep(Duration::from_millis(1));
                }
                IndexedMeshUploadStatus::Failed(error) => panic!("state upload failed: {error:?}"),
            }
        }
    }

    #[cfg(windows)]
    fn uploaded_incoming_states(snapshot: &IndexedMeshSnapshot) -> [ResourceAccessState; 2] {
        let states = [
            snapshot.positions().outgoing_state(),
            snapshot.indices().outgoing_state(),
        ];
        assert_eq!(states, [ResourceAccessState::CopyDestination; 2]);
        states
    }

    #[cfg(windows)]
    fn poll_complete(submission: &mut FixedFrameSubmission, deadline: Instant) -> FrameImage {
        loop {
            match submission.poll() {
                FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                    assert!(Instant::now() < deadline, "U02 frame timed out");
                    thread::sleep(Duration::from_millis(1));
                }
                FixedFrameStatus::Complete(image) => return image,
                FixedFrameStatus::Failed(error) => panic!("U02 frame failed: {error:?}"),
            }
        }
    }

    #[cfg(windows)]
    fn completed_tight(
        device: &Device,
        submission: &mut FixedFrameSubmission,
        deadline: Instant,
    ) -> Vec<u8> {
        poll_complete(submission, deadline);
        let exported = submission
            .frame
            .as_ref()
            .expect("completed frame retains export")
            .exports
            .texture(submission.target_export)
            .expect("target export");
        let readback = readback_exported_raster_texture_for_test(device, exported).unwrap();
        assert_eq!(readback.bytes_per_row, 256);
        assert!(
            readback
                .padded
                .chunks_exact(256)
                .all(|row| row[32..].iter().all(|byte| *byte == 0))
        );
        readback.tight
    }

    #[cfg(windows)]
    #[allow(
        clippy::too_many_arguments,
        reason = "test artifact fields are intentionally explicit"
    )]
    fn emit_u02_artifact(
        backend: Backend,
        device: &Device,
        graph: &FixedGraph,
        geometry: &Geometry,
        expected: &[u8],
        actual: &[u8],
        diagnostics: &[String],
        mode: &str,
        imported_states: [ResourceAccessState; 2],
    ) {
        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "unrecorded".into());
        assert!(
            commit == "unrecorded"
                || (commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()))
        );
        println!(
            "artifact schema=fluxel-u02-v1; case=U02; mode={mode}; commit={commit}; os={}; backend={backend:?}; hardware={:?}; driver={}; execution_plan={:?}; raster_artifact={:?}; geometry_positions={:?}; geometry_indices={:?}; expected={expected:?}; actual={actual:?}; first_difference={:?}; imported_states={imported_states:?}; final_states=[CopyDestination,CopyDestination]; target_outgoing=CopySource; completion=Complete; diagnostics={diagnostics:?}",
            std::env::consts::OS,
            device.hardware(),
            device.hardware().driver,
            graph.compiled.execution_plan(),
            RasterKernel::IndexedPositionFloat32x3.portable_identity(),
            geometry.positions(),
            geometry.indices(),
            first_difference(actual, expected),
        );
    }

    fn fixed_oracle() -> Vec<u8> {
        let mut output = vec![0_u8; 8 * 8 * 4];
        for pixel in output.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0, 0, 0, 255]);
        }
        let points = [(2.0_f32, 6.0_f32), (6.0, 6.0), (4.0, 2.0)];
        let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
            (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
        };
        for y in 0..8 {
            for x in 0..8 {
                let point = (x as f32 + 0.5, y as f32 + 0.5);
                let edges = [
                    edge(points[0], points[1], point),
                    edge(points[1], points[2], point),
                    edge(points[2], points[0], point),
                ];
                if edges.iter().all(|edge| *edge <= 0.0) || edges.iter().all(|edge| *edge >= 0.0) {
                    output[(y * 8 + x) * 4..(y * 8 + x + 1) * 4]
                        .copy_from_slice(&[48, 176, 112, 255]);
                }
            }
        }
        output
    }

    fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
        actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected)
    }
}
