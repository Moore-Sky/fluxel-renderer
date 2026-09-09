//! Copy-only native execution of portable RenderGraph plans.

use core::{fmt, time::Duration};
use std::ops::Range;

use fluxel_rendergraph::{
    BoundBuffer, BoundTexture, BufferCapabilities, BufferCopyRegion, BufferDesc, BufferRange,
    BufferUsage, CompletionFailure, CompletionStatus, DeviceCapabilities, DeviceLimits,
    ExecutionBackend, IndexFormat, QueueCapabilities, QueueDescriptor, QueueId,
    RasterPassDescriptor, RecordingCapabilities, RecordingModel, ResourceAccessState, ScissorRect,
    SynchronizationCapabilities, TextureCopyRegion, TextureDesc, TextureFormat,
    TextureFormatCapabilities, TextureRange, TextureUsage, TimestampCapabilities,
    TransientResourceCapabilities, TransitionCapabilities, Viewport,
};

use crate::{
    Buffer, BufferDescriptor, Device, MemoryPolicy, ResourceCreateError, ResourceLease, Texture,
    TextureDescriptor,
};

/// Failure while lowering or executing the 0.1.2 copy-only native slice.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeExecutionError {
    /// An owned transient could not be created.
    Resource(ResourceCreateError),
    /// The plan selected a command family outside the copy-only slice.
    UnsupportedCommandFamily,
    /// A logical queue other than the backend's only queue was requested.
    UnknownQueue(QueueId),
    /// A command referenced a resource owned by another device.
    ForeignResource,
    /// An encoder was used through a backend for another device.
    ForeignEncoder,
    /// A finished command buffer was submitted through another device.
    ForeignCommandBuffer,
    /// A copy range or texture region was invalid at the native boundary.
    InvalidTransfer(&'static str),
    /// A native recording operation failed before submission.
    Recording(String),
    /// A finished command buffer was rejected without being accepted.
    SubmitRejected(String),
    /// A completion query failed after submission was accepted.
    Completion(String),
}

impl fmt::Display for NativeExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(formatter, "resource creation failed: {error}"),
            Self::UnsupportedCommandFamily => {
                formatter.write_str("command family is not supported by the copy-only backend")
            }
            Self::UnknownQueue(queue) => write!(formatter, "unknown logical queue {queue:?}"),
            Self::ForeignResource => {
                formatter.write_str("resource belongs to another native device")
            }
            Self::ForeignEncoder => formatter.write_str("encoder belongs to another native device"),
            Self::ForeignCommandBuffer => {
                formatter.write_str("command buffer belongs to another native device")
            }
            Self::InvalidTransfer(reason) => write!(formatter, "invalid native transfer: {reason}"),
            Self::Recording(reason) => write!(formatter, "native recording failed: {reason}"),
            Self::SubmitRejected(reason) => {
                write!(formatter, "native submission was rejected: {reason}")
            }
            Self::Completion(reason) => {
                write!(formatter, "native completion query failed: {reason}")
            }
        }
    }
}

impl std::error::Error for NativeExecutionError {}

impl From<ResourceCreateError> for NativeExecutionError {
    fn from(value: ResourceCreateError) -> Self {
        Self::Resource(value)
    }
}

/// Outcome of a bounded CPU wait used outside graph execution.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WaitError {
    /// The accepted submission remained pending at the deadline.
    Timeout,
    /// The accepted submission reached a terminal failure.
    Failed(CompletionFailure),
    /// The native completion query itself failed.
    Backend(NativeExecutionError),
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for WaitError {}

/// Opaque recording encoder for one copy-only graph submission.
pub struct CopyEncoder {
    native: crate::imp::CopyEncoder,
    device: fluxel_rendergraph::DeviceIdentity,
    leases: Vec<ResourceLease>,
}
/// Opaque finished command buffer for one copy-only graph submission.
pub struct CopyCommandBuffer {
    native: crate::imp::CopyCommandBuffer,
    device: fluxel_rendergraph::DeviceIdentity,
    leases: Vec<ResourceLease>,
}
/// Opaque cloneable completion for an accepted native submission.
#[derive(Clone)]
pub struct NativeCompletion(pub(crate) crate::imp::NativeCompletion);

/// Uninhabited raster-pipeline placeholder for the copy-only backend.
pub enum UnsupportedRasterPipeline {}
/// Uninhabited compute-pipeline placeholder for the copy-only backend.
pub enum UnsupportedComputePipeline {}
/// Uninhabited binding placeholder for the copy-only backend.
pub enum UnsupportedBindings {}

struct Retired {
    completion: NativeCompletion,
    leases: Vec<ResourceLease>,
}

/// A serial DX12/Vulkan backend that executes only RenderGraph Copy plans.
pub struct CopyBackend {
    device: Device,
    capabilities: DeviceCapabilities,
    retired: Vec<Retired>,
}

impl CopyBackend {
    /// Creates a copy-only backend over an already opened native device.
    pub fn new(device: Device) -> Self {
        Self {
            capabilities: Self::portable_capabilities(),
            device,
            retired: Vec::new(),
        }
    }

    /// Returns the normalized capability profile shared by both native backends.
    pub fn portable_capabilities() -> DeviceCapabilities {
        copy_capabilities()
    }

    /// Waits outside graph execution for an accepted submission.
    pub fn wait(&self, completion: &NativeCompletion, timeout: Duration) -> Result<(), WaitError> {
        match crate::imp::wait_completion(&completion.0, timeout)
            .map_err(|error| WaitError::Backend(NativeExecutionError::Completion(error)))?
        {
            CompletionStatus::Complete => Ok(()),
            CompletionStatus::Pending => Err(WaitError::Timeout),
            CompletionStatus::Failed(reason) => Err(WaitError::Failed(reason)),
            _ => Err(WaitError::Backend(NativeExecutionError::Completion(
                "unknown completion status".into(),
            ))),
        }
    }
}

#[cfg(all(test, windows, feature = "dx12", feature = "vulkan"))]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use fluxel_rendergraph::{
        BindingSetId, BoundBindings, BoundComputePipeline, BoundRasterPipeline, BufferBindingId,
        BufferRead, BufferWrite, ComputePipelineId, DiagnosticContext, ExportBufferContract,
        ExportBufferSlot, ExternalOwnership, FrameBindingError, FrameBindingErrorKind, FrameInputs,
        FrameResourceProvider, ImportBufferContract, InitialContents, PassResourceResolver,
        RasterPipelineId, RecordResult, RecordingError, RecordingErrorKind, RenderGraph,
        RenderObjectProvider, ResolvedBindingResource, TextureBindingId, TextureRead, TextureWrite,
        WriteCoverage,
    };

    struct Resources {
        device: fluxel_rendergraph::DeviceIdentity,
        buffers: HashMap<BufferBindingId, (Buffer, ResourceAccessState)>,
        textures: HashMap<TextureBindingId, (Texture, ResourceAccessState)>,
    }

    impl FrameResourceProvider<CopyBackend> for Resources {
        fn texture(
            &self,
            id: fluxel_rendergraph::TextureBindingId,
        ) -> Result<BoundTexture<Texture, ResourceLease>, FrameBindingError> {
            let (texture, state) = self.textures.get(&id).ok_or_else(|| {
                missing(
                    FrameBindingErrorKind::MissingTexture,
                    "texture not registered",
                )
            })?;
            Ok(BoundTexture {
                device: self.device,
                identity: texture.identity(),
                physical: texture.clone(),
                descriptor: texture.descriptor().texture,
                usage: texture.allowed_usage(),
                initial_state: *state,
                lease: texture.lease().into(),
            })
        }
        fn buffer(
            &self,
            id: BufferBindingId,
        ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
            let (buffer, state) = self.buffers.get(&id).ok_or_else(|| {
                missing(
                    FrameBindingErrorKind::MissingBuffer,
                    "buffer not registered",
                )
            })?;
            Ok(BoundBuffer {
                device: self.device,
                identity: buffer.identity(),
                physical: buffer.clone(),
                descriptor: buffer.descriptor().buffer,
                usage: buffer.allowed_usage(),
                initial_state: *state,
                lease: buffer.lease().into(),
            })
        }
    }

    fn missing(kind: FrameBindingErrorKind, detail: &str) -> FrameBindingError {
        FrameBindingError {
            kind,
            texture_slot: None,
            buffer_slot: None,
            resource: None,
            surface_binding: None,
            detail: detail.into(),
        }
    }

    struct NoObjects;
    impl RenderObjectProvider<CopyBackend> for NoObjects {
        fn raster_pipeline(
            &self,
            _: RasterPipelineId,
        ) -> Result<BoundRasterPipeline<UnsupportedRasterPipeline, ResourceLease>, RecordingError>
        {
            Err(no_object())
        }
        fn compute_pipeline(
            &self,
            _: ComputePipelineId,
        ) -> Result<BoundComputePipeline<UnsupportedComputePipeline, ResourceLease>, RecordingError>
        {
            Err(no_object())
        }
        fn bindings(
            &self,
            _: BindingSetId,
            _: &[ResolvedBindingResource<'_, Texture, Buffer>],
            _: &[u32],
        ) -> Result<BoundBindings<UnsupportedBindings, ResourceLease>, RecordingError> {
            Err(no_object())
        }
    }
    fn no_object() -> RecordingError {
        RecordingError {
            kind: RecordingErrorKind::MissingFrameBinding,
            context: DiagnosticContext {
                passes: vec![],
                resource: None,
                texture_slot: None,
                buffer_slot: None,
                detail: "copy fixture has no render objects".into(),
                capabilities: None,
            },
        }
    }

    struct CopyData {
        source: BufferRead,
        destination: BufferWrite,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    }

    struct TextureCopyData {
        source: TextureRead,
        destination: TextureWrite,
        region: TextureCopyRegion,
    }

    fn add_copy(
        graph: &mut RenderGraph,
        name: &str,
        source: &fluxel_rendergraph::BufferVersion,
        destination: fluxel_rendergraph::BufferVersion,
        source_offset: u64,
        destination_offset: u64,
        size: u64,
    ) -> fluxel_rendergraph::BufferVersion {
        graph
            .add_copy_pass(
                name,
                |pass| {
                    let src = pass.read_buffer(
                        source,
                        BufferRange::Bytes {
                            offset: source_offset,
                            size,
                        },
                    );
                    let (out, dst) = pass.write_buffer(
                        destination,
                        BufferRange::Bytes {
                            offset: destination_offset,
                            size,
                        },
                        WriteCoverage::Full,
                    );
                    (
                        out,
                        CopyData {
                            source: src,
                            destination: dst,
                            source_offset,
                            destination_offset,
                            size,
                        },
                    )
                },
                |commands,
                 _resolver: &mut PassResourceResolver<'_>,
                 data,
                 _frame|
                 -> RecordResult {
                    commands.copy_buffer(
                        &data.source,
                        &data.destination,
                        BufferCopyRegion {
                            source_offset: data.source_offset,
                            destination_offset: data.destination_offset,
                            size: data.size,
                        },
                    )
                },
            )
            .output
    }

    macro_rules! native_case {
        ($name:ident, $runner:ident, $backend:expr) => {
            #[test]
            #[ignore = "requires native validation plus a real GPU"]
            fn $name() {
                let _guard = hardware_guard();
                crate::imp::initialize_validation_capture();
                $runner($backend);
            }
        };
    }

    native_case!(c01_dx12_partial_buffer_copy, run_c01, crate::Backend::Dx12);
    native_case!(
        c01_vulkan_partial_buffer_copy,
        run_c01,
        crate::Backend::Vulkan
    );
    native_case!(
        c02_dx12_texture_copy_row_padding,
        run_c02,
        crate::Backend::Dx12
    );
    native_case!(
        c02_vulkan_texture_copy_row_padding,
        run_c02,
        crate::Backend::Vulkan
    );
    native_case!(
        c03_dx12_same_state_overlapping_waw,
        run_c03,
        crate::Backend::Dx12
    );
    native_case!(
        c03_vulkan_same_state_overlapping_waw,
        run_c03,
        crate::Backend::Vulkan
    );

    fn run_c01(backend: crate::Backend) {
        let size = 64;
        let mut graph = RenderGraph::new();
        let source_slot = graph.import_buffer_slot(
            "c01-source",
            ImportBufferContract {
                descriptor: BufferDesc { size },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let destination_slot = graph.import_buffer_slot(
            "c01-destination",
            ImportBufferContract {
                descriptor: BufferDesc { size },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let output = add_copy(
            &mut graph,
            "c01-copy",
            &source_slot.version,
            destination_slot.version,
            8,
            16,
            24,
        );
        let export = graph.export_buffer(
            output,
            ExportBufferContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph
            .compile(&CopyBackend::portable_capabilities())
            .unwrap()
            .graph;
        let source: Vec<u8> = (0..size as u8).collect();
        let mut expected = vec![0xCD; size as usize];
        expected[16..40].copy_from_slice(&source[8..32]);
        run_buffer_case(
            "C01",
            backend,
            &compiled,
            export,
            &[
                (source_slot.slot, source),
                (destination_slot.slot, vec![0xCD; size as usize]),
            ],
            &expected,
        );
    }

    fn run_c03(backend: crate::Backend) {
        let size = 64;
        let mut graph = RenderGraph::new();
        let source_a = graph.import_buffer_slot(
            "c03-source-a",
            ImportBufferContract {
                descriptor: BufferDesc { size },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let source_b = graph.import_buffer_slot(
            "c03-source-b",
            ImportBufferContract {
                descriptor: BufferDesc { size },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let destination = graph.import_buffer_slot(
            "c03-destination",
            ImportBufferContract {
                descriptor: BufferDesc { size },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let after_a = add_copy(
            &mut graph,
            "c03-copy-a",
            &source_a.version,
            destination.version,
            0,
            0,
            32,
        );
        let after_b = add_copy(
            &mut graph,
            "c03-copy-b",
            &source_b.version,
            after_a,
            8,
            16,
            24,
        );
        let export = graph.export_buffer(
            after_b,
            ExportBufferContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph
            .compile(&CopyBackend::portable_capabilities())
            .unwrap()
            .graph;
        let source_a_bytes = vec![0xA1; size as usize];
        let source_b_bytes = vec![0xB2; size as usize];
        let mut expected = vec![0xCD; size as usize];
        expected[0..32].copy_from_slice(&source_a_bytes[0..32]);
        expected[16..40].copy_from_slice(&source_b_bytes[8..32]);
        run_buffer_case(
            "C03",
            backend,
            &compiled,
            export,
            &[
                (source_a.slot, source_a_bytes),
                (source_b.slot, source_b_bytes),
                (destination.slot, vec![0xCD; size as usize]),
            ],
            &expected,
        );
    }

    fn run_buffer_case(
        case: &str,
        backend: crate::Backend,
        compiled: &fluxel_rendergraph::CompiledGraph,
        export: ExportBufferSlot,
        inputs: &[(fluxel_rendergraph::ImportBufferSlot, Vec<u8>)],
        expected: &[u8],
    ) {
        let device = Device::open(
            backend,
            crate::DeviceOptions {
                validation: crate::Validation::Required,
                ..crate::DeviceOptions::default()
            },
        )
        .unwrap();
        crate::imp::clear_validation_diagnostics(&device.inner);
        let usage = BufferUsage::from_kinds([
            fluxel_rendergraph::BufferUsageKind::CopySource,
            fluxel_rendergraph::BufferUsageKind::CopyDestination,
        ]);
        let mut resources = Resources {
            device: device.identity(),
            buffers: HashMap::new(),
            textures: HashMap::new(),
        };
        let mut frame_inputs = FrameInputs::new(());
        for (index, (slot, bytes)) in inputs.iter().enumerate() {
            let buffer = device
                .create_buffer(BufferDescriptor {
                    buffer: BufferDesc {
                        size: bytes.len() as u64,
                    },
                    usage,
                    memory: MemoryPolicy::DeviceOnly,
                })
                .unwrap();
            let state = crate::imp::upload_buffer_for_test(
                &device.inner,
                buffer.native(),
                buffer.lease().into(),
                bytes,
            )
            .unwrap();
            let id = BufferBindingId::new(index as u64 + 1);
            resources.buffers.insert(id, (buffer, state));
            frame_inputs.bind_buffer(*slot, id);
        }
        let executor = fluxel_rendergraph::FrameExecutor::new(CopyBackend::new(device.clone()));
        let mut frame = executor
            .execute(
                compiled,
                compiled.instantiate_local(frame_inputs),
                &resources,
                &NoObjects,
            )
            .unwrap();
        let completion = frame.submission.completion().clone();
        executor
            .try_backend()
            .unwrap()
            .wait(&completion, Duration::from_secs(10))
            .unwrap();
        assert_eq!(
            frame.submission.status().unwrap(),
            CompletionStatus::Complete
        );
        let exported = frame.exports.buffer(export).unwrap();
        let actual = crate::imp::readback_buffer_for_test(
            &device.inner,
            exported.physical.native(),
            exported.lease.clone(),
            exported.outgoing_state,
            exported.descriptor.size,
        )
        .unwrap();
        assert_eq!(
            actual,
            expected,
            "{backend:?} first difference: {:?}",
            actual.iter().zip(expected).position(|(a, b)| a != b)
        );
        let diagnostics = crate::imp::validation_diagnostics(&device.inner);
        assert!(
            diagnostics.is_empty(),
            "{case}/{backend:?} validation diagnostics: {diagnostics:#?}"
        );
        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
        let input = match case {
            "C01" => "src[8..32] -> dst[16..40], dst sentinel=0xCD",
            "C03" => "srcA[0..32] -> dst[0..32], then srcB[8..32] -> dst[16..40]",
            _ => "unknown",
        };
        eprintln!(
            "artifact case={case} backend={backend:?} commit={commit} os={}; hardware={:?}; shader=N/A; input={input}; resource=buffer size={}; expected={expected:?}; actual={actual:?}; first_difference={:?}; completion=Complete; diagnostics={diagnostics:?}",
            std::env::consts::OS,
            device.hardware(),
            exported.descriptor.size,
            actual.iter().zip(expected).position(|(a, b)| a != b)
        );
    }

    fn run_c02(backend: crate::Backend) {
        let source_desc = TextureDesc {
            dimension: fluxel_rendergraph::TextureDimension::D2,
            extent: fluxel_rendergraph::Extent3d {
                width: 7,
                height: 5,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let destination_desc = TextureDesc {
            extent: fluxel_rendergraph::Extent3d {
                width: 11,
                height: 9,
                depth: 1,
            },
            ..source_desc
        };
        let region = TextureCopyRegion {
            source_origin: [1, 1, 0],
            destination_origin: [3, 2, 0],
            extent: [5, 3, 1],
            source_mip_level: 0,
            destination_mip_level: 0,
        };
        let mut graph = RenderGraph::new();
        let source_slot = graph.import_texture_slot(
            "c02-source",
            fluxel_rendergraph::ImportTextureContract {
                descriptor: source_desc,
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let destination_slot = graph.import_texture_slot(
            "c02-destination",
            fluxel_rendergraph::ImportTextureContract {
                descriptor: destination_desc,
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        let copied = graph.add_copy_pass(
            "c02-copy",
            |pass| {
                let source = pass.read_texture(&source_slot.version, TextureRange::Whole);
                let (output, destination) = pass.write_texture(
                    destination_slot.version,
                    TextureRange::Whole,
                    WriteCoverage::Unknown,
                );
                (
                    output,
                    TextureCopyData {
                        source,
                        destination,
                        region,
                    },
                )
            },
            |commands, _, data, _| {
                commands.copy_texture(&data.source, &data.destination, data.region)
            },
        );
        let export = graph.export_texture(
            copied.output,
            fluxel_rendergraph::ExportTextureContract {
                final_state: ResourceAccessState::CopyDestination,
            },
        );
        let compiled = graph
            .compile(&CopyBackend::portable_capabilities())
            .unwrap()
            .graph;
        let mut source = Vec::new();
        for y in 0..source_desc.extent.height {
            for x in 0..source_desc.extent.width {
                source.extend_from_slice(&[x as u8, y as u8, (x + y) as u8, 0xFF]);
            }
        }
        let destination = vec![
            0xCD;
            (destination_desc.extent.width * destination_desc.extent.height * 4)
                as usize
        ];
        let mut expected = destination.clone();
        for y in 0..region.extent[1] as usize {
            let src_start = ((region.source_origin[1] as usize + y)
                * source_desc.extent.width as usize
                + region.source_origin[0] as usize)
                * 4;
            let dst_start = ((region.destination_origin[1] as usize + y)
                * destination_desc.extent.width as usize
                + region.destination_origin[0] as usize)
                * 4;
            let count = region.extent[0] as usize * 4;
            expected[dst_start..dst_start + count]
                .copy_from_slice(&source[src_start..src_start + count]);
        }
        run_texture_case(
            backend,
            &compiled,
            export,
            (source_slot.slot, source, source_desc),
            (destination_slot.slot, destination, destination_desc),
            &expected,
        );
    }

    fn run_texture_case(
        backend: crate::Backend,
        compiled: &fluxel_rendergraph::CompiledGraph,
        export: fluxel_rendergraph::ExportTextureSlot,
        source: (fluxel_rendergraph::ImportTextureSlot, Vec<u8>, TextureDesc),
        destination: (fluxel_rendergraph::ImportTextureSlot, Vec<u8>, TextureDesc),
        expected: &[u8],
    ) {
        let device = Device::open(
            backend,
            crate::DeviceOptions {
                validation: crate::Validation::Required,
                ..crate::DeviceOptions::default()
            },
        )
        .unwrap();
        crate::imp::clear_validation_diagnostics(&device.inner);
        let usage = TextureUsage::from_kinds([
            fluxel_rendergraph::TextureUsageKind::CopySource,
            fluxel_rendergraph::TextureUsageKind::CopyDestination,
        ]);
        let mut resources = Resources {
            device: device.identity(),
            buffers: HashMap::new(),
            textures: HashMap::new(),
        };
        let mut frame_inputs = FrameInputs::new(());
        for (index, (slot, bytes, descriptor)) in [source, destination].into_iter().enumerate() {
            let texture = device
                .create_texture(TextureDescriptor {
                    texture: descriptor,
                    usage,
                    memory: MemoryPolicy::DeviceOnly,
                })
                .unwrap();
            let state = crate::imp::upload_texture_for_test(
                &device.inner,
                texture.native(),
                texture.lease().into(),
                descriptor,
                &bytes,
            )
            .unwrap();
            let id = TextureBindingId::new(index as u64 + 1);
            resources.textures.insert(id, (texture, state));
            frame_inputs.bind_texture(slot, id);
        }
        let executor = fluxel_rendergraph::FrameExecutor::new(CopyBackend::new(device.clone()));
        let mut frame = executor
            .execute(
                compiled,
                compiled.instantiate_local(frame_inputs),
                &resources,
                &NoObjects,
            )
            .unwrap();
        let completion = frame.submission.completion().clone();
        executor
            .try_backend()
            .unwrap()
            .wait(&completion, Duration::from_secs(10))
            .unwrap();
        assert_eq!(
            frame.submission.status().unwrap(),
            CompletionStatus::Complete
        );
        let exported = frame.exports.texture(export).unwrap();
        let readback = crate::imp::readback_texture_for_test(
            &device.inner,
            exported.physical.native(),
            exported.lease.clone(),
            exported.descriptor,
            exported.outgoing_state,
        )
        .unwrap();
        assert_eq!(
            readback.tight,
            expected,
            "{backend:?} first difference: {:?}",
            readback
                .tight
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
        );
        let row_bytes = exported.descriptor.extent.width as usize * 4;
        for row in 0..exported.descriptor.extent.height as usize {
            assert!(
                readback.padded[row * readback.bytes_per_row as usize + row_bytes
                    ..(row + 1) * readback.bytes_per_row as usize]
                    .iter()
                    .all(|byte| *byte == 0),
                "{backend:?} row padding modified"
            );
        }
        let diagnostics = crate::imp::validation_diagnostics(&device.inner);
        assert!(
            diagnostics.is_empty(),
            "C02/{backend:?} validation diagnostics: {diagnostics:#?}"
        );
        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
        eprintln!(
            "artifact case=C02 backend={backend:?} commit={commit} os={}; hardware={:?}; shader=N/A; input=src 7x5 origin(1,1) -> dst 11x9 origin(3,2), extent 5x3; resource=texture descriptor={:?} row_pitch={} padding=zero; expected={expected:?}; actual={:?}; first_difference={:?}; completion=Complete; diagnostics={diagnostics:?}",
            std::env::consts::OS,
            device.hardware(),
            exported.descriptor,
            readback.bytes_per_row,
            readback.tight,
            readback
                .tight
                .iter()
                .zip(expected)
                .position(|(a, b)| a != b)
        );
    }

    fn hardware_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn copy_capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, false, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(DeviceLimits::new(0, 256))
        .buffers(BufferCapabilities::new(false, false, false))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .copies(true, true)
                .build(),
        )
        .build()
}

impl ExecutionBackend for CopyBackend {
    type Texture = Texture;
    type Buffer = Buffer;
    type RasterPipeline = UnsupportedRasterPipeline;
    type ComputePipeline = UnsupportedComputePipeline;
    type Bindings = UnsupportedBindings;
    type Encoder = CopyEncoder;
    type CommandBuffer = CopyCommandBuffer;
    type Completion = NativeCompletion;
    type Lease = ResourceLease;
    type Error = NativeExecutionError;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
    fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.device.identity()
    }

    fn create_transient_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<BoundTexture<Self::Texture, Self::Lease>, Self::Error> {
        let texture = self.device.create_texture(TextureDescriptor {
            texture: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundTexture {
            device: self.device.identity(),
            identity: texture.identity(),
            physical: texture.clone(),
            descriptor,
            usage: texture.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: texture.lease().into(),
        })
    }
    fn create_transient_buffer(
        &mut self,
        descriptor: BufferDesc,
        usage: BufferUsage,
    ) -> Result<BoundBuffer<Self::Buffer, Self::Lease>, Self::Error> {
        let buffer = self.device.create_buffer(BufferDescriptor {
            buffer: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundBuffer {
            device: self.device.identity(),
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor,
            usage: buffer.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: buffer.lease().into(),
        })
    }
    fn begin_encoder(&mut self, queue: QueueId) -> Result<Self::Encoder, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        crate::imp::begin_copy_encoder(&self.device.inner)
            .map(|native| CopyEncoder {
                native,
                device: self.device.identity(),
                leases: Vec::new(),
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn transition_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        texture: &Self::Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_texture(texture)?;
        crate::imp::transition_texture(
            &mut encoder.native,
            texture.native(),
            texture.descriptor().texture,
            range,
            before,
            after,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(texture.lease().into());
        Ok(())
    }
    fn transition_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        _range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_buffer(buffer)?;
        crate::imp::transition_buffer(&mut encoder.native, buffer.native(), before, after)
            .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(buffer.lease().into());
        Ok(())
    }
    fn begin_raster(
        &mut self,
        _: &mut Self::Encoder,
        _: &RasterPassDescriptor<'_, Self::Texture>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_raster(&mut self, _: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_compute(&mut self, _: &mut Self::Encoder, _: &str) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_compute(&mut self, _: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_copy(&mut self, _: &mut Self::Encoder, _: &str) -> Result<(), Self::Error> {
        Ok(())
    }
    fn end_copy(&mut self, _: &mut Self::Encoder) -> Result<(), Self::Error> {
        Ok(())
    }
    fn set_raster_pipeline(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::RasterPipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_compute_pipeline(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::ComputePipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_bindings(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::Bindings,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_vertex_buffer(
        &mut self,
        _: &mut Self::Encoder,
        _: u32,
        _: &Self::Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_index_buffer(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::Buffer,
        _: u64,
        _: IndexFormat,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_viewport(&mut self, _: &mut Self::Encoder, _: Viewport) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_scissor(&mut self, _: &mut Self::Encoder, _: ScissorRect) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw(
        &mut self,
        _: &mut Self::Encoder,
        _: Range<u32>,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw_indexed(
        &mut self,
        _: &mut Self::Encoder,
        _: Range<u32>,
        _: i32,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn dispatch(&mut self, _: &mut Self::Encoder, _: [u32; 3]) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn copy_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Texture,
        destination: &Self::Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_texture(source)?;
        self.check_texture(destination)?;
        validate_texture_copy(
            source.descriptor().texture,
            destination.descriptor().texture,
            region,
        )?;
        crate::imp::copy_texture(
            &mut encoder.native,
            source.native(),
            destination.native(),
            source.descriptor().texture,
            region,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(source.lease().into());
        encoder.leases.push(destination.lease().into());
        Ok(())
    }
    fn copy_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_buffer(source)?;
        self.check_buffer(destination)?;
        validate_buffer_copy(
            source.descriptor().buffer,
            destination.descriptor().buffer,
            region,
        )?;
        crate::imp::copy_buffer(
            &mut encoder.native,
            source.native(),
            destination.native(),
            region,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(source.lease().into());
        encoder.leases.push(destination.lease().into());
        Ok(())
    }
    fn finish_encoder(
        &mut self,
        encoder: Self::Encoder,
    ) -> Result<Self::CommandBuffer, Self::Error> {
        self.check_encoder(&encoder)?;
        let CopyEncoder {
            native,
            device,
            leases,
        } = encoder;
        crate::imp::finish_copy_encoder(native)
            .map(|native| CopyCommandBuffer {
                native,
                device,
                leases,
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: Self::CommandBuffer,
    ) -> Result<Self::Completion, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        require_device(
            command_buffer.device,
            self.device.identity(),
            NativeExecutionError::ForeignCommandBuffer,
        )?;
        let CopyCommandBuffer {
            native,
            device: _,
            leases,
        } = command_buffer;
        crate::imp::submit_copy(native, leases)
            .map(NativeCompletion)
            .map_err(NativeExecutionError::SubmitRejected)
    }
    fn completion_status(&self, completion: &Self::Completion) -> CompletionStatus {
        crate::imp::completion_status(&completion.0)
            .unwrap_or(CompletionStatus::Failed(CompletionFailure::DeviceLost))
    }
    fn retire(&mut self, completion: Self::Completion, leases: Vec<Self::Lease>) {
        self.retired.push(Retired { completion, leases });
    }
    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        let before = self.retired.len();
        let mut query_error = None;
        self.retired.retain(|entry| {
            let _ = entry.leases.len();
            match crate::imp::completion_status(&entry.completion.0) {
                Ok(CompletionStatus::Pending) => true,
                Ok(_) => false,
                Err(error) => {
                    query_error.get_or_insert(error);
                    true
                }
            }
        });
        if let Some(error) = query_error {
            Err(NativeExecutionError::Completion(error))
        } else {
            Ok(before - self.retired.len())
        }
    }
}

fn validate_buffer_copy(
    source: BufferDesc,
    destination: BufferDesc,
    region: BufferCopyRegion,
) -> Result<(), NativeExecutionError> {
    if region.size == 0 {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy size is zero",
        ));
    }
    let source_end = region.source_offset.checked_add(region.size).ok_or(
        NativeExecutionError::InvalidTransfer("source range overflow"),
    )?;
    let destination_end = region.destination_offset.checked_add(region.size).ok_or(
        NativeExecutionError::InvalidTransfer("destination range overflow"),
    )?;
    if source_end > source.size || destination_end > destination.size {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy is out of bounds",
        ));
    }
    Ok(())
}

fn validate_texture_copy(
    source: TextureDesc,
    destination: TextureDesc,
    region: TextureCopyRegion,
) -> Result<(), NativeExecutionError> {
    if source.format != destination.format {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture formats differ",
        ));
    }
    if region.extent.contains(&0) {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture copy extent is zero",
        ));
    }
    if region.source_mip_level >= source.mip_levels
        || region.destination_mip_level >= destination.mip_levels
    {
        return Err(NativeExecutionError::InvalidTransfer(
            "texture mip is out of bounds",
        ));
    }
    let mip_extent = |desc: TextureDesc, mip: u32| {
        [
            (desc.extent.width >> mip).max(1),
            (desc.extent.height >> mip).max(1),
            (desc.extent.depth >> mip).max(1),
        ]
    };
    for (origin, extent, limit) in [
        (
            region.source_origin,
            region.extent,
            mip_extent(source, region.source_mip_level),
        ),
        (
            region.destination_origin,
            region.extent,
            mip_extent(destination, region.destination_mip_level),
        ),
    ] {
        for axis in 0..3 {
            if origin[axis]
                .checked_add(extent[axis])
                .is_none_or(|end| end > limit[axis])
            {
                return Err(NativeExecutionError::InvalidTransfer(
                    "texture copy is out of bounds",
                ));
            }
        }
    }
    Ok(())
}

impl CopyBackend {
    fn check_encoder(&self, encoder: &CopyEncoder) -> Result<(), NativeExecutionError> {
        require_device(
            encoder.device,
            self.device.identity(),
            NativeExecutionError::ForeignEncoder,
        )
    }

    fn check_buffer(&self, buffer: &Buffer) -> Result<(), NativeExecutionError> {
        (buffer.device_identity() == self.device.identity())
            .then_some(())
            .ok_or(NativeExecutionError::ForeignResource)
    }
    fn check_texture(&self, texture: &Texture) -> Result<(), NativeExecutionError> {
        (texture.device_identity() == self.device.identity())
            .then_some(())
            .ok_or(NativeExecutionError::ForeignResource)
    }
}

fn require_device(
    actual: fluxel_rendergraph::DeviceIdentity,
    expected: fluxel_rendergraph::DeviceIdentity,
    mismatch: NativeExecutionError,
) -> Result<(), NativeExecutionError> {
    (actual == expected).then_some(()).ok_or(mismatch)
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use fluxel_rendergraph::{Extent3d, TextureDimension};

    #[test]
    fn copy_profile_exposes_only_one_copy_queue() {
        let capabilities = CopyBackend::portable_capabilities();
        assert_eq!(capabilities.queues.len(), 1);
        let queue = &capabilities.queues[0];
        assert!(queue.capabilities.copy);
        assert!(!queue.capabilities.compute);
        assert!(!queue.capabilities.raster);
    }

    #[test]
    fn encoder_and_command_buffer_identity_checks_fail_closed() {
        let first = fluxel_rendergraph::DeviceIdentity::new(1);
        let second = fluxel_rendergraph::DeviceIdentity::new(2);
        assert_eq!(
            require_device(first, second, NativeExecutionError::ForeignEncoder),
            Err(NativeExecutionError::ForeignEncoder)
        );
        assert_eq!(
            require_device(first, second, NativeExecutionError::ForeignCommandBuffer),
            Err(NativeExecutionError::ForeignCommandBuffer)
        );
        assert_eq!(
            require_device(first, first, NativeExecutionError::ForeignEncoder),
            Ok(())
        );
    }

    #[test]
    fn native_buffer_validation_rejects_zero_overflow_and_out_of_bounds() {
        let descriptor = BufferDesc { size: 32 };
        for region in [
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 0,
                size: 0,
            },
            BufferCopyRegion {
                source_offset: u64::MAX,
                destination_offset: 0,
                size: 2,
            },
            BufferCopyRegion {
                source_offset: 16,
                destination_offset: 17,
                size: 16,
            },
        ] {
            assert!(matches!(
                validate_buffer_copy(descriptor, descriptor, region),
                Err(NativeExecutionError::InvalidTransfer(_))
            ));
        }
    }

    #[test]
    fn native_texture_validation_rejects_format_and_extent_mismatches() {
        let descriptor = TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 8,
                height: 8,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let region = TextureCopyRegion {
            source_origin: [0, 0, 0],
            destination_origin: [4, 0, 0],
            extent: [5, 1, 1],
            source_mip_level: 0,
            destination_mip_level: 0,
        };
        assert!(matches!(
            validate_texture_copy(descriptor, descriptor, region),
            Err(NativeExecutionError::InvalidTransfer(_))
        ));
        assert!(matches!(
            validate_texture_copy(
                descriptor,
                TextureDesc {
                    format: TextureFormat::Bgra8Unorm,
                    ..descriptor
                },
                TextureCopyRegion {
                    destination_origin: [0, 0, 0],
                    extent: [1, 1, 1],
                    ..region
                }
            ),
            Err(NativeExecutionError::InvalidTransfer(_))
        ));
    }
}
