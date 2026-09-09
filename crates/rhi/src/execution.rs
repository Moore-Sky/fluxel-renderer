//! Native Copy and fixed-Compute execution of portable RenderGraph plans.

use core::{fmt, time::Duration};
use std::collections::HashMap;
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
    Buffer, BufferDescriptor, ComputeBindings, ComputeCreateError, ComputePipeline, Device,
    MemoryPolicy, ResourceCreateError, ResourceLease, Texture, TextureDescriptor,
};

/// Failure while lowering or executing the current native Copy/Compute slice.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeExecutionError {
    /// An owned transient could not be created.
    Resource(ResourceCreateError),
    /// The plan selected a command family outside this backend's supported slice.
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
    /// A compute dispatch was zero-sized or exceeded device limits.
    InvalidDispatch,
    /// The active pipeline and bound fixed binding recipe do not match.
    ComputeBindingMismatch,
}

impl fmt::Display for NativeExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resource(error) => write!(formatter, "resource creation failed: {error}"),
            Self::UnsupportedCommandFamily => {
                formatter.write_str("command family is not supported by this native backend")
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
            Self::InvalidDispatch => formatter.write_str("invalid compute dispatch dimensions"),
            Self::ComputeBindingMismatch => {
                formatter.write_str("compute bindings do not match the active pipeline")
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
    active_compute: Option<ComputePipeline>,
    bound_compute_pipeline: Option<ComputePipeline>,
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
        BindingResource, BindingSetId, BoundBindings, BoundComputePipeline, BoundRasterPipeline,
        BufferBindingId, BufferRead, BufferReadWrite, BufferReadWriteUse, BufferWrite,
        ComputePipelineId, DiagnosticContext, ExportBufferContract, ExportBufferSlot,
        ExternalOwnership, FrameBindingError, FrameBindingErrorKind, FrameInputs,
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

    impl FrameResourceProvider<ComputeBackend> for Resources {
        fn texture(
            &self,
            id: fluxel_rendergraph::TextureBindingId,
        ) -> Result<BoundTexture<Texture, ResourceLease>, FrameBindingError> {
            <Self as FrameResourceProvider<CopyBackend>>::texture(self, id)
        }
        fn buffer(
            &self,
            id: BufferBindingId,
        ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
            <Self as FrameResourceProvider<CopyBackend>>::buffer(self, id)
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
    native_case!(k01_dx12_wrapping_add, run_k01, crate::Backend::Dx12);
    native_case!(k01_vulkan_wrapping_add, run_k01, crate::Backend::Vulkan);
    native_case!(k02_dx12_ordered_add_multiply, run_k02, crate::Backend::Dx12);
    native_case!(
        k02_vulkan_ordered_add_multiply,
        run_k02,
        crate::Backend::Vulkan
    );
    #[test]
    #[ignore = "requires native validation plus a real GPU"]
    fn k01_paired_same_compiled_plan() {
        let _guard = hardware_guard();
        crate::imp::initialize_validation_capture();
        let case = build_k01();
        run_compute_case_bundle("K01", crate::Backend::Dx12, &case);
        run_compute_case_bundle("K01", crate::Backend::Vulkan, &case);
    }
    #[test]
    #[ignore = "requires native validation plus a real GPU"]
    fn k02_paired_same_compiled_plan() {
        let _guard = hardware_guard();
        crate::imp::initialize_validation_capture();
        let case = build_k02();
        run_compute_case_bundle("K02", crate::Backend::Dx12, &case);
        run_compute_case_bundle("K02", crate::Backend::Vulkan, &case);
    }
    native_case!(
        negative_dx12_compute_fail_closed,
        run_compute_negative_paths,
        crate::Backend::Dx12
    );
    native_case!(
        negative_vulkan_compute_fail_closed,
        run_compute_negative_paths,
        crate::Backend::Vulkan
    );

    fn run_compute_negative_paths(backend_kind: crate::Backend) {
        let device = Device::open(
            backend_kind,
            crate::DeviceOptions {
                validation: crate::Validation::Required,
                ..crate::DeviceOptions::default()
            },
        )
        .unwrap();
        crate::imp::clear_validation_diagnostics(&device.inner);
        let usage = BufferUsage::from_kinds([
            fluxel_rendergraph::BufferUsageKind::StorageRead,
            fluxel_rendergraph::BufferUsageKind::StorageWrite,
        ]);
        let buffer = device
            .create_buffer(BufferDescriptor {
                buffer: BufferDesc { size: 64 },
                usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        let pipeline_a = device
            .create_compute_pipeline(crate::ComputeKernel::WrappingAdd)
            .unwrap();
        let pipeline_b = device
            .create_compute_pipeline(crate::ComputeKernel::WrappingMultiply)
            .unwrap();
        let binding_a = device
            .create_compute_bindings(&pipeline_a, &buffer, 0, 64)
            .unwrap();
        let binding_b = device
            .create_compute_bindings(&pipeline_b, &buffer, 0, 64)
            .unwrap();
        let mut provider = ComputeObjectProvider::new(&device);
        let pipeline_id = ComputePipelineId::new(901);
        let binding_id = BindingSetId::new(902);
        provider
            .register_pipeline(pipeline_id, pipeline_a.clone())
            .unwrap();
        provider.register_bindings(binding_id, pipeline_id).unwrap();
        assert!(
            provider
                .compute_pipeline(ComputePipelineId::new(999))
                .is_err()
        );
        assert!(provider.bindings(BindingSetId::new(999), &[], &[]).is_err());
        let wrong_semantic = [ResolvedBindingResource::Buffer {
            physical: &buffer,
            range: BufferRange::Whole,
            semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                fluxel_rendergraph::BufferReadUse::Storage,
            ),
        }];
        assert!(provider.bindings(binding_id, &wrong_semantic, &[]).is_err());
        let misaligned = [ResolvedBindingResource::Buffer {
            physical: &buffer,
            range: BufferRange::Bytes { offset: 2, size: 4 },
            semantic: fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
                BufferReadWriteUse::Storage,
            ),
        }];
        assert!(provider.bindings(binding_id, &misaligned, &[]).is_err());
        let foreign_device = Device::open(backend_kind, crate::DeviceOptions::default()).unwrap();
        let foreign_pipeline = foreign_device
            .create_compute_pipeline(crate::ComputeKernel::WrappingAdd)
            .unwrap();
        let mut backend = ComputeBackend::new(device.clone());
        let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
        backend.begin_compute(&mut encoder, "negative").unwrap();
        assert_eq!(
            backend.dispatch(&mut encoder, [1, 1, 1]),
            Err(NativeExecutionError::ComputeBindingMismatch)
        );
        assert_eq!(
            backend.set_compute_pipeline(&mut encoder, &foreign_pipeline),
            Err(NativeExecutionError::ForeignResource)
        );
        backend
            .set_compute_pipeline(&mut encoder, &pipeline_a)
            .unwrap();
        assert_eq!(
            backend.dispatch(&mut encoder, [1, 1, 1]),
            Err(NativeExecutionError::ComputeBindingMismatch)
        );
        assert_eq!(
            backend.set_bindings(&mut encoder, &binding_b),
            Err(NativeExecutionError::ComputeBindingMismatch)
        );
        backend.set_bindings(&mut encoder, &binding_a).unwrap();
        let maximum = backend
            .capabilities()
            .limits
            .max_compute_workgroups_per_dimension;
        assert_eq!(
            backend.dispatch(&mut encoder, [maximum[0].saturating_add(1), 1, 1]),
            Err(NativeExecutionError::InvalidDispatch)
        );
        // No finish/submit: every negative assertion is pre-submission.
        drop(encoder);
        assert!(crate::imp::validation_diagnostics(&device.inner).is_empty());
    }

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

    struct ComputeData {
        pipeline: ComputePipelineId,
        bindings: BindingSetId,
        storage: BufferReadWrite,
        groups: [u32; 3],
    }

    struct ComputeCase {
        compiled: fluxel_rendergraph::CompiledGraph,
        export: ExportBufferSlot,
        slot: fluxel_rendergraph::ImportBufferSlot,
        input: Vec<u8>,
        expected: Vec<u8>,
        artifacts: Vec<(ComputePipelineId, crate::ComputeKernel, BindingSetId)>,
        input_description: &'static str,
    }

    fn add_compute(
        graph: &mut RenderGraph,
        name: &str,
        input: fluxel_rendergraph::BufferVersion,
        pipeline: ComputePipelineId,
        bindings: BindingSetId,
        range: BufferRange,
        groups: [u32; 3],
    ) -> fluxel_rendergraph::BufferVersion {
        graph
            .add_compute_pass(
                name,
                move |pass| {
                    let (output, storage) =
                        pass.read_write_buffer(input, BufferReadWriteUse::Storage, range);
                    (
                        output,
                        ComputeData {
                            pipeline,
                            bindings,
                            storage,
                            groups,
                        },
                    )
                },
                |commands, resolver, data, _| {
                    commands.set_pipeline(data.pipeline)?;
                    let resolved = resolver.resolve_bindings(
                        data.bindings,
                        &[BindingResource::BufferReadWrite(&data.storage)],
                        &[],
                    )?;
                    commands.set_bindings(&resolved)?;
                    commands.dispatch(data.groups)
                },
            )
            .output
    }

    fn run_k01(backend: crate::Backend) {
        let case = build_k01();
        run_compute_case_bundle("K01", backend, &case);
    }

    fn build_k01() -> ComputeCase {
        let (mut graph, input_version, input_slot) = compute_input_graph("k01-input");
        let pipeline = ComputePipelineId::new(101);
        let bindings = BindingSetId::new(201);
        let output = add_compute(
            &mut graph,
            "k01-wrapping-add",
            input_version,
            pipeline,
            bindings,
            compute_range(),
            [1, 1, 1],
        );
        let export = graph.export_buffer(
            output,
            ExportBufferContract {
                final_state: ResourceAccessState::ShaderStorageReadWrite,
            },
        );
        let compiled = graph
            .compile(&ComputeBackend::portable_capabilities())
            .unwrap()
            .graph;
        let input = compute_input_bytes();
        let mut expected = input.clone();
        for bytes in expected[..256].chunks_exact_mut(4) {
            let value = u32::from_le_bytes(bytes.try_into().unwrap()).wrapping_add(1);
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        ComputeCase {
            compiled,
            export,
            slot: input_slot,
            input,
            expected,
            artifacts: vec![(pipeline, crate::ComputeKernel::WrappingAdd, bindings)],
            input_description: "one RW storage buffer bytes[0..256]; wrapping add; bytes[256..320] sentinel=0xCD",
        }
    }

    fn run_k02(backend: crate::Backend) {
        let case = build_k02();
        run_compute_case_bundle("K02", backend, &case);
    }

    fn build_k02() -> ComputeCase {
        let (mut graph, input_version, input_slot) = compute_input_graph("k02-input");
        let add_pipeline = ComputePipelineId::new(102);
        let multiply_pipeline = ComputePipelineId::new(103);
        let add_bindings = BindingSetId::new(202);
        let multiply_bindings = BindingSetId::new(203);
        let after_add = add_compute(
            &mut graph,
            "k02-wrapping-add",
            input_version,
            add_pipeline,
            add_bindings,
            compute_range(),
            [1, 1, 1],
        );
        let output = add_compute(
            &mut graph,
            "k02-wrapping-multiply",
            after_add,
            multiply_pipeline,
            multiply_bindings,
            compute_range(),
            [1, 1, 1],
        );
        let export = graph.export_buffer(
            output,
            ExportBufferContract {
                final_state: ResourceAccessState::ShaderStorageReadWrite,
            },
        );
        let compiled = graph
            .compile(&ComputeBackend::portable_capabilities())
            .unwrap()
            .graph;
        let input = compute_input_bytes();
        let mut expected = input.clone();
        for bytes in expected[..256].chunks_exact_mut(4) {
            let value = u32::from_le_bytes(bytes.try_into().unwrap());
            bytes.copy_from_slice(&value.wrapping_add(1).wrapping_mul(3).to_le_bytes());
        }
        ComputeCase {
            compiled,
            export,
            slot: input_slot,
            input,
            expected,
            artifacts: vec![
                (
                    add_pipeline,
                    crate::ComputeKernel::WrappingAdd,
                    add_bindings,
                ),
                (
                    multiply_pipeline,
                    crate::ComputeKernel::WrappingMultiply,
                    multiply_bindings,
                ),
            ],
            input_description: "two ordered RW storage passes bytes[0..256]: wrapping add then wrapping multiply; bytes[256..320] sentinel=0xCD",
        }
    }

    fn run_compute_case_bundle(case: &str, backend: crate::Backend, bundle: &ComputeCase) {
        run_compute_case(
            case,
            backend,
            &bundle.compiled,
            bundle.export,
            bundle.slot,
            bundle.input.clone(),
            &bundle.expected,
            &bundle.artifacts,
            bundle.input_description,
        );
    }

    fn compute_input_graph(
        name: &str,
    ) -> (
        RenderGraph,
        fluxel_rendergraph::BufferVersion,
        fluxel_rendergraph::ImportBufferSlot,
    ) {
        let mut graph = RenderGraph::new();
        let slot = graph.import_buffer_slot(
            name,
            ImportBufferContract {
                descriptor: BufferDesc { size: 320 },
                initial_state: ResourceAccessState::CopyDestination,
                ownership: ExternalOwnership::Caller,
                initial_contents: InitialContents::Defined,
            },
        );
        (graph, slot.version, slot.slot)
    }

    fn compute_range() -> BufferRange {
        BufferRange::Bytes {
            offset: 0,
            size: 256,
        }
    }

    fn compute_input_bytes() -> Vec<u8> {
        let mut bytes = Vec::with_capacity(320);
        for index in 0..64_u32 {
            bytes.extend_from_slice(&(u32::MAX - index * 17).to_le_bytes());
        }
        bytes.extend(std::iter::repeat_n(0xCD, 64));
        bytes
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "hardware artifact fields are intentionally explicit"
    )]
    fn run_compute_case(
        case: &str,
        backend: crate::Backend,
        compiled: &fluxel_rendergraph::CompiledGraph,
        export: ExportBufferSlot,
        slot: fluxel_rendergraph::ImportBufferSlot,
        input: Vec<u8>,
        expected: &[u8],
        artifacts: &[(ComputePipelineId, crate::ComputeKernel, BindingSetId)],
        input_description: &str,
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
            fluxel_rendergraph::BufferUsageKind::StorageRead,
            fluxel_rendergraph::BufferUsageKind::StorageWrite,
        ]);
        let buffer = device
            .create_buffer(BufferDescriptor {
                buffer: BufferDesc {
                    size: input.len() as u64,
                },
                usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        let initial_state = crate::imp::upload_buffer_for_test(
            &device.inner,
            buffer.native(),
            buffer.lease().into(),
            &input,
        )
        .unwrap();
        let mut resources = Resources {
            device: device.identity(),
            buffers: HashMap::from([(BufferBindingId::new(1), (buffer, initial_state))]),
            textures: HashMap::new(),
        };
        let mut frame_inputs = FrameInputs::new(());
        frame_inputs.bind_buffer(slot, BufferBindingId::new(1));
        let mut provider = ComputeObjectProvider::new(&device);
        for (pipeline_id, kernel, bindings) in artifacts {
            let pipeline = device.create_compute_pipeline(*kernel).unwrap();
            provider.register_pipeline(*pipeline_id, pipeline).unwrap();
            provider.register_bindings(*bindings, *pipeline_id).unwrap();
        }
        let executor = fluxel_rendergraph::FrameExecutor::new(ComputeBackend::new(device.clone()));
        let mut frame = executor
            .execute(
                compiled,
                compiled.instantiate_local(frame_inputs),
                &resources,
                &provider,
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
        assert_eq!(
            exported.outgoing_state,
            ResourceAccessState::ShaderStorageReadWrite
        );
        let actual = readback_exported_buffer_for_test(&device, exported).unwrap();
        assert_eq!(
            actual,
            expected,
            "{backend:?} first difference: {:?}",
            actual
                .iter()
                .zip(expected)
                .position(|(left, right)| left != right)
        );
        let diagnostics = crate::imp::validation_diagnostics(&device.inner);
        assert!(
            diagnostics.is_empty(),
            "{case}/{backend:?} validation diagnostics: {diagnostics:#?}"
        );
        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
        let artifacts: Vec<_> = artifacts
            .iter()
            .map(|(_, kernel, _)| kernel.portable_identity())
            .collect();
        let plan_summary = format!("{:?}", compiled.execution_plan());
        eprintln!(
            "artifact case={case} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label={case}-compute-v1; execution_plan={plan_summary}; shader={artifacts:?}; binding_layout=one RW storage buffer range=0..256; dispatch=[1,1,1]; input={input_description}; expected={expected:?}; actual={actual:?}; first_difference={:?}; outgoing_state={:?}; completion=Complete; diagnostics={diagnostics:?}",
            std::env::consts::OS,
            device.hardware(),
            actual
                .iter()
                .zip(expected)
                .position(|(left, right)| left != right),
            exported.outgoing_state,
        );
        // Keep the provider/resource binding path explicit in this fixture.
        resources.buffers.clear();
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

/// Registry for the only fixed compute pipeline and binding recipes in 0.1.3.
///
/// Registration is explicit so graph IDs never become implicit native handles.
pub struct ComputeObjectProvider {
    owner: Device,
    device: fluxel_rendergraph::DeviceIdentity,
    pipelines: HashMap<fluxel_rendergraph::ComputePipelineId, ComputePipeline>,
    bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (fluxel_rendergraph::ComputePipelineId, ComputePipeline),
    >,
}

impl ComputeObjectProvider {
    /// Creates an empty registry for `device`.
    pub fn new(device: &Device) -> Self {
        Self {
            owner: device.clone(),
            device: device.identity(),
            pipelines: HashMap::new(),
            bindings: HashMap::new(),
        }
    }

    /// Registers a graph pipeline identity with a device-affine fixed artifact.
    pub fn register_pipeline(
        &mut self,
        id: fluxel_rendergraph::ComputePipelineId,
        pipeline: ComputePipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.pipelines.insert(id, pipeline);
        Ok(())
    }

    /// Registers a binding recipe and the pipeline layout it must resolve against.
    pub fn register_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .pipelines
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        self.bindings
            .insert(id, (expected_pipeline, pipeline.clone()));
        Ok(())
    }
}

fn provider_error(
    kind: fluxel_rendergraph::RecordingErrorKind,
    detail: impl Into<String>,
) -> fluxel_rendergraph::RecordingError {
    fluxel_rendergraph::RecordingError {
        kind,
        context: fluxel_rendergraph::DiagnosticContext {
            passes: vec![],
            resource: None,
            texture_slot: None,
            buffer_slot: None,
            detail: detail.into(),
            capabilities: None,
        },
    }
}

fn compute_binding_error_kind(
    error: &ComputeCreateError,
) -> fluxel_rendergraph::RecordingErrorKind {
    match error {
        ComputeCreateError::ForeignDevice
        | ComputeCreateError::InvalidBindingRange
        | ComputeCreateError::StorageUsageRequired
        | ComputeCreateError::UnsupportedComputeLimits => {
            fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe
        }
        ComputeCreateError::ShaderValidation(_)
        | ComputeCreateError::ShaderCompilation(_)
        | ComputeCreateError::NativeObjectCreation(_)
        | ComputeCreateError::NativeFailure(_) => {
            fluxel_rendergraph::RecordingErrorKind::BackendObjectCreation
        }
    }
}

impl fluxel_rendergraph::RenderObjectProvider<ComputeBackend> for ComputeObjectProvider {
    fn raster_pipeline(
        &self,
        _: fluxel_rendergraph::RasterPipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundRasterPipeline<UnsupportedRasterPipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        Err(provider_error(
            fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
            "raster is outside the compute slice",
        ))
    }

    fn compute_pipeline(
        &self,
        id: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundComputePipeline<ComputePipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        let pipeline = self.pipelines.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown compute pipeline",
            )
        })?;
        if pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "foreign compute pipeline",
            ));
        }
        Ok(fluxel_rendergraph::BoundComputePipeline {
            device: self.device,
            physical: pipeline.clone(),
            lease: pipeline.lease().into(),
        })
    }

    fn bindings(
        &self,
        id: fluxel_rendergraph::BindingSetId,
        resources: &[fluxel_rendergraph::ResolvedBindingResource<'_, Texture, Buffer>],
        dynamic_offsets: &[u32],
    ) -> Result<
        fluxel_rendergraph::BoundBindings<ComputeBindings, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        if !dynamic_offsets.is_empty() || resources.len() != 1 {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute fixture requires exactly one RW storage buffer and no dynamic offsets",
            ));
        }
        let (_, pipeline) = self.bindings.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown compute binding recipe",
            )
        })?;
        let fluxel_rendergraph::ResolvedBindingResource::Buffer {
            physical: buffer,
            range,
            semantic,
        } = &resources[0]
        else {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute fixture requires a buffer",
            ));
        };
        if !matches!(
            semantic,
            fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
                fluxel_rendergraph::BufferReadWriteUse::Storage
            )
        ) {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::DeclaredUseMismatch,
                "compute fixture requires BufferReadWrite(Storage)",
            ));
        }
        if buffer.device_identity() != self.device || pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute binding is foreign to provider device",
            ));
        }
        let (offset, size) = match *range {
            BufferRange::Whole => (0, buffer.descriptor().buffer.size),
            BufferRange::Bytes { offset, size } => (offset, size),
        };
        let bindings = self
            .owner
            .create_compute_bindings(pipeline, buffer, offset, size)
            .map_err(|error| {
                provider_error(compute_binding_error_kind(&error), error.to_string())
            })?;
        Ok(fluxel_rendergraph::BoundBindings {
            device: self.device,
            physical: bindings.clone(),
            lease: bindings.lease().into(),
        })
    }
}

/// A serial DX12/Vulkan backend that adds the fixed 0.1.3 compute slice.
pub struct ComputeBackend {
    device: Device,
    capabilities: DeviceCapabilities,
    retired: Vec<Retired>,
}

impl ComputeBackend {
    /// Creates a compute-and-copy backend over an already opened native device.
    pub fn new(device: Device) -> Self {
        Self {
            capabilities: compute_capabilities(&device),
            device,
            retired: Vec::new(),
        }
    }

    /// Returns the normalized capability profile used by fixed compute fixtures.
    pub fn portable_capabilities() -> DeviceCapabilities {
        // The cross-backend profile is conservative. A real backend instance
        // additionally carries its verified native workgroup count.
        compute_capabilities_from_limit([65_535, 65_535, 65_535])
    }

    /// Waits outside graph execution for an accepted submission.
    pub fn wait(&self, completion: &NativeCompletion, timeout: Duration) -> Result<(), WaitError> {
        CopyBackend::new(self.device.clone()).wait(completion, timeout)
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
                active_compute: None,
                bound_compute_pipeline: None,
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
            active_compute: _,
            bound_compute_pipeline: _,
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

fn compute_capabilities(device: &Device) -> DeviceCapabilities {
    let portable_limit = [65_535; 3];
    debug_assert!(
        device
            .capabilities()
            .max_compute_workgroups_per_dimension
            .into_iter()
            .zip(portable_limit)
            .all(|(native, required)| native >= required),
        "Device::open validates the portable dispatch baseline"
    );
    compute_capabilities_from_limit(portable_limit)
}

fn compute_capabilities_from_limit(maximum: [u32; 3]) -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(DeviceLimits::new(0, 256).with_max_compute_workgroups_per_dimension(maximum))
        .buffers(BufferCapabilities::new(true, true, false))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .copies(true, true)
                .build(),
        )
        .build()
}

impl ExecutionBackend for ComputeBackend {
    type Texture = Texture;
    type Buffer = Buffer;
    type RasterPipeline = UnsupportedRasterPipeline;
    type ComputePipeline = ComputePipeline;
    type Bindings = ComputeBindings;
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
    ) -> Result<BoundTexture<Texture, ResourceLease>, Self::Error> {
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
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, Self::Error> {
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
    fn begin_encoder(&mut self, queue: QueueId) -> Result<CopyEncoder, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        crate::imp::begin_copy_encoder(&self.device.inner)
            .map(|native| CopyEncoder {
                native,
                device: self.device.identity(),
                leases: Vec::new(),
                active_compute: None,
                bound_compute_pipeline: None,
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn transition_texture(
        &mut self,
        encoder: &mut CopyEncoder,
        texture: &Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.copy()
            .transition_texture(encoder, texture, range, before, after)
    }
    fn transition_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        buffer: &Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.copy()
            .transition_buffer(encoder, buffer, range, before, after)
    }
    fn begin_raster(
        &mut self,
        _: &mut CopyEncoder,
        _: &RasterPassDescriptor<'_, Texture>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_raster(&mut self, _: &mut CopyEncoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_compute(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        crate::imp::begin_compute(&mut encoder.native, label)
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        Ok(())
    }
    fn end_compute(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        crate::imp::end_compute(&mut encoder.native).map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        Ok(())
    }
    fn begin_copy(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.copy().begin_copy(encoder, label)
    }
    fn end_copy(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.copy().end_copy(encoder)
    }
    fn set_raster_pipeline(
        &mut self,
        _: &mut CopyEncoder,
        _: &UnsupportedRasterPipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_compute_pipeline(
        &mut self,
        encoder: &mut CopyEncoder,
        pipeline: &ComputePipeline,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_pipeline(pipeline)?;
        crate::imp::set_compute_pipeline(&mut encoder.native, pipeline.native())
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = Some(pipeline.clone());
        encoder.bound_compute_pipeline = None;
        encoder.leases.push(pipeline.lease().into());
        Ok(())
    }
    fn set_bindings(
        &mut self,
        encoder: &mut CopyEncoder,
        bindings: &ComputeBindings,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if bindings.device_identity() != self.device.identity() {
            return Err(NativeExecutionError::ForeignResource);
        }
        let pipeline = encoder
            .active_compute
            .as_ref()
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        if !bindings.pipeline().same_object(pipeline) {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::set_compute_bindings(&mut encoder.native, bindings.native())
            .map_err(NativeExecutionError::Recording)?;
        encoder.bound_compute_pipeline = Some(pipeline.clone());
        encoder.leases.push(bindings.lease().into());
        Ok(())
    }
    fn set_vertex_buffer(
        &mut self,
        _: &mut CopyEncoder,
        _: u32,
        _: &Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_index_buffer(
        &mut self,
        _: &mut CopyEncoder,
        _: &Buffer,
        _: u64,
        _: IndexFormat,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_viewport(&mut self, _: &mut CopyEncoder, _: Viewport) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_scissor(&mut self, _: &mut CopyEncoder, _: ScissorRect) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw(
        &mut self,
        _: &mut CopyEncoder,
        _: Range<u32>,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw_indexed(
        &mut self,
        _: &mut CopyEncoder,
        _: Range<u32>,
        _: i32,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn dispatch(&mut self, encoder: &mut CopyEncoder, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !valid_compute_dispatch(
            groups,
            self.capabilities
                .limits
                .max_compute_workgroups_per_dimension,
        ) {
            return Err(NativeExecutionError::InvalidDispatch);
        }
        if encoder.active_compute.is_none() || encoder.bound_compute_pipeline.is_none() {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::dispatch(&mut encoder.native, groups).map_err(NativeExecutionError::Recording)
    }
    fn copy_texture(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Texture,
        destination: &Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_texture(encoder, source, destination, region)
    }
    fn copy_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Buffer,
        destination: &Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_buffer(encoder, source, destination, region)
    }
    fn finish_encoder(&mut self, encoder: CopyEncoder) -> Result<CopyCommandBuffer, Self::Error> {
        self.copy().finish_encoder(encoder)
    }
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: CopyCommandBuffer,
    ) -> Result<NativeCompletion, Self::Error> {
        self.copy().submit(queue, command_buffer)
    }
    fn completion_status(&self, completion: &NativeCompletion) -> CompletionStatus {
        crate::imp::completion_status(&completion.0)
            .unwrap_or(CompletionStatus::Failed(CompletionFailure::DeviceLost))
    }
    fn retire(&mut self, completion: NativeCompletion, leases: Vec<ResourceLease>) {
        self.retired.push(Retired { completion, leases });
    }
    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        let before = self.retired.len();
        let mut query_error = None;
        self.retired.retain(
            |entry| match crate::imp::completion_status(&entry.completion.0) {
                Ok(CompletionStatus::Pending) => true,
                Ok(_) => false,
                Err(error) => {
                    query_error.get_or_insert(error);
                    true
                }
            },
        );
        match query_error {
            Some(error) => Err(NativeExecutionError::Completion(error)),
            None => Ok(before - self.retired.len()),
        }
    }
}

fn valid_compute_dispatch(groups: [u32; 3], maximum: [u32; 3]) -> bool {
    groups
        .into_iter()
        .zip(maximum)
        .all(|(given, maximum)| given != 0 && given <= maximum)
}

impl ComputeBackend {
    fn copy(&self) -> CopyBackend {
        CopyBackend {
            device: self.device.clone(),
            capabilities: copy_capabilities(),
            retired: Vec::new(),
        }
    }
    fn check_encoder(&self, encoder: &CopyEncoder) -> Result<(), NativeExecutionError> {
        require_device(
            encoder.device,
            self.device.identity(),
            NativeExecutionError::ForeignEncoder,
        )
    }
    fn check_pipeline(&self, pipeline: &ComputePipeline) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )
    }
}

/// Reads one completed compute export through the test-only staging path.
///
/// The only accepted state input is the graph's recorded outgoing state.  This
/// deliberately prevents a fixture from repairing an incorrect graph export by
/// guessing `CopySource` or `Undefined` before the helper transition.
#[cfg(test)]
#[allow(
    dead_code,
    reason = "K01/K02 hardware fixtures consume this crate-private wrapper"
)]
pub(crate) fn readback_exported_buffer_for_test(
    device: &Device,
    exported: &fluxel_rendergraph::ExportedBuffer<ComputeBackend>,
) -> Result<Vec<u8>, NativeExecutionError> {
    if exported.physical.device_identity() != device.identity() {
        return Err(NativeExecutionError::ForeignResource);
    }
    match &exported.lease {
        ResourceLease::Buffer(lease)
            if lease.device_identity() == device.identity()
                && lease.identity() == exported.physical.identity() => {}
        _ => return Err(NativeExecutionError::ForeignResource),
    }
    crate::imp::readback_buffer_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.outgoing_state,
        exported.descriptor.size,
    )
    .map_err(NativeExecutionError::Recording)
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
    if !region.source_offset.is_multiple_of(4)
        || !region.destination_offset.is_multiple_of(4)
        || !region.size.is_multiple_of(4)
    {
        return Err(NativeExecutionError::InvalidTransfer(
            "buffer copy offsets and size must be 4-byte aligned",
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
    fn binding_creation_errors_preserve_contract_vs_backend_failure() {
        assert_eq!(
            compute_binding_error_kind(&ComputeCreateError::InvalidBindingRange),
            fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe
        );
        assert_eq!(
            compute_binding_error_kind(&ComputeCreateError::NativeFailure("descriptor".into())),
            fluxel_rendergraph::RecordingErrorKind::BackendObjectCreation
        );
    }

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
    fn compute_profile_adds_compute_without_mutating_copy_profile() {
        let copy = CopyBackend::portable_capabilities();
        let compute = ComputeBackend::portable_capabilities();
        assert!(!copy.queues[0].capabilities.compute);
        assert!(compute.queues[0].capabilities.compute);
        assert!(compute.queues[0].capabilities.copy);
        assert_eq!(
            compute.limits.max_compute_workgroups_per_dimension,
            [65_535; 3]
        );
    }

    #[test]
    fn fixed_compute_artifacts_have_stable_entries_workgroups_and_source_hash() {
        use crate::ComputeKernel;
        for (kernel, entry) in [
            (ComputeKernel::WrappingAdd, "wrapping_add"),
            (ComputeKernel::WrappingMultiply, "wrapping_multiply"),
        ] {
            assert_eq!(kernel.entry_point(), entry);
            assert_eq!(kernel.workgroup_size(), [64, 1, 1]);
            assert_ne!(kernel.source_hash(), 0);
            assert_eq!(
                kernel.source_hash(),
                super::super::resource::fnv1a64(kernel.wgsl_source().as_bytes())
            );
        }
    }

    #[test]
    fn dispatch_dimensions_fail_closed_before_native_recording() {
        let maximum = [4, 8, 16];
        assert!(valid_compute_dispatch([4, 8, 16], maximum));
        assert!(!valid_compute_dispatch([0, 1, 1], maximum));
        assert!(!valid_compute_dispatch([5, 1, 1], maximum));
        assert!(!valid_compute_dispatch([1, 9, 1], maximum));
        assert!(!valid_compute_dispatch([1, 1, 17], maximum));
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
    fn native_buffer_validation_rejects_zero_misaligned_overflow_and_out_of_bounds() {
        let descriptor = BufferDesc { size: 32 };
        for region in [
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 0,
                size: 0,
            },
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
