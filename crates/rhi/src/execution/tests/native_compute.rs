//! Native compute witnesses.

use super::native_common::{NoObjects, Resources, TextureCopyData};
use super::native_raster::r01_oracle;
use super::native_raster::{ComputeCase, RasterBufferCase, add_compute};
use crate::*;
use fluxel_rendergraph::*;
struct X01ComputeData {
    pipeline: ComputePipelineId,
    bindings: BindingSetId,
    texture: TextureRead,
    output: BufferWrite,
}
struct X01CopyData {
    source: BufferRead,
    destination: BufferWrite,
}

pub(super) fn run_x01(backend: crate::Backend) {
    run_x01_case(backend, &build_x01(), "independent");
}

pub(super) fn build_x01() -> RasterBufferCase {
    let texture = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
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
    let bytes = BufferDesc { size: 256 };
    let raster_id = RasterPipelineId::new(403);
    let compute_id = ComputePipelineId::new(404);
    let bindings_id = BindingSetId::new(405);
    let mut graph = RenderGraph::new();
    let image = graph.create_texture("x01-raster-target", texture);
    let storage = graph.create_buffer("x01-packed", bytes);
    let copied = graph.create_buffer("x01-copy-destination", bytes);
    let raster = graph.add_raster_pass(
        "x01-raster",
        |pass| {
            let out = pass.color_attachment(
                image,
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
            (out, raster_id)
        },
        |commands, _, pipeline, _| {
            commands.set_pipeline(*pipeline)?;
            commands.draw(0..3, 0..1)
        },
    );
    let compute = graph.add_compute_pass(
        "x01-pack",
        |pass| {
            let sampled = pass.read_texture(
                &raster.output,
                fluxel_rendergraph::TextureReadUse::Sampled,
                TextureRange::Whole,
            );
            let (out, storage) = pass.write_buffer(
                storage,
                fluxel_rendergraph::BufferWriteUse::Storage,
                BufferRange::Whole,
                WriteCoverage::Full,
            );
            (
                out,
                X01ComputeData {
                    pipeline: compute_id,
                    bindings: bindings_id,
                    texture: sampled,
                    output: storage,
                },
            )
        },
        |commands, resolver, data, _| {
            commands.set_pipeline(data.pipeline)?;
            let bindings = resolver.resolve_bindings(
                data.bindings,
                &[
                    BindingResource::TextureRead(&data.texture),
                    BindingResource::BufferWrite(&data.output),
                ],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    let copy = graph.add_copy_pass(
        "x01-copy",
        |pass| {
            let source = pass.read_buffer(&compute.output, BufferRange::Whole);
            let (out, destination) =
                pass.write_buffer(copied, BufferRange::Whole, WriteCoverage::Full);
            (
                out,
                X01CopyData {
                    source,
                    destination,
                },
            )
        },
        |commands, _, data, _| {
            commands.copy_buffer(
                &data.source,
                &data.destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 256,
                },
            )
        },
    );
    let export = graph.export_buffer(
        copy.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())
        .unwrap()
        .graph;
    let expected: Vec<u8> = r01_oracle()
        .chunks_exact(4)
        .flat_map(|pixel| u32::from_le_bytes(pixel.try_into().unwrap()).to_le_bytes())
        .collect();
    RasterBufferCase {
        compiled,
        export,
        raster_pipeline: raster_id,
        compute_pipeline: compute_id,
        bindings: bindings_id,
        expected,
    }
}

pub(super) fn run_x01_case(backend: crate::Backend, case: &RasterBufferCase, mode: &str) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let mut provider = RasterObjectProvider::new(&device);
    provider
        .register_raster_pipeline(
            case.raster_pipeline,
            device
                .create_raster_pipeline(crate::RasterKernel::Triangle)
                .unwrap(),
        )
        .unwrap();
    provider
        .register_compute_pipeline(
            case.compute_pipeline,
            device
                .create_compute_pipeline(crate::ComputeKernel::TexturePackRgba8)
                .unwrap(),
        )
        .unwrap();
    provider
        .register_bindings(case.bindings, case.compute_pipeline)
        .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    let executor = fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            &case.compiled,
            case.compiled.instantiate_local(FrameInputs::new(())),
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
    let exported = frame.exports.buffer(case.export).unwrap();
    let outgoing_state = exported.outgoing_state;
    let descriptor = exported.descriptor;
    let actual = readback_raster_exported_buffer_for_test(&device, exported).unwrap();
    assert_eq!(
        actual,
        case.expected,
        "{backend:?} first difference: {:?}",
        actual.iter().zip(&case.expected).position(|(a, b)| a != b)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "X01/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let first_difference = actual
        .iter()
        .zip(&case.expected)
        .position(|(left, right)| left != right);
    eprintln!(
        "artifact case=X01 mode={mode} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label=X01-raster-compute-copy-v1; execution_plan={:?}; raster_artifact={:?}; compute_artifact={:?}; binding_layout=sampled rgba8 texture + one storage buffer; dispatch=[1,1,1]; input=Raster triangle 8x8 -> textureLoad row-major rgba8 pack -> buffer copy; resource=buffer descriptor={descriptor:?}; expected={:?}; actual={actual:?}; first_difference={first_difference:?}; outgoing_state={outgoing_state:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        case.compiled.execution_plan(),
        crate::RasterKernel::Triangle.portable_identity(),
        crate::ComputeKernel::TexturePackRgba8.portable_identity(),
        case.expected,
    );
}

pub(super) fn build_k01() -> ComputeCase {
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

pub(super) fn run_k02(backend: crate::Backend) {
    let case = build_k02();
    run_compute_case_bundle("K02", backend, &case);
}

pub(super) fn build_k02() -> ComputeCase {
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

pub(super) fn run_compute_case_bundle(case: &str, backend: crate::Backend, bundle: &ComputeCase) {
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

pub(super) fn run_buffer_case(
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

pub(super) fn run_c02(backend: crate::Backend) {
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
        |commands, _, data, _| commands.copy_texture(&data.source, &data.destination, data.region),
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
    let destination =
        vec![0xCD; (destination_desc.extent.width * destination_desc.extent.height * 4) as usize];
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

pub(super) fn hardware_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
