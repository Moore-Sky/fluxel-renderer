//! The first fixed-light fixture intentionally shares U06's non-affine
//! camera triangle, but its oracle has no renderer helpers in its path.
use super::*;
use crate::{
    BasicMaterial, Camera, Geometry, NormalGeometry, NormalIndexedMeshSnapshot,
    NormalIndexedMeshUpload, NormalIndexedMeshUploadStatus,
};
use fluxel_rhi::{Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test};
use std::{
    thread,
    time::{Duration, Instant},
};

#[cfg(windows)]
#[test]
fn u08_cpu_oracle_freezes_perspective_and_lambert_counters() {
    for sample in [witness([0, 4]), witness([1, 4])] {
        assert!(sample.edge_margin > 0.01);
        assert!(sample.quantization_margin > 0.01);
        assert!(distance(sample.expected, sample.affine) > 2);
        assert!(distance(sample.expected, sample.unnormalized) > 2);
        assert!(distance(sample.expected, sample.wrong_light) > 2);
        assert!(distance(sample.expected, sample.normal_slot) > 2);
    }
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u08_normal_lambert_dx12() {
    run(Backend::Dx12);
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u08_normal_lambert_vulkan() {
    run(Backend::Vulkan);
}

#[cfg(windows)]
#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u08_normal_lambert_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    let capabilities = actual_raster_capabilities(&dx);
    assert_eq!(capabilities, actual_raster_capabilities(&vk));
    let dx_mesh = mesh(&dx);
    let vk_mesh = mesh(&vk);
    let graph =
        Arc::new(build_normal_lambert_camera_graph(&dx_mesh, [8, 8], &capabilities).unwrap());
    let mut first: Option<Vec<u8>> = None;
    for (backend, device, mesh) in [
        (Backend::Dx12, &dx, &dx_mesh),
        (Backend::Vulkan, &vk, &vk_mesh),
    ] {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let mut submission = renderer
            .start_normal_lambert(
                mesh,
                Arc::clone(&graph),
                FrameUniform::new(&camera(), &material()).unwrap(),
                mesh.reserve_for_draw().unwrap(),
            )
            .unwrap();
        let actual = complete(device, &mut submission);
        assert_oracle(backend, &actual.tight);
        if let Some(reference) = &first {
            assert_eq!(
                actual.tight,
                *reference,
                "U08 paired first difference {:?}",
                first_difference(&actual.tight, reference)
            );
        } else {
            first = Some(actual.tight.clone());
        }
        assert_validation(device);
        artifact(
            "paired-same-compiled-graph",
            backend,
            device,
            &graph,
            &submission,
            &actual,
        );
        run_plus_z_regression(backend, device, Some(Arc::clone(&graph)));
    }
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
fn geometry() -> NormalGeometry {
    // Exactly U06's primary geometry: w is (0.95, 1.05, 1.0), which
    // makes perspective and screen-linear normal interpolation distinct.
    let positions =
        Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap();
    NormalGeometry::new(
        positions,
        vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    )
    .unwrap()
}

#[cfg(windows)]
fn camera() -> Camera {
    Camera::new(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0.05, 0.1],
            [0., 0., 0.5, 1.],
        ],
    )
}

#[cfg(windows)]
fn material() -> BasicMaterial {
    BasicMaterial::new([0.72, 0.45, 0.28, 0.63])
}

#[cfg(windows)]
fn mesh(device: &Device) -> NormalIndexedMeshSnapshot {
    let mut upload = NormalIndexedMeshUpload::begin(device, &geometry()).unwrap();
    wait(|| match upload.poll() {
        NormalIndexedMeshUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        NormalIndexedMeshUploadStatus::Pending => None,
        NormalIndexedMeshUploadStatus::Failed(e) => panic!("U08 normal upload {e:?}"),
    })
}

#[cfg(windows)]
fn plus_z_geometry() -> NormalGeometry {
    let positions =
        Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap();
    NormalGeometry::new(positions, vec![[0.0, 0.0, 1.0]; 3]).unwrap()
}

#[cfg(windows)]
fn plus_z_mesh(device: &Device) -> NormalIndexedMeshSnapshot {
    let mut upload = NormalIndexedMeshUpload::begin(device, &plus_z_geometry()).unwrap();
    wait(|| match upload.poll() {
        NormalIndexedMeshUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        NormalIndexedMeshUploadStatus::Pending => None,
        NormalIndexedMeshUploadStatus::Failed(e) => panic!("U08 +Z normal upload {e:?}"),
    })
}

#[cfg(windows)]
fn assert_plus_z_unlit(backend: Backend, actual: &[u8]) {
    let expected = plus_z_unlit_expected();
    for pixel in [[0, 4], [1, 4]] {
        let got = rgba(actual, pixel);
        assert!(
            got.into_iter()
                .zip(expected)
                .all(|(a, e)| a.abs_diff(e) <= 1),
            "U08 {backend:?} +Z must reduce to unlit material at {pixel:?}: actual={got:?}; expected={expected:?}"
        );
    }
}

#[cfg(windows)]
fn plus_z_unlit_expected() -> [u8; 4] {
    material().base_color().map(quantize)
}

#[cfg(windows)]
fn run_plus_z_regression(
    backend: Backend,
    device: &Device,
    paired_graph: Option<Arc<CameraGraph>>,
) {
    let mesh = plus_z_mesh(device);
    let renderer = FixedFrameRenderer::new(device.clone());
    let mut submission = if let Some(graph) = paired_graph {
        renderer
            .start_normal_lambert(
                &mesh,
                graph,
                FrameUniform::new(&camera(), &material()).unwrap(),
                mesh.reserve_for_draw().unwrap(),
            )
            .unwrap()
    } else {
        renderer
            .draw_lambert(&mesh, &camera(), &material(), [8, 8])
            .unwrap()
    };
    let actual = complete(device, &mut submission);
    assert_plus_z_unlit(backend, &actual.tight);
    assert_validation(device);
    let graph = submission.graph.as_ref().unwrap();
    let exports = &submission.completed.as_ref().unwrap().exports;
    let states = [
        exports
            .buffer(graph.position_export)
            .unwrap()
            .outgoing_state,
        exports.buffer(graph.index_export).unwrap().outgoing_state,
        exports
            .buffer(graph.normal_export.unwrap())
            .unwrap()
            .outgoing_state,
    ];
    let target_state = exports
        .texture(submission.target_export.unwrap())
        .unwrap()
        .outgoing_state;
    assert_eq!(states, [ResourceAccessState::CopyDestination; 3]);
    assert_eq!(target_state, ResourceAccessState::CopySource);
    println!(
        "artifact schema=fluxel-u08-plus-z-v2; case=U08-plus-z-unlit-regression; commit={}; backend={backend:?}; capabilities={:?}; normal_bits={:?}; normal_le_bytes={:?}; identity={:?}; plan={:?}; expected={:?}; actual_witnesses={:?}; full_readback={:?}; position_index_normal_outgoing={states:?}; target_outgoing={target_state:?}; completion=Complete; diagnostics={:?}",
        exact_commit(),
        actual_raster_capabilities(device),
        canonical_normal_bits(plus_z_geometry().normals()),
        canonical_normal_le_bytes(plus_z_geometry().normals()),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert.portable_identity(),
        graph.compiled.execution_plan(),
        plus_z_unlit_expected(),
        [rgba(&actual.tight, [0, 4]), rgba(&actual.tight, [1, 4])],
        actual.tight,
        fluxel_rhi::test_support::validation_diagnostics(device),
    );
}

#[cfg(windows)]
fn canonical_normal_bits(normals: &[[f32; 3]]) -> Vec<u32> {
    normals
        .iter()
        .flat_map(|normal| normal.iter().map(|component| component.to_bits()))
        .collect()
}

#[cfg(windows)]
fn canonical_normal_le_bytes(normals: &[[f32; 3]]) -> Vec<u8> {
    normals
        .iter()
        .flat_map(|normal| {
            normal
                .iter()
                .flat_map(|component| component.to_bits().to_le_bytes())
        })
        .collect()
}

#[cfg(windows)]
fn wait<T>(mut poll: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(value) = poll() {
            return value;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
}

#[cfg(windows)]
fn run(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let mesh = mesh(&device);
    fluxel_rhi::test_support::clear_validation_diagnostics(&device);
    let mut submission = FixedFrameRenderer::new(device.clone())
        .draw_lambert(&mesh, &camera(), &material(), [8, 8])
        .unwrap();
    let actual = complete(&device, &mut submission);
    assert_oracle(backend, &actual.tight);
    assert_validation(&device);
    artifact(
        "independent-public-path",
        backend,
        &device,
        submission.graph.as_ref().unwrap(),
        &submission,
        &actual,
    );
    run_zero_fallback(backend, &device);
    run_plus_z_regression(backend, &device, None);
}

#[cfg(windows)]
fn complete(
    device: &Device,
    submission: &mut FixedFrameSubmission,
) -> fluxel_rhi::RasterTextureReadback {
    wait(|| match submission.poll() {
        FixedFrameStatus::Complete(_) => Some(()),
        FixedFrameStatus::Pending | FixedFrameStatus::Busy => None,
        FixedFrameStatus::Failed(e) => panic!("U08 completion {e:?}"),
    });
    readback_exported_raster_texture_for_test(
        device,
        submission
            .completed
            .as_ref()
            .unwrap()
            .exports
            .texture(submission.target_export.unwrap())
            .unwrap(),
    )
    .unwrap()
}

#[cfg(windows)]
fn zero_geometry() -> NormalGeometry {
    // Pixel [3,3]'s center is the exact screen-space centroid. All w are
    // one, so the three perspective weights are identical. The symmetric
    // pair shares identical-magnitude y bits; the fixed operation order
    // yields a bitwise zero interpolated vector before the shader branch.
    let positions = Geometry::from_positions(vec![
        [-0.625, 0.375, 0.0],
        [0.375, 0.375, 0.0],
        [-0.125, -0.375, 0.0],
    ])
    .with_indices(vec![0, 1, 2])
    .unwrap();
    let y = 0.866_025_4;
    NormalGeometry::new(
        positions,
        vec![[1.0, 0.0, 0.0], [-0.5, y, 0.0], [-0.5, -y, 0.0]],
    )
    .unwrap()
}

#[cfg(windows)]
fn zero_mesh(device: &Device) -> NormalIndexedMeshSnapshot {
    let mut upload = NormalIndexedMeshUpload::begin(device, &zero_geometry()).unwrap();
    wait(|| match upload.poll() {
        NormalIndexedMeshUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        NormalIndexedMeshUploadStatus::Pending => None,
        NormalIndexedMeshUploadStatus::Failed(e) => panic!("U08 zero normal upload {e:?}"),
    })
}

#[cfg(windows)]
fn zero_interpolated_normal() -> ([f32; 3], [f32; 3]) {
    let weights = [1.0_f32 / 3.0; 3];
    let normals = zero_geometry().normals().to_vec();
    let mut interpolated = [0.0; 3];
    for (axis, output) in interpolated.iter_mut().enumerate() {
        for vertex in 0..3 {
            *output += weights[vertex] * normals[vertex][axis];
        }
    }
    (weights, interpolated)
}

#[cfg(windows)]
fn run_zero_fallback(backend: Backend, device: &Device) {
    let mesh = zero_mesh(device);
    let mut submission = FixedFrameRenderer::new(device.clone())
        .draw_lambert(&mesh, &camera(), &material(), [8, 8])
        .unwrap();
    let actual = complete(device, &mut submission);
    let got = rgba(&actual.tight, [3, 3]);
    assert!(
        got[..3].iter().all(|channel| *channel <= 1),
        "U08 {backend:?} zero fallback actual={got:?}"
    );
    assert!(
        got[3].abs_diff(quantize(material().base_color()[3])) <= 1,
        "U08 {backend:?} zero fallback alpha={got:?}"
    );
    assert_validation(device);
    let graph = submission.graph.as_ref().unwrap();
    let exports = &submission.completed.as_ref().unwrap().exports;
    let states = [
        exports
            .buffer(graph.position_export)
            .unwrap()
            .outgoing_state,
        exports.buffer(graph.index_export).unwrap().outgoing_state,
        exports
            .buffer(graph.normal_export.unwrap())
            .unwrap()
            .outgoing_state,
    ];
    let target_state = exports
        .texture(submission.target_export.unwrap())
        .unwrap()
        .outgoing_state;
    assert_eq!(states, [ResourceAccessState::CopyDestination; 3]);
    assert_eq!(target_state, ResourceAccessState::CopySource);
    let (weights, interpolated) = zero_interpolated_normal();
    assert!(interpolated.iter().all(|value| value.to_bits() == 0));
    let identity =
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert.portable_identity();
    println!(
        "artifact schema=fluxel-u08-zero-v3; case=U08-zero-interpolation; commit={}; backend={backend:?}; capabilities={:?}; geometry={:?}; camera={:?}; normal_bits={:?}; normal_le_bytes={:?}; weights={weights:?}; interpolated_bits={:?}; identity={identity:?}; plan={:?}; pixel=[3,3]; expected=[0,0,0,{}]; actual={got:?}; full_readback={:?}; position_index_normal_outgoing={states:?}; target_outgoing={target_state:?}; completion=Complete; diagnostics={:?}",
        exact_commit(),
        actual_raster_capabilities(device),
        zero_geometry().geometry(),
        camera(),
        canonical_normal_bits(zero_geometry().normals()),
        canonical_normal_le_bytes(zero_geometry().normals()),
        interpolated.map(f32::to_bits),
        graph.compiled.execution_plan(),
        quantize(material().base_color()[3]),
        actual.tight,
        fluxel_rhi::test_support::validation_diagnostics(device)
    );
}

#[cfg(windows)]
#[allow(
    dead_code,
    reason = "the complete CPU reconstruction is retained through Witness Debug in U08 artifacts"
)]
#[derive(Debug)]
struct Witness {
    pixel: [u32; 2],
    screen_vertices: [(f32, f32); 3],
    clip_w: [f32; 3],
    raw_barycentric: [f32; 3],
    barycentric: [f32; 3],
    perspective_weights: [f32; 3],
    interpolated_normal: [f32; 3],
    interpolated_len2: f32,
    lambert_e: f32,
    affine_interpolated_normal: [f32; 3],
    position_slot_interpolated_normal: [f32; 3],
    affine_lambert: f32,
    unnormalized_lambert: f32,
    wrong_light_lambert: f32,
    normal_slot_lambert: f32,
    expected: [u8; 4],
    affine: [u8; 4],
    unnormalized: [u8; 4],
    wrong_light: [u8; 4],
    normal_slot: [u8; 4],
    edge_margin: f32,
    quantization_margin: f32,
}

#[cfg(windows)]
fn witness(pixel: [u32; 2]) -> Witness {
    let geometry = geometry();
    let positions = geometry.geometry().positions();
    let normals = geometry.normals();
    let vertex = |p: [f32; 3]| {
        let w = 1.0 + p[2] * 0.1;
        (w, ((p[0] / w + 1.) * 4., (1. - p[1] / w) * 4.))
    };
    let vertices = [
        vertex(positions[0]),
        vertex(positions[1]),
        vertex(positions[2]),
    ];
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let p = (pixel[0] as f32 + 0.5, pixel[1] as f32 + 0.5);
    let area = edge(vertices[0].1, vertices[1].1, vertices[2].1);
    let raw = [
        edge(vertices[1].1, vertices[2].1, p),
        edge(vertices[2].1, vertices[0].1, p),
        edge(vertices[0].1, vertices[1].1, p),
    ];
    assert!(
        raw.iter().all(|v| *v > 0.),
        "U08 witness must be strictly interior"
    );
    let bary = raw.map(|v| v / area);
    let denominator: f32 = (0..3).map(|i| bary[i] / vertices[i].0).sum();
    let perspective_weights = [0, 1, 2].map(|i| bary[i] / vertices[i].0 / denominator);
    let interpolate_perspective = |values: &[[f32; 3]]| {
        [0, 1, 2].map(|axis| {
            (0..3)
                .map(|i| bary[i] * values[i][axis] / vertices[i].0)
                .sum::<f32>()
                / denominator
        })
    };
    let perspective = interpolate_perspective(normals);
    // Counter only: this is the value the fragment would see if a native
    // implementation accidentally consumed position slot 0 as normal
    // slot 1. It is not a valid alternate renderer contract.
    let position_as_normal = interpolate_perspective(positions);
    let affine = [0, 1, 2].map(|a| (0..3).map(|i| bary[i] * normals[i][a]).sum::<f32>());
    let lambert = |n: [f32; 3], normalize: bool, light: f32| {
        let len2 = n.into_iter().map(|v| v * v).sum::<f32>();
        if len2 > 0. {
            (if normalize { n[2] / len2.sqrt() } else { n[2] }) * light
        } else {
            0.
        }
        .max(0.)
    };
    let shade = |value| {
        material().base_color().map(|c| {
            quantize(
                c * if c == material().base_color()[3] {
                    1.
                } else {
                    value
                },
            )
        })
    };
    let expected_lambert = lambert(perspective, true, 1.);
    let interpolated_len2 = perspective.into_iter().map(|value| value * value).sum();
    let expected_linear = material().base_color().map(|c| {
        c * if c == material().base_color()[3] {
            1.
        } else {
            expected_lambert
        }
    });
    let expected = shade(expected_lambert);
    let affine_lambert = lambert(affine, true, 1.);
    let unnormalized_lambert = lambert(perspective, false, 1.);
    let wrong_light_lambert = lambert(perspective, true, -1.);
    let normal_slot_lambert = lambert(position_as_normal, true, 1.);
    Witness {
        pixel,
        screen_vertices: vertices.map(|vertex| vertex.1),
        clip_w: vertices.map(|vertex| vertex.0),
        raw_barycentric: raw,
        barycentric: bary,
        perspective_weights,
        interpolated_normal: perspective,
        interpolated_len2,
        lambert_e: expected_lambert,
        affine_interpolated_normal: affine,
        position_slot_interpolated_normal: position_as_normal,
        affine_lambert,
        unnormalized_lambert,
        wrong_light_lambert,
        normal_slot_lambert,
        expected,
        affine: shade(affine_lambert),
        unnormalized: shade(unnormalized_lambert),
        wrong_light: shade(wrong_light_lambert),
        normal_slot: shade(normal_slot_lambert),
        edge_margin: bary.into_iter().map(f32::abs).fold(f32::INFINITY, f32::min),
        quantization_margin: expected_linear
            .into_iter()
            .map(|v| {
                let scaled = v.clamp(0., 1.) * 255.;
                (scaled - (scaled.floor() + 0.5)).abs()
            })
            .fold(f32::INFINITY, f32::min),
    }
}

#[cfg(windows)]
fn quantize(value: f32) -> u8 {
    (value.clamp(0., 1.) * 255. + 0.5).floor() as u8
}

#[cfg(windows)]
fn assert_oracle(backend: Backend, actual: &[u8]) {
    let witnesses = [witness([0, 4]), witness([1, 4])];
    for w in &witnesses {
        assert!(w.edge_margin > 0.01);
        assert!(w.quantization_margin > 0.01);
        let got: [u8; 4] = actual[(w.pixel[1] as usize * 8 + w.pixel[0] as usize) * 4..][..4]
            .try_into()
            .unwrap();
        assert!(
            got.into_iter()
                .zip(w.expected)
                .all(|(a, e)| a.abs_diff(e) <= 1),
            "U08 {backend:?} witness={w:?} actual={got:?}"
        );
        for counter in [w.affine, w.unnormalized, w.wrong_light, w.normal_slot] {
            assert!(
                distance(w.expected, counter) > 2,
                "U08 insufficient counter distinction witness={w:?}"
            );
        }
    }
}

#[cfg(windows)]
#[test]
fn u08_cpu_zero_fallback_is_bitwise_zero() {
    let (weights, interpolated) = zero_interpolated_normal();
    assert_eq!(weights, [1.0_f32 / 3.0; 3]);
    assert_eq!(interpolated.map(f32::to_bits), [0; 3]);
}

#[cfg(windows)]
#[test]
fn u08_cpu_plus_z_degenerates_to_unlit_material() {
    assert_eq!(plus_z_unlit_expected(), [184, 115, 71, 161]);
}

#[cfg(windows)]
fn distance(a: [u8; 4], b: [u8; 4]) -> u8 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| x.abs_diff(y))
        .max()
        .unwrap()
}

#[cfg(windows)]
fn assert_validation(device: &Device) {
    let unexpected: Vec<_> = fluxel_rhi::test_support::validation_diagnostics(device)
        .into_iter()
        .filter(|d| {
            !(d.contains("severity=D3D12_MESSAGE_SEVERITY(2)")
                && d.contains("category=D3D12_MESSAGE_CATEGORY(9)")
                && d.contains("id=D3D12_MESSAGE_ID(820)")
                && d.contains("The application did not pass any clear value to resource creation"))
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "U08 validation diagnostics {unexpected:?}"
    );
}

#[cfg(windows)]
fn artifact(
    mode: &str,
    backend: Backend,
    device: &Device,
    graph: &CameraGraph,
    submission: &FixedFrameSubmission,
    actual: &fluxel_rhi::RasterTextureReadback,
) {
    let exports = &submission.completed.as_ref().unwrap().exports;
    let states = [
        exports
            .buffer(graph.position_export)
            .unwrap()
            .outgoing_state,
        exports.buffer(graph.index_export).unwrap().outgoing_state,
        exports
            .buffer(graph.normal_export.unwrap())
            .unwrap()
            .outgoing_state,
    ];
    assert_eq!(states, [ResourceAccessState::CopyDestination; 3]);
    let target_state = exports
        .texture(submission.target_export.unwrap())
        .unwrap()
        .outgoing_state;
    assert_eq!(target_state, ResourceAccessState::CopySource);
    println!(
        "artifact schema=fluxel-u08-v3; case=U08; mode={mode}; commit={}; backend={backend:?}; hardware={:?}; driver={}; capabilities={:?}; normal_bits={:?}; normal_le_bytes={:?}; material={:?}; identity={:?}; plan={:?}; expected={:?}; actual={:?}; delta={:?}; full_readback={:?}; position_index_normal_outgoing={states:?}; target_outgoing={target_state:?}; completion=Complete; diagnostics={:?}",
        exact_commit(),
        device.hardware(),
        device.hardware().driver,
        actual_raster_capabilities(device),
        canonical_normal_bits(geometry().normals()),
        canonical_normal_le_bytes(geometry().normals()),
        material(),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert.portable_identity(),
        graph.compiled.execution_plan(),
        [witness([0, 4]), witness([1, 4])],
        [rgba(&actual.tight, [0, 4]), rgba(&actual.tight, [1, 4])],
        [
            delta(rgba(&actual.tight, [0, 4]), witness([0, 4]).expected),
            delta(rgba(&actual.tight, [1, 4]), witness([1, 4]).expected)
        ],
        actual.tight,
        fluxel_rhi::test_support::validation_diagnostics(device)
    );
}

#[cfg(windows)]
fn rgba(bytes: &[u8], p: [u32; 2]) -> [u8; 4] {
    bytes[(p[1] as usize * 8 + p[0] as usize) * 4..][..4]
        .try_into()
        .unwrap()
}
#[cfg(windows)]
fn delta(a: [u8; 4], b: [u8; 4]) -> [u8; 4] {
    [
        a[0].abs_diff(b[0]),
        a[1].abs_diff(b[1]),
        a[2].abs_diff(b[2]),
        a[3].abs_diff(b[3]),
    ]
}
#[cfg(windows)]
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}
#[cfg(windows)]
fn exact_commit() -> String {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect("U08 exact SHA required");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if commit.bytes().all(|byte| byte == b'0') {
        return "working-tree".into();
    }
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert!(head.status.success());
    assert_eq!(String::from_utf8(head.stdout).unwrap().trim(), commit);
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .unwrap();
    assert!(
        status.status.success() && status.stdout.is_empty(),
        "U08 exact-SHA requires clean worktree"
    );
    commit
}
