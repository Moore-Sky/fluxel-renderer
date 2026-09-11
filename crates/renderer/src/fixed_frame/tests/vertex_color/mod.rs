//! Vertex-color fixed-raster conformance fixture and independent CPU oracle.
//!
//! U09 keeps encoded `UNORM8x4` transport, perspective interpolation, tint,
//! quantization, and the three-stream reservation boundary observable.  Its
//! oracle intentionally has no production uniform or shader helpers in path.

use super::*;
use crate::{
    Camera, Geometry, VertexColorGeometry, VertexColorIndexedMeshSnapshot,
    VertexColorIndexedMeshUpload, VertexColorIndexedMeshUploadStatus, VertexColorMaterial,
};
use fluxel_rendergraph::ResourceAccessState;
use fluxel_rhi::experimental::fixed_artifacts::RasterKernel;
use fluxel_rhi::{
    Backend, Device, DeviceOptions, Validation, readback_exported_raster_texture_for_test,
    test_support,
};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

#[test]
fn u09_cpu_oracle_freezes_vertex_color_counters() {
    let oracle = oracle_image();
    let covered = (0u32..8)
        .flat_map(|y| (0u32..8).map(move |x| sample([x, y]).covered))
        .filter(|covered| *covered)
        .count();
    assert!(covered > 0 && covered < 64, "coverage={covered}");
    assert_eq!(rgba(&oracle, [2, 5]), [39, 121, 56, 112]);
    let witness = sample([2, 5]);
    assert!(witness.covered);
    assert!(witness.edge_margin > 1.0 / 64.0, "{witness:?}");
    assert!(witness.quantization_margin > 1.0 / 64.0, "{witness:?}");
    assert_ne!(witness.expected, witness.screen_linear);
    assert!(
        witness
            .flat_vertices
            .iter()
            .all(|flat| *flat != witness.expected),
        "{witness:?}"
    );
    assert_ne!(witness.expected, witness.no_tint);
    assert_ne!(witness.expected, witness.no_unorm_normalization);
    assert_ne!(witness.expected, witness.slot_zero_bytes);
    assert!(is_top_left((1.0, 1.0), (0.0, 1.0)));
    assert!(!is_top_left((0.0, 1.0), (1.0, 1.0)));
}

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u09_vertex_color_dx12() {
    run(Backend::Dx12);
}

#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u09_vertex_color_vulkan() {
    run(Backend::Vulkan);
}

#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u09_vertex_color_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    let capabilities = actual_raster_capabilities(&dx);
    assert_eq!(capabilities, actual_raster_capabilities(&vk));
    let dx_snapshot = snapshot(&dx);
    let vk_snapshot = snapshot(&vk);
    let graph =
        Arc::new(build_vertex_color_camera_graph(&dx_snapshot, [8, 8], &capabilities).unwrap());
    let vk_graph = Arc::clone(&graph);
    assert!(Arc::ptr_eq(&graph, &vk_graph));
    let mut reference = None;
    for (backend, device, mesh, graph) in [
        (Backend::Dx12, &dx, &dx_snapshot, Arc::clone(&graph)),
        (Backend::Vulkan, &vk, &vk_snapshot, vk_graph),
    ] {
        test_support::clear_validation_diagnostics(device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let uniform = FrameUniform::new_vertex_color(&camera(), &material()).unwrap();
        let mut submission = renderer
            .start_vertex_color(
                mesh,
                graph.clone(),
                uniform,
                mesh.reserve_for_draw().unwrap(),
            )
            .unwrap();
        let image = complete(device, &mut submission);
        assert_eq!(image.tight, oracle_image(), "U09 {backend:?} CPU oracle");
        if let Some(first) = &reference {
            assert_eq!(image.tight, *first, "U09 paired full readback");
        } else {
            reference = Some(image.tight.clone());
        }
        assert_exports(&submission, &graph);
        assert!(test_support::validation_diagnostics(device).is_empty());
        artifact(
            "paired-same-compiled-graph",
            backend,
            device,
            &graph,
            &submission,
            &image,
        );
    }
}

#[test]
#[ignore = "requires a Windows DX12 device with test-support fault injection"]
fn u09_vertex_color_preaccept_and_accepted_poison_contract() {
    let _guard = native_fixture_guard();
    let device = open(Backend::Dx12);
    let renderer = FixedFrameRenderer::new(device.clone());
    let camera = camera();
    let material = material();

    // A start rejection precedes every accepted operation, so all three
    // streams remain reusable by a cloned snapshot generation.
    let preaccept = snapshot(&device);
    let clone = preaccept.clone();
    test_support::inject_submit_rejected_once();
    assert!(matches!(
        renderer.draw_vertex_color(&preaccept, &camera, &material, [8, 8]),
        Err(DrawStartError::UniformStart(_))
    ));
    drop(
        renderer
            .draw_vertex_color(&clone, &camera, &material, [8, 8])
            .unwrap(),
    );

    // Unknown uniform completion still predates raster acceptance; it must
    // release rather than poison every shared stream gate.
    let observation = snapshot(&device);
    let clone = observation.clone();
    test_support::inject_submit_accepted_unknown_once();
    let mut pending = renderer
        .draw_vertex_color(&observation, &camera, &material, [8, 8])
        .unwrap();
    wait_terminal(&mut pending, |status| {
        matches!(
            status,
            FixedFrameStatus::Failed(FixedFrameFailure::UniformCompletion(_))
        )
    });
    drop(pending);
    drop(
        renderer
            .draw_vertex_color(&clone, &camera, &material, [8, 8])
            .unwrap(),
    );

    // The uniform submit is accepted by `draw_vertex_color`; installing this
    // fault before polling makes the later raster submit accepted-unknown.
    let accepted = snapshot(&device);
    let clone = accepted.clone();
    let mut pending = renderer
        .draw_vertex_color(&accepted, &camera, &material, [8, 8])
        .unwrap();
    test_support::inject_submit_accepted_unknown_once();
    wait_terminal(&mut pending, |status| {
        matches!(
            status,
            FixedFrameStatus::Failed(FixedFrameFailure::RasterCompletion(_))
        )
    });
    assert!(matches!(
        renderer.draw_vertex_color(&clone, &camera, &material, [8, 8]),
        Err(DrawStartError::SnapshotPoisoned)
    ));

    // Dropping an accepted frame is the same unproven-outgoing-state case.
    let early = snapshot(&device);
    let clone = early.clone();
    let mut pending = renderer
        .draw_vertex_color(&early, &camera, &material, [8, 8])
        .unwrap();
    wait_until_raster_accepted(&mut pending);
    drop(pending);
    assert!(matches!(
        renderer.draw_vertex_color(&clone, &camera, &material, [8, 8]),
        Err(DrawStartError::SnapshotPoisoned)
    ));
}

fn run(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let mesh = snapshot(&device);
    test_support::clear_validation_diagnostics(&device);
    let renderer = FixedFrameRenderer::new(device.clone());
    let mut submission = renderer
        .draw_vertex_color(&mesh, &camera(), &material(), [8, 8])
        .unwrap();
    let image = complete(&device, &mut submission);
    assert_eq!(
        image.tight,
        oracle_image(),
        "U09 {backend:?} full 8x8 CPU oracle"
    );
    let graph = submission.graph.as_ref().unwrap();
    assert_exports(&submission, graph);
    assert!(test_support::validation_diagnostics(&device).is_empty());
    artifact(
        "independent-public-path",
        backend,
        &device,
        graph,
        &submission,
        &image,
    );
}

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

fn geometry() -> VertexColorGeometry {
    VertexColorGeometry::new(
        Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
            .with_indices(vec![0, 2, 1])
            .unwrap(),
        vec![[154, 175, 26, 10], [147, 99, 210, 216], [12, 221, 22, 166]],
    )
    .unwrap()
}

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

fn material() -> VertexColorMaterial {
    VertexColorMaterial::new([0.36, 0.73, 0.69, 0.92]).unwrap()
}

fn snapshot(device: &Device) -> VertexColorIndexedMeshSnapshot {
    let mut upload = VertexColorIndexedMeshUpload::begin(device, &geometry()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            VertexColorIndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
            VertexColorIndexedMeshUploadStatus::Pending => assert!(Instant::now() < deadline),
            VertexColorIndexedMeshUploadStatus::Failed(error) => panic!("U09 upload {error:?}"),
        }
        thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone, Debug)]
struct Image {
    tight: Vec<u8>,
    padded: Vec<u8>,
    bytes_per_row: u32,
}

fn complete(device: &Device, submission: &mut FixedFrameSubmission) -> Image {
    wait_terminal(submission, |status| {
        matches!(status, FixedFrameStatus::Complete(_))
    });
    let frame = submission.completed.as_ref().unwrap();
    let export = frame
        .exports
        .texture(submission.target_export.unwrap())
        .unwrap();
    let image = readback_exported_raster_texture_for_test(device, export).unwrap();
    assert_eq!(image.bytes_per_row, 256);
    assert!(
        image
            .padded
            .chunks_exact(256)
            .all(|row| row[32..].iter().all(|byte| *byte == 0))
    );
    Image {
        tight: image.tight,
        padded: image.padded,
        bytes_per_row: image.bytes_per_row,
    }
}

fn assert_exports(submission: &FixedFrameSubmission, graph: &CameraGraph) {
    let exports = &submission.completed.as_ref().unwrap().exports;
    for export in [
        graph.position_export,
        graph.index_export,
        graph.vertex_color_export.unwrap(),
    ] {
        assert_eq!(
            exports.buffer(export).unwrap().outgoing_state,
            ResourceAccessState::CopyDestination
        );
    }
    assert_eq!(
        exports
            .texture(
                graph
                    .target_export
                    .expect("headless graph exports its target")
            )
            .unwrap()
            .outgoing_state,
        ResourceAccessState::CopySource
    );
}

fn wait_terminal<F: Fn(&FixedFrameStatus) -> bool>(
    submission: &mut FixedFrameSubmission,
    expected: F,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let status = submission.poll();
        if expected(&status) {
            return;
        }
        assert!(
            matches!(status, FixedFrameStatus::Pending | FixedFrameStatus::Busy),
            "U09 unexpected {status:?}"
        );
        assert!(Instant::now() < deadline, "U09 timed out");
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_until_raster_accepted(submission: &mut FixedFrameSubmission) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(submission.phase, CameraPhase::RasterAccepted(_)) {
        assert!(matches!(
            submission.poll(),
            FixedFrameStatus::Pending | FixedFrameStatus::Busy
        ));
        assert!(Instant::now() < deadline, "U09 raster acceptance timed out");
        thread::sleep(Duration::from_millis(1));
    }
}

fn rgba(bytes: &[u8], pixel: [u32; 2]) -> [u8; 4] {
    bytes[((pixel[1] * 8 + pixel[0]) * 4) as usize..][..4]
        .try_into()
        .unwrap()
}

#[derive(Debug)]
struct Sample {
    covered: bool,
    edge_margin: f32,
    quantization_margin: f32,
    expected: [u8; 4],
    screen_linear: [u8; 4],
    flat_vertices: [[u8; 4]; 3],
    no_tint: [u8; 4],
    no_unorm_normalization: [u8; 4],
    slot_zero_bytes: [u8; 4],
}

fn sample(pixel: [u32; 2]) -> Sample {
    let fixture = geometry();
    let positions = fixture.geometry().positions();
    let colors = fixture.colors();
    let project = |position: [f32; 3]| {
        let w = 1. + position[2] * 0.1;
        (
            w,
            ((position[0] / w + 1.) * 4., (1. - position[1] / w) * 4.),
        )
    };
    let v = [
        project(positions[0]),
        project(positions[2]),
        project(positions[1]),
    ];
    let c = [colors[0], colors[2], colors[1]];
    let point = (pixel[0] as f32 + 0.5, pixel[1] as f32 + 0.5);
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let area = edge(v[0].1, v[1].1, v[2].1);
    let pairs = [(v[1].1, v[2].1), (v[2].1, v[0].1), (v[0].1, v[1].1)];
    let raw = pairs.map(|(a, b)| edge(a, b, point));
    let covered = pairs.into_iter().zip(raw).all(|((a, b), value)| {
        if area > 0. {
            value > 0. || (value == 0. && is_top_left(a, b))
        } else {
            value < 0. || (value == 0. && is_top_left(b, a))
        }
    });
    if !covered {
        return Sample {
            covered,
            edge_margin: 0.,
            quantization_margin: 0.,
            expected: [0, 0, 0, 255],
            screen_linear: [0; 4],
            flat_vertices: [[0; 4]; 3],
            no_tint: [0; 4],
            no_unorm_normalization: [0; 4],
            slot_zero_bytes: [0; 4],
        };
    }
    let barycentric = raw.map(|value| value / area);
    let denominator: f32 = (0..3).map(|i| barycentric[i] / v[i].0).sum();
    let perspective = [0, 1, 2, 3].map(|channel| {
        (0..3)
            .map(|i| barycentric[i] * (c[i][channel] as f32 / 255.) / v[i].0)
            .sum::<f32>()
            / denominator
    });
    // Model binding slot 0 as the encoded-color stream. With the slot-1
    // stride of four bytes, indexed vertex fetches then read successive
    // four-byte chunks from the actual tightly packed position payload.
    let position_bytes: Vec<u8> = positions
        .iter()
        .flatten()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let wrong_slot_colors: [[u8; 4]; 3] = [0usize, 2, 1].map(|vertex| {
        position_bytes[vertex * 4..vertex * 4 + 4]
            .try_into()
            .unwrap()
    });
    let wrong_slot_perspective = [0, 1, 2, 3].map(|channel| {
        (0..3)
            .map(|i| barycentric[i] * (wrong_slot_colors[i][channel] as f32 / 255.) / v[i].0)
            .sum::<f32>()
            / denominator
    });
    let affine: [f32; 4] = [0, 1, 2, 3].map(|channel| {
        (0..3)
            .map(|i| barycentric[i] * c[i][channel] as f32 / 255.)
            .sum()
    });
    let tint = *material().tint();
    let encode = |value: [f32; 4]| value.map(|x| (x * 255.).round().clamp(0., 255.) as u8);
    let expected_linear = [0, 1, 2, 3].map(|i| perspective[i] * tint[i]);
    Sample {
        covered,
        edge_margin: barycentric
            .into_iter()
            .map(f32::abs)
            .fold(f32::INFINITY, f32::min),
        quantization_margin: expected_linear
            .into_iter()
            .map(|x| (x * 255. - (x * 255.).floor() - 0.5).abs())
            .fold(f32::INFINITY, f32::min),
        expected: encode(expected_linear),
        screen_linear: encode([0, 1, 2, 3].map(|i| affine[i] * tint[i])),
        flat_vertices: c
            .map(|color| encode([0, 1, 2, 3].map(|i| color[i] as f32 / 255. * tint[i]))),
        no_tint: encode(perspective),
        no_unorm_normalization: encode([0, 1, 2, 3].map(|i| perspective[i] * 255. * tint[i])),
        slot_zero_bytes: encode([0, 1, 2, 3].map(|i| wrong_slot_perspective[i] * tint[i])),
    }
}

fn is_top_left(a: (f32, f32), b: (f32, f32)) -> bool {
    b.1 < a.1 || (b.1 == a.1 && b.0 < a.0)
}

fn oracle_image() -> Vec<u8> {
    let mut output = vec![0; 8 * 8 * 4];
    for y in 0u32..8 {
        for x in 0u32..8 {
            output[((y * 8 + x) * 4) as usize..((y * 8 + x + 1) * 4) as usize]
                .copy_from_slice(&sample([x, y]).expected);
        }
    }
    output
}

fn artifact(
    mode: &str,
    backend: Backend,
    device: &Device,
    graph: &CameraGraph,
    submission: &FixedFrameSubmission,
    image: &Image,
) {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect("U09 exact SHA required");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .unwrap();
    assert!(
        status.status.success() && status.stdout.is_empty(),
        "U09 exact-SHA needs clean worktree"
    );
    let states = &submission.completed.as_ref().unwrap().exports;
    println!(
        "artifact schema=fluxel-u09-v1; case=U09-vertex-color; mode={mode}; commit={commit}; backend={backend:?}; hardware={:?}; driver={}; identity={:?}; plan={:?}; counters=perspective,top-left,tint,unorm8-round-nearest,slot1; witness={:?}; expected={:?}; actual={:?}; padded={:?}; pitch={}; outgoing=[{:?},{:?},{:?}]; target={:?}; completion=Complete; diagnostics={:?}",
        device.hardware(),
        device.hardware().driver,
        RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor.portable_identity(),
        graph.compiled.execution_plan(),
        sample([2, 5]),
        oracle_image(),
        image.tight,
        image.padded,
        image.bytes_per_row,
        states.buffer(graph.position_export).unwrap().outgoing_state,
        states.buffer(graph.index_export).unwrap().outgoing_state,
        states
            .buffer(graph.vertex_color_export.unwrap())
            .unwrap()
            .outgoing_state,
        states
            .texture(
                graph
                    .target_export
                    .expect("headless graph exports its target")
            )
            .unwrap()
            .outgoing_state,
        test_support::validation_diagnostics(device),
    );
}
