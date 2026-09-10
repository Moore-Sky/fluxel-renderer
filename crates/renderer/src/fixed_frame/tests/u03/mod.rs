//! Camera/material fixed-frame native conformance and lifecycle tests.

use super::*;
use crate::{BasicMaterial, Camera, Geometry, IndexedMeshUpload, IndexedMeshUploadStatus};
use fluxel_rhi::{Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test};
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
    let dx_capabilities = actual_raster_capabilities(&dx12);
    assert_eq!(dx_capabilities, actual_raster_capabilities(&vulkan));
    let graph = Arc::new(build_camera_graph(&dx_snapshot, [8, 8], &dx_capabilities).unwrap());
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
