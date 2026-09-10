//! Legacy fixed-color native conformance tests.

use super::*;
use crate::{Geometry, IndexedMeshUpload, IndexedMeshUploadStatus};
use fluxel_rhi::{Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test};
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
    let dx_capabilities = actual_raster_capabilities(&dx12);
    assert_eq!(dx_capabilities, actual_raster_capabilities(&vulkan));
    let graph = build_graph(&dx_snapshot, [8, 8], &dx_capabilities).unwrap();
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
    let capabilities = actual_raster_capabilities(&device);
    let graph = build_graph(&snapshot, [8, 8], &capabilities).unwrap();
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
                output[(y * 8 + x) * 4..(y * 8 + x + 1) * 4].copy_from_slice(&[48, 176, 112, 255]);
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
