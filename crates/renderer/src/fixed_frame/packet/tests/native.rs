//! Windows native conformance fixtures for ordered legacy packet draws.
//!
//! These tests exercise the public packet entry points against a real DX12 or
//! Vulkan device.  Fault injection distinguishes pre-raster release from
//! accepted-raster poison, while D01 and D02 compare one packet graph's exact
//! target bytes with independent CPU triangle oracles. D03 additionally
//! proves per-draw model placement while reusing one immutable mesh snapshot.

use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use fluxel_rendergraph::ResourceAccessState;
use fluxel_rhi::experimental::fixed_artifacts::RasterKernel;
use fluxel_rhi::{
    Backend, Device, DeviceOptions, Validation, readback_exported_raster_texture_for_test,
    test_support,
};

use super::super::{
    RenderPacket, RenderPacketFailure, RenderPacketReservationError, RenderPacketStartError,
    RenderPacketStatus, RenderPacketSubmission,
};
use crate::{
    BasicMaterial, Camera, DrawList, FixedFrameRenderer, Geometry, IndexedMeshSnapshot,
    IndexedMeshUpload, IndexedMeshUploadStatus, Mesh, ModelTransform,
};

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn d01_ordered_multi_draw_dx12() {
    run_oracle(Backend::Dx12, Case::Distinct);
}

#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn d01_ordered_multi_draw_vulkan() {
    run_oracle(Backend::Vulkan, Case::Distinct);
}

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn d02_later_overlapping_draw_wins_dx12() {
    run_oracle(Backend::Dx12, Case::Overlap);
}

#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn d02_later_overlapping_draw_wins_vulkan() {
    run_oracle(Backend::Vulkan, Case::Overlap);
}

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn d03_transformed_reused_snapshot_dx12() {
    run_d03(Backend::Dx12);
}

#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn d03_transformed_reused_snapshot_vulkan() {
    run_d03(Backend::Vulkan);
}

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn packet_build_contract_rejects_global_and_indexed_input_errors() {
    let _guard = crate::native_fixture_guard();
    let device = open(Backend::Dx12);
    let renderer = FixedFrameRenderer::new(device.clone());
    let (camera, meshes, snapshots) = fixture(&device, Case::Distinct);

    let empty = DrawList::new(&camera);
    assert!(matches!(
        renderer.lower_draw_list(&empty, &[], [8, 8]),
        Err(super::super::RenderPacketBuildError::EmptyDrawList)
    ));
    let mut one = DrawList::new(&camera);
    one.push(&meshes[0]);
    assert!(matches!(
        renderer.lower_draw_list(&one, &[], [8, 8]),
        Err(
            super::super::RenderPacketBuildError::SnapshotCountMismatch {
                draws: 1,
                snapshots: 0
            }
        )
    ));
    assert!(matches!(
        renderer.lower_draw_list(&one, &snapshots[..1], [0, 8]),
        Err(super::super::RenderPacketBuildError::InvalidExtent)
    ));

    let mismatched_mesh = Mesh::new(
        triangle(-0.5, 0.5),
        BasicMaterial::new([1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
    let mut mismatched = DrawList::new(&camera);
    mismatched.push(&mismatched_mesh);
    assert!(matches!(
        renderer.lower_draw_list(&mismatched, &snapshots[..1], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::PositionMetadataMismatch,
        })
    ));

    let index_mismatch_mesh = Mesh::new(
        Geometry::from_positions(meshes[0].geometry().positions().to_vec())
            .with_indices(vec![0, 2, 1])
            .unwrap(),
        BasicMaterial::default(),
    );
    let mut index_mismatch = DrawList::new(&camera);
    index_mismatch.push(&index_mismatch_mesh);
    assert!(matches!(
        renderer.lower_draw_list(&index_mismatch, &snapshots[..1], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::IndexMetadataMismatch,
        })
    ));

    let mut signed_zero_positions = meshes[0].geometry().positions().to_vec();
    signed_zero_positions[0][2] = -0.0;
    let signed_zero_mesh = Mesh::new(
        Geometry::from_positions(signed_zero_positions)
            .with_indices(meshes[0].geometry().indices().to_vec())
            .unwrap(),
        BasicMaterial::default(),
    );
    let mut signed_zero = DrawList::new(&camera);
    signed_zero.push(&signed_zero_mesh);
    assert!(matches!(
        renderer.lower_draw_list(&signed_zero, &snapshots[..1], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::PositionMetadataMismatch,
        })
    ));

    let foreign_device = open(Backend::Dx12);
    let foreign_snapshot = ready_snapshot(&foreign_device, meshes[0].geometry());
    assert!(matches!(
        renderer.lower_draw_list(&one, &[foreign_snapshot], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::ForeignSnapshotDevice,
        })
    ));
    let packet = renderer
        .lower_draw_list(&one, &snapshots[..1], [8, 8])
        .unwrap();
    let foreign_renderer = FixedFrameRenderer::new(foreign_device);
    assert!(matches!(
        foreign_renderer.submit_packet(packet),
        Err(RenderPacketStartError::ForeignPacketDevice)
    ));

    let sibling_renderer = FixedFrameRenderer::new(device);
    let packet = renderer
        .lower_draw_list(&one, &snapshots[..1], [8, 8])
        .unwrap();
    drop(sibling_renderer.submit_packet(packet).unwrap());

    // All inputs are finite, but the model product itself overflows.  This is
    // a draw-local build failure and must be reported before any submission.
    let overflow_camera = Camera::new(
        *Camera::default().view(),
        [
            [f32::MAX, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    let overflow_model = ModelTransform::from_column_major([
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    let mut overflow = DrawList::new(&overflow_camera);
    overflow.push_transformed(&meshes[0], overflow_model);
    assert!(matches!(
        renderer.lower_draw_list(&overflow, &snapshots[..1], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::ModelTransformProductNonFinite,
        })
    ));

    let outside_model = ModelTransform::from_column_major([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    let mut outside = DrawList::new(&camera);
    outside.push_transformed(&meshes[0], outside_model);
    assert!(matches!(
        renderer.lower_draw_list(&outside, &snapshots[..1], [8, 8]),
        Err(super::super::RenderPacketBuildError::Draw {
            index: 0,
            reason: super::super::RenderPacketDrawBuildError::ClipOutOfBounds,
        })
    ));
}

#[test]
#[ignore = "requires Windows DX12 and Vulkan devices with required validation"]
fn d01_d02_paired_backend_oracles() {
    let _guard = crate::native_fixture_guard();
    for case in [Case::Distinct, Case::Overlap] {
        let dx12_device = open(Backend::Dx12);
        let vulkan_device = open(Backend::Vulkan);
        let dx12_renderer = FixedFrameRenderer::new(dx12_device.clone());
        let vulkan_renderer = FixedFrameRenderer::new(vulkan_device.clone());
        assert_eq!(dx12_renderer.capabilities, vulkan_renderer.capabilities);
        let (dx12_camera, dx12_meshes, dx12_snapshots) = fixture(&dx12_device, case);
        let (vulkan_camera, vulkan_meshes, vulkan_snapshots) = fixture(&vulkan_device, case);
        let dx12_packet = packet(&dx12_renderer, &dx12_camera, &dx12_meshes, &dx12_snapshots);
        let vulkan_packet = packet(
            &vulkan_renderer,
            &vulkan_camera,
            &vulkan_meshes,
            &vulkan_snapshots,
        );
        let graph =
            super::super::graph::build_packet_graph(&dx12_packet, &dx12_renderer.capabilities)
                .unwrap();
        let vulkan_graph = graph.clone();
        assert!(
            Arc::ptr_eq(&graph.compiled, &vulkan_graph.compiled),
            "{case:?} paired fixtures must use one compiled graph allocation"
        );
        test_support::clear_validation_diagnostics(&dx12_device);
        let dx12 = observe_packet(
            &dx12_device,
            case,
            dx12_renderer
                .submit_packet_with_graph_for_test(dx12_packet, graph)
                .unwrap(),
        );
        test_support::clear_validation_diagnostics(&vulkan_device);
        let vulkan = observe_packet(
            &vulkan_device,
            case,
            vulkan_renderer
                .submit_packet_with_graph_for_test(vulkan_packet, vulkan_graph)
                .unwrap(),
        );
        assert_eq!(
            dx12.bytes, vulkan.bytes,
            "{case:?} DX12/Vulkan full readback"
        );
    }
}

#[test]
#[ignore = "requires Windows DX12 and Vulkan devices with required validation"]
fn d03_paired_backend_oracle_uses_one_compiled_graph() {
    let _guard = crate::native_fixture_guard();
    let dx12_device = open(Backend::Dx12);
    let vulkan_device = open(Backend::Vulkan);
    let dx12_renderer = FixedFrameRenderer::new(dx12_device.clone());
    let vulkan_renderer = FixedFrameRenderer::new(vulkan_device.clone());
    assert_eq!(dx12_renderer.capabilities, vulkan_renderer.capabilities);
    let dx12_fixture = d03_fixture(&dx12_device);
    let vulkan_fixture = d03_fixture(&vulkan_device);
    let dx12_packet = d03_packet(&dx12_renderer, &dx12_fixture);
    let vulkan_packet = d03_packet(&vulkan_renderer, &vulkan_fixture);
    let graph =
        super::super::graph::build_packet_graph(&dx12_packet, &dx12_renderer.capabilities).unwrap();
    let vulkan_graph = graph.clone();
    assert!(Arc::ptr_eq(&graph.compiled, &vulkan_graph.compiled));
    test_support::clear_validation_diagnostics(&dx12_device);
    let dx12 = observe_d03(
        &dx12_device,
        dx12_renderer
            .submit_packet_with_graph_for_test(dx12_packet, graph)
            .unwrap(),
    );
    test_support::clear_validation_diagnostics(&vulkan_device);
    let vulkan = observe_d03(
        &vulkan_device,
        vulkan_renderer
            .submit_packet_with_graph_for_test(vulkan_packet, vulkan_graph)
            .unwrap(),
    );
    assert_eq!(dx12.bytes, vulkan.bytes, "D03 DX12/Vulkan full readback");
}

#[test]
#[ignore = "requires a Windows DX12 device with test-support fault injection"]
fn packet_fault_boundaries_release_before_raster_and_poison_after_acceptance() {
    let _guard = crate::native_fixture_guard();
    let device = open(Backend::Dx12);
    let renderer = FixedFrameRenderer::new(device.clone());
    let (camera, meshes, snapshots) = fixture(&device, Case::Overlap);

    // Immutable generations admit concurrent readers. Dropping both packets
    // before raster acceptance releases their independent leases for a later
    // packet instead of poisoning the shared generation.
    let first = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    let second = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    drop(first);
    drop(second);
    drop(
        renderer
            .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
            .unwrap(),
    );

    // The first upload rejection occurs before any work is accepted, hence a
    // retry can reserve and start the same generation.
    test_support::inject_submit_rejected_after(0);
    let first = packet(&renderer, &camera, &meshes, &snapshots);
    assert!(matches!(
        renderer.submit_packet(first),
        Err(RenderPacketStartError::FirstUniformStart(_))
    ));
    let retry = packet(&renderer, &camera, &meshes, &snapshots);
    drop(renderer.submit_packet(retry).unwrap());

    // The second uniform rejection follows one accepted upload but still
    // predates raster acceptance, so it releases every deduplicated gate.
    test_support::inject_submit_rejected_after(1);
    let mut later_rejected = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    assert!(matches!(
        wait_terminal(&mut later_rejected),
        RenderPacketStatus::Failed(RenderPacketFailure::UniformStart { draw_index: 1, .. })
    ));
    drop(later_rejected);
    drop(
        renderer
            .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
            .unwrap(),
    );

    // Keep raster submission pending behind the shared executor. Dropping at
    // this point is still pre-accept and must leave the snapshots reusable.
    let mut ready = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    let executor_guard = renderer.executor.try_backend().unwrap();
    wait_until_busy(&mut ready);
    drop(ready);
    drop(executor_guard);
    drop(
        renderer
            .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
            .unwrap(),
    );

    // Once the raster submit was accepted, abandoning the packet has no proof
    // of outgoing states and must poison each unique generation.
    let mut accepted = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    wait_until_raster_accepted(&mut accepted);
    drop(accepted);
    assert!(matches!(
        renderer.submit_packet(packet(&renderer, &camera, &meshes, &snapshots)),
        Err(RenderPacketStartError::Reservation {
            cause: RenderPacketReservationError::Poisoned,
            ..
        })
    ));
}

fn run_oracle(backend: Backend, case: Case) {
    let _guard = crate::native_fixture_guard();
    let _ = run_oracle_locked(backend, case);
}

fn run_oracle_locked(backend: Backend, case: Case) -> OracleResult {
    let device = open(backend);
    test_support::clear_validation_diagnostics(&device);
    let renderer = FixedFrameRenderer::new(device.clone());
    let (camera, meshes, snapshots) = fixture(&device, case);
    let submission = renderer
        .submit_packet(packet(&renderer, &camera, &meshes, &snapshots))
        .unwrap();
    observe_packet(&device, case, submission)
}

fn run_d03(backend: Backend) {
    let _guard = crate::native_fixture_guard();
    let device = open(backend);
    let renderer = FixedFrameRenderer::new(device.clone());
    let fixture = d03_fixture(&device);
    test_support::clear_validation_diagnostics(&device);
    let submission = renderer
        .submit_packet(d03_packet(&renderer, &fixture))
        .unwrap();
    let _ = observe_d03(&device, submission);
}

struct D03Fixture {
    camera: Camera,
    mesh: Mesh,
    snapshots: [IndexedMeshSnapshot; 2],
    transforms: [ModelTransform; 2],
}

fn d03_fixture(device: &Device) -> D03Fixture {
    let oracle = d03_oracle_fixture();
    let mesh = Mesh::new(
        oracle.geometry.clone(),
        BasicMaterial::new([1.0, 0.0, 0.0, 1.0]).unwrap(),
    );
    let snapshot = ready_snapshot(device, mesh.geometry());
    D03Fixture {
        camera: oracle.camera,
        mesh,
        snapshots: [snapshot.clone(), snapshot],
        transforms: oracle.transforms,
    }
}

struct D03OracleFixture {
    camera: Camera,
    geometry: Geometry,
    transforms: [ModelTransform; 2],
}

fn d03_oracle_fixture() -> D03OracleFixture {
    let camera = Camera::new(
        [
            [1.0, 0.15, 0.0, 0.0],
            [0.10, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.05, -0.04, 0.0, 1.0],
        ],
        [
            [0.85, 0.0, 0.0, 0.0],
            [0.0, 0.90, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    let geometry =
        Geometry::from_positions(vec![[-0.6, -0.5, 0.0], [0.6, -0.5, 0.0], [-0.6, 0.5, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap();
    let transforms = [
        ModelTransform::from_column_major([
            [0.45, 0.0, 0.0, 0.0],
            [0.0, 0.50, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-0.55, 0.15, 0.0, 1.0],
        ])
        .unwrap(),
        ModelTransform::from_column_major([
            [0.50, 0.12, 0.0, 0.0],
            [0.08, 0.42, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.52, -0.18, 0.0, 1.0],
        ])
        .unwrap(),
    ];
    assert_ne!(
        multiply(
            transforms[0].world_from_model(),
            transforms[1].world_from_model()
        ),
        multiply(
            transforms[1].world_from_model(),
            transforms[0].world_from_model()
        ),
        "D03 placements must be noncommuting"
    );
    D03OracleFixture {
        camera,
        geometry,
        transforms,
    }
}

fn d03_packet(renderer: &FixedFrameRenderer, fixture: &D03Fixture) -> RenderPacket {
    let mut list = DrawList::new(&fixture.camera);
    list.push_transformed(&fixture.mesh, fixture.transforms[0]);
    list.push_transformed(&fixture.mesh, fixture.transforms[1]);
    let packet = renderer
        .lower_draw_list(&list, &fixture.snapshots, [8, 8])
        .unwrap();
    let projection_view = multiply(fixture.camera.projection(), fixture.camera.view());
    for (draw, transform) in packet.draws().iter().zip(fixture.transforms) {
        assert_eq!(
            draw.uniform().view_projection(),
            &multiply(&projection_view, transform.world_from_model()),
            "D03 packet uniforms must preserve DrawList insertion order"
        );
    }
    packet
}

fn observe_d03(device: &Device, mut submission: RenderPacketSubmission) -> OracleResult {
    let backend = device.hardware().backend;
    wait_complete(&mut submission);
    let frame = submission.completed.as_ref().unwrap();
    let graph = submission.packet_graph();
    assert_eq!(
        graph.snapshots.len(),
        1,
        "D03 imports one snapshot generation"
    );
    assert_eq!(
        graph.draws.len(),
        2,
        "D03 retains two independent draw uniforms"
    );
    for snapshot in &graph.snapshots {
        assert_eq!(
            frame
                .exports
                .buffer(snapshot.position_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            frame
                .exports
                .buffer(snapshot.index_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
    }
    let readback = readback_exported_raster_texture_for_test(
        device,
        frame
            .exports
            .texture(submission.target_export.unwrap())
            .unwrap(),
    )
    .unwrap();
    assert!(
        readback
            .padded
            .chunks_exact(readback.bytes_per_row as usize)
            .all(|row| row[8 * 4..].iter().all(|byte| *byte == 0))
    );
    let fixture = d03_oracle_fixture();
    let expected = d03_oracle(&fixture);
    assert!(
        expected
            .chunks_exact(4)
            .any(|pixel| pixel == [0, 0, 0, 255])
    );
    assert!(
        expected
            .chunks_exact(4)
            .any(|pixel| pixel == [255, 0, 0, 255])
    );
    assert_eq!(
        readback.tight,
        expected,
        "{backend:?} D03 first difference: {:?}",
        first_difference(&readback.tight, &expected)
    );
    let diagnostics = test_support::validation_diagnostics(device);
    assert!(diagnostics.is_empty(), "{backend:?} D03: {diagnostics:?}");
    artifact_d03(
        device,
        graph.compiled.execution_plan(),
        &fixture,
        &expected,
        &readback.tight,
        &diagnostics,
    );
    OracleResult {
        bytes: readback.tight,
    }
}

/// Independently applies column-major `projection * view * model`; it does
/// not share renderer uniform construction or serialization code.
fn d03_oracle(fixture: &D03OracleFixture) -> Vec<u8> {
    let mut out = vec![0; 8 * 8 * 4];
    for pixel in out.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0, 0, 0, 255]);
    }
    let projection_view = multiply(fixture.camera.projection(), fixture.camera.view());
    let mut occupied = [false; 8 * 8];
    for (draw_index, transform) in fixture.transforms.into_iter().enumerate() {
        let clip_from_model = multiply(&projection_view, transform.world_from_model());
        let positions = fixture.geometry.positions();
        let points = [positions[0], positions[1], positions[2]].map(|position| {
            let clip = multiply_vector(
                &clip_from_model,
                [position[0], position[1], position[2], 1.0],
            );
            (
                (clip[0] / clip[3] + 1.0) * 4.0,
                (1.0 - clip[1] / clip[3]) * 4.0,
            )
        });
        let mut draw = vec![0; 8 * 8 * 4];
        fill_points(&mut draw, points, [255, 0, 0, 255]);
        let coverage = draw
            .chunks_exact(4)
            .map(|pixel| pixel == [255, 0, 0, 255])
            .collect::<Vec<_>>();
        assert!(
            coverage.iter().any(|covered| *covered),
            "D03 draw {draw_index} must cover at least one sample"
        );
        assert!(
            coverage
                .iter()
                .zip(occupied)
                .all(|(covered, previous)| !covered || !previous),
            "D03 transformed coverage must remain disjoint"
        );
        for (pixel_index, covered) in coverage.into_iter().enumerate() {
            if covered {
                occupied[pixel_index] = true;
                out[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
        }
    }
    out
}

fn multiply(left: &[[f32; 4]; 4], right: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut output = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            output[column][row] = (0..4).map(|k| left[k][row] * right[column][k]).sum();
        }
    }
    output
}

fn multiply_vector(matrix: &[[f32; 4]; 4], vector: [f32; 4]) -> [f32; 4] {
    [0, 1, 2, 3].map(|row| {
        (0..4)
            .map(|column| matrix[column][row] * vector[column])
            .sum()
    })
}

fn fill_points(out: &mut [u8], points: [(f32, f32); 3], color: [u8; 4]) {
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let top_left = |a: (f32, f32), b: (f32, f32)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
    let winding = edge(points[0], points[1], points[2]);
    for y in 0..8 {
        for x in 0..8 {
            let sample = (x as f32 + 0.5, y as f32 + 0.5);
            let edges = [
                (points[0], points[1]),
                (points[1], points[2]),
                (points[2], points[0]),
            ];
            if edges.iter().all(|&(a, b)| {
                let value = edge(a, b, sample);
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
}

fn artifact_d03(
    device: &Device,
    plan: &fluxel_rendergraph::ExecutionPlan,
    fixture: &D03OracleFixture,
    expected: &[u8],
    actual: &[u8],
    diagnostics: &[String],
) {
    let commit = std::env::var("FLUXEL_TEST_COMMIT")
        .expect("D03 evidence requires FLUXEL_TEST_COMMIT at the exact tested SHA");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    println!(
        "artifact schema=fluxel-draw-packet-d03-v1; commit={commit}; backend={:?}; hardware={:?}; driver={}; camera={:?}; transforms={:?}; positions={:?}; draw_order=[0,1]; snapshot_imports=1; draw_uniforms=2; execution_plan={plan:?}; expected={expected:?}; actual={actual:?}; first_difference={:?}; snapshot_outgoing=CopyDestination; target_outgoing=CopySource; completion=Complete; diagnostics={diagnostics:?}",
        device.hardware().backend,
        device.hardware(),
        device.hardware().driver,
        fixture.camera,
        fixture.transforms,
        fixture.geometry.positions(),
        first_difference(actual, expected),
    );
}

fn observe_packet(
    device: &Device,
    case: Case,
    mut submission: RenderPacketSubmission,
) -> OracleResult {
    let backend = device.hardware().backend;
    wait_complete(&mut submission);

    let frame = submission.completed.as_ref().unwrap();
    let graph = submission.packet_graph();
    for snapshot in &graph.snapshots {
        assert_eq!(
            frame
                .exports
                .buffer(snapshot.position_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            frame
                .exports
                .buffer(snapshot.index_export)
                .unwrap()
                .outgoing_state,
            ResourceAccessState::CopyDestination
        );
    }
    let readback = readback_exported_raster_texture_for_test(
        device,
        frame
            .exports
            .texture(submission.target_export.unwrap())
            .unwrap(),
    )
    .unwrap();
    assert!(
        readback
            .padded
            .chunks_exact(readback.bytes_per_row as usize)
            .all(|row| row[8 * 4..].iter().all(|byte| *byte == 0))
    );
    let expected = oracle(case);
    assert_eq!(
        readback.tight,
        expected,
        "{backend:?} {case:?} first difference: {:?}",
        first_difference(&readback.tight, &expected)
    );
    let diagnostics = test_support::validation_diagnostics(device);
    assert!(
        diagnostics.is_empty(),
        "{backend:?} {case:?}: {diagnostics:?}"
    );
    artifact(
        device.hardware().backend,
        device,
        case,
        graph.compiled.execution_plan(),
        &expected,
        &readback.tight,
        &diagnostics,
    );
    OracleResult {
        bytes: readback.tight,
    }
}

fn artifact(
    backend: Backend,
    device: &Device,
    case: Case,
    plan: &fluxel_rendergraph::ExecutionPlan,
    expected: &[u8],
    actual: &[u8],
    diagnostics: &[String],
) {
    let commit = std::env::var("FLUXEL_TEST_COMMIT")
        .expect("packet evidence requires FLUXEL_TEST_COMMIT at the exact tested SHA");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    println!(
        "artifact schema=fluxel-draw-packet-v1; case={case:?}; commit={commit}; os={}; backend={backend:?}; hardware={:?}; driver={}; execution_plan={plan:?}; raster_identity={:?}; draw_order=[0,1]; expected={expected:?}; actual={actual:?}; first_difference={:?}; snapshot_outgoing=CopyDestination; target_outgoing=CopySource; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        device.hardware().driver,
        RasterKernel::IndexedPositionFloat32x3CameraMaterial.portable_identity(),
        first_difference(actual, expected),
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

fn fixture(device: &Device, case: Case) -> (Camera, Vec<Mesh>, Vec<IndexedMeshSnapshot>) {
    let camera = Camera::default();
    let specs = match case {
        Case::Distinct => [
            (triangle(-0.95, -0.15), [1.0, 0.0, 0.0, 1.0]),
            (triangle(0.15, 0.95), [0.0, 1.0, 0.0, 1.0]),
        ],
        Case::Overlap => [
            (triangle(-0.65, 0.65), [1.0, 0.0, 0.0, 1.0]),
            (triangle(-0.65, 0.65), [0.0, 1.0, 0.0, 1.0]),
        ],
    };
    let meshes = specs
        .iter()
        .map(|(geometry, color)| Mesh::new(geometry.clone(), BasicMaterial::new(*color).unwrap()))
        .collect::<Vec<_>>();
    let snapshots = match case {
        Case::Distinct => specs
            .iter()
            .map(|(geometry, _)| ready_snapshot(device, geometry))
            .collect(),
        // Repeating one immutable generation is deliberately legal.  The
        // packet must reserve and import it once while preserving two draws.
        Case::Overlap => {
            let snapshot = ready_snapshot(device, &specs[0].0);
            vec![snapshot.clone(), snapshot]
        }
    };
    (camera, meshes, snapshots)
}

fn packet(
    renderer: &FixedFrameRenderer,
    camera: &Camera,
    meshes: &[Mesh],
    snapshots: &[IndexedMeshSnapshot],
) -> RenderPacket {
    let mut list = DrawList::new(camera);
    for mesh in meshes {
        list.push(mesh);
    }
    renderer.lower_draw_list(&list, snapshots, [8, 8]).unwrap()
}

fn triangle(left: f32, right: f32) -> Geometry {
    Geometry::from_positions(vec![
        [left, -0.75, 0.0],
        [right, -0.75, 0.0],
        [left, 0.75, 0.0],
    ])
    .with_indices(vec![0, 1, 2])
    .unwrap()
}

fn ready_snapshot(device: &Device, geometry: &Geometry) -> IndexedMeshSnapshot {
    let mut upload = IndexedMeshUpload::begin(device, geometry).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            IndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
            IndexedMeshUploadStatus::Pending => assert!(Instant::now() < deadline),
            IndexedMeshUploadStatus::Failed(error) => panic!("packet fixture upload {error:?}"),
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_complete(submission: &mut RenderPacketSubmission) {
    let status = wait_terminal(submission);
    assert!(
        matches!(status, RenderPacketStatus::Complete(_)),
        "{status:?}"
    );
}

fn wait_terminal(submission: &mut RenderPacketSubmission) -> RenderPacketStatus {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            status @ RenderPacketStatus::Complete(_) | status @ RenderPacketStatus::Failed(_) => {
                return status;
            }
            RenderPacketStatus::Pending | RenderPacketStatus::Busy => {
                assert!(Instant::now() < deadline)
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_until_busy(submission: &mut RenderPacketSubmission) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            RenderPacketStatus::Busy => return,
            RenderPacketStatus::Pending => assert!(Instant::now() < deadline),
            status => panic!("expected pre-raster Busy, got {status:?}"),
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_until_raster_accepted(submission: &mut RenderPacketSubmission) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !submission.has_accepted_raster() {
        assert!(Instant::now() < deadline);
        assert!(matches!(
            submission.poll(),
            RenderPacketStatus::Pending | RenderPacketStatus::Busy
        ));
        thread::sleep(Duration::from_millis(1));
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Distinct,
    Overlap,
}

struct OracleResult {
    bytes: Vec<u8>,
}

fn oracle(case: Case) -> Vec<u8> {
    let mut out = vec![0; 8 * 8 * 4];
    for pixel in out.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0, 0, 0, 255]);
    }
    let draws = match case {
        Case::Distinct => [
            ([-0.95, -0.15], [255, 0, 0, 255]),
            ([0.15, 0.95], [0, 255, 0, 255]),
        ],
        Case::Overlap => [
            ([-0.65, 0.65], [255, 0, 0, 255]),
            ([-0.65, 0.65], [0, 255, 0, 255]),
        ],
    };
    for ([left, right], color) in draws {
        fill_triangle(&mut out, triangle(left, right).positions(), color);
    }
    out
}

fn fill_triangle(out: &mut [u8], positions: &[[f32; 3]], color: [u8; 4]) {
    let points = [
        ((positions[0][0] + 1.0) * 4.0, (1.0 - positions[0][1]) * 4.0),
        ((positions[1][0] + 1.0) * 4.0, (1.0 - positions[1][1]) * 4.0),
        ((positions[2][0] + 1.0) * 4.0, (1.0 - positions[2][1]) * 4.0),
    ];
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let top_left = |a: (f32, f32), b: (f32, f32)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
    let winding = edge(points[0], points[1], points[2]);
    for y in 0..8 {
        for x in 0..8 {
            let sample = (x as f32 + 0.5, y as f32 + 0.5);
            let edges = [
                (points[0], points[1]),
                (points[1], points[2]),
                (points[2], points[0]),
            ];
            if edges.iter().all(|&(a, b)| {
                let value = edge(a, b, sample);
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
}

fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(left, right)| left != right)
}
