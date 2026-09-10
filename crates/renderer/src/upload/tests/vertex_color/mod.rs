//! Unit contracts for the closed vertex-color upload domain.

use fluxel_rendergraph::CompletionFailure;
use fluxel_rhi::BufferUploadError;

use super::super::*;
use crate::Geometry;

pub(crate) fn geometry_material_and_payload_preserve_closed_contract() {
    assert_eq!(
        VertexColorGeometry::new(Geometry::from_positions(Vec::new()), Vec::new()).unwrap_err(),
        VertexColorGeometryError::EmptyPositions
    );
    let unindexed = Geometry::from_positions(vec![[0.0; 3]]);
    assert_eq!(
        VertexColorGeometry::new(unindexed, vec![[0; 4]]).unwrap_err(),
        VertexColorGeometryError::MissingIndices
    );
    let non_triangle = Geometry::from_positions(vec![[0.0; 3]])
        .with_indices(vec![0, 0])
        .unwrap();
    assert_eq!(
        VertexColorGeometry::new(non_triangle, vec![[0; 4]]).unwrap_err(),
        VertexColorGeometryError::NonTriangleIndexCount { index_count: 2 }
    );
    let geometry = Geometry::from_positions(vec![[1.0, -0.0, 2.5], [0.0, 1.0, 0.0]])
        .with_indices(vec![0, 1, 0])
        .unwrap();
    assert_eq!(
        VertexColorGeometry::new(geometry.clone(), vec![[1, 2, 3, 4]]).unwrap_err(),
        VertexColorGeometryError::ColorCountMismatch {
            positions: 2,
            colors: 1
        }
    );
    let vertex_color =
        VertexColorGeometry::new(geometry, vec![[0, 127, 255, 1], [4, 3, 2, 1]]).unwrap();
    let payload = VertexColorMeshPayload::from_geometry(&vertex_color).unwrap();
    assert_eq!(payload.positions.len(), 24);
    assert_eq!(payload.colors, [0, 127, 255, 1, 4, 3, 2, 1]);
    assert_eq!(payload.indices, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(payload.position_count, 2);
    assert_eq!(payload.index_count, 3);

    assert_eq!(
        VertexColorMaterial::new([f32::NAN, 0.0, 0.0, 0.0]).unwrap_err(),
        VertexColorMaterialError::NonFinite { component: 0 }
    );
    assert_eq!(
        VertexColorMaterial::new([0.0, 1.1, 0.0, 0.0]).unwrap_err(),
        VertexColorMaterialError::OutOfRange { component: 1 }
    );
    assert_eq!(
        VertexColorMaterial::new([0.0, 1.0, 0.5, 1.0])
            .unwrap()
            .tint(),
        &[0.0, 1.0, 0.5, 1.0]
    );
}

pub(crate) fn publication_and_failure_mapping_preserve_three_stream_contract() {
    use crate::upload::shared::RetainedUploadState::{Missing, Pending, Ready};
    for publication in [
        VertexColorSnapshotPublication::new(Pending, Missing, Missing, false),
        VertexColorSnapshotPublication::new(Ready, Pending, Ready, false),
        VertexColorSnapshotPublication::new(Ready, Ready, Pending, false),
        VertexColorSnapshotPublication::new(Ready, Ready, Ready, true),
    ] {
        assert!(!publication.can_publish());
    }
    assert!(VertexColorSnapshotPublication::new(Ready, Ready, Ready, false).can_publish());

    let error = BufferUploadError::ForeignDevice;
    assert_eq!(
        completion_failure(
            VertexColorGeometryStream::Color,
            CompletionFailure::DeviceLost
        ),
        VertexColorIndexedMeshUploadFailure::ColorCompletion(CompletionFailure::DeviceLost)
    );
    assert_eq!(
        observation_failure(VertexColorGeometryStream::Index, error.clone()),
        VertexColorIndexedMeshUploadFailure::IndexObservation(error.clone())
    );
    assert_eq!(
        unknown_completion_failure(VertexColorGeometryStream::Position),
        VertexColorIndexedMeshUploadFailure::PositionUnknownCompletion
    );
}

#[cfg(windows)]
pub(crate) fn u09_vertex_color_three_upload_fault_contract_dx12() {
    run_three_upload_fault_contract(fluxel_rhi::Backend::Dx12);
}

#[cfg(windows)]
pub(crate) fn u09_vertex_color_three_upload_fault_contract_vulkan() {
    run_three_upload_fault_contract(fluxel_rhi::Backend::Vulkan);
}

/// Exercises the boundary where one or more uploads have already been accepted.
///
/// A failed later start must retain its accepted siblings, and an uncertain
/// completion must remain scoped to those retained native resources.  A later
/// clean upload proves that dropping the owning failed operation neither
/// publishes a partial snapshot nor poisons the shared device.
#[cfg(windows)]
fn run_three_upload_fault_contract(backend: fluxel_rhi::Backend) {
    use fluxel_rendergraph::CompletionFailure;
    use fluxel_rhi::{Device, DeviceOptions, Validation, test_support};

    let _guard = crate::native_fixture_guard();
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!("U09 vertex-color upload {backend:?} device open failed: {error}")
    });
    test_support::clear_validation_diagnostics(&device);
    let geometry = fixture_geometry();

    // No work was accepted, so this is the sole retryable start error.
    test_support::inject_submit_rejected_after(0);
    assert!(matches!(
        VertexColorIndexedMeshUpload::begin(&device, &geometry),
        Err(VertexColorIndexedMeshUploadStartError::PositionUpload(_))
    ));

    // A later rejection remains an owning failed operation: it cannot publish
    // a partial snapshot and dropping it must retain/release its earlier work.
    test_support::inject_submit_rejected_after(1);
    let mut color_rejected = VertexColorIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        color_rejected.poll(),
        VertexColorIndexedMeshUploadStatus::Failed(
            VertexColorIndexedMeshUploadFailure::ColorStartAfterPositionAccepted(_)
        )
    ));
    assert!(color_rejected.ready_snapshot().is_none());
    drop(color_rejected);

    test_support::inject_submit_rejected_after(2);
    let mut index_rejected = VertexColorIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        index_rejected.poll(),
        VertexColorIndexedMeshUploadStatus::Failed(
            VertexColorIndexedMeshUploadFailure::IndexStartAfterPositionAndColorAccepted(_)
        )
    ));
    assert!(index_rejected.ready_snapshot().is_none());
    drop(index_rejected);

    // Every accepted-unknown result is attributed to exactly its selected
    // stream.  The operation still owns all accepted siblings until drop.
    for (after, stream) in [
        (0, VertexColorGeometryStream::Position),
        (1, VertexColorGeometryStream::Color),
        (2, VertexColorGeometryStream::Index),
    ] {
        test_support::inject_submit_accepted_unknown_after(after);
        let mut upload = VertexColorIndexedMeshUpload::begin(&device, &geometry).unwrap();
        let status = poll_until_terminal(&mut upload, backend);
        assert!(matches!(
            (stream, status),
            (
                VertexColorGeometryStream::Position,
                VertexColorIndexedMeshUploadStatus::Failed(
                    VertexColorIndexedMeshUploadFailure::PositionCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            ) | (
                VertexColorGeometryStream::Color,
                VertexColorIndexedMeshUploadStatus::Failed(
                    VertexColorIndexedMeshUploadFailure::ColorCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            ) | (
                VertexColorGeometryStream::Index,
                VertexColorIndexedMeshUploadStatus::Failed(
                    VertexColorIndexedMeshUploadFailure::IndexCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            )
        ));
        assert!(upload.ready_snapshot().is_none());
        drop(upload);
    }

    // An accepted-but-unproven stream is quarantined with its operation; it
    // cannot make a later independent generation unusable.
    test_support::inject_submit_accepted_unknown_after(1);
    let dropped_partial = VertexColorIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(dropped_partial.ready_snapshot().is_none());
    drop(dropped_partial);

    let mut clean = VertexColorIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert_eq!(
        poll_until_terminal(&mut clean, backend),
        VertexColorIndexedMeshUploadStatus::Ready
    );
    assert!(clean.ready_snapshot().is_some());
    let diagnostics = test_support::validation_diagnostics(&device);
    assert!(diagnostics.is_empty(), "U09 {backend:?}: {diagnostics:?}");
}

#[cfg(windows)]
fn poll_until_terminal(
    upload: &mut VertexColorIndexedMeshUpload,
    backend: fluxel_rhi::Backend,
) -> VertexColorIndexedMeshUploadStatus {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            status @ VertexColorIndexedMeshUploadStatus::Ready
            | status @ VertexColorIndexedMeshUploadStatus::Failed(_) => return status,
            VertexColorIndexedMeshUploadStatus::Pending if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(1));
            }
            VertexColorIndexedMeshUploadStatus::Pending => {
                panic!("U09 vertex-color upload {backend:?} timed out")
            }
        }
    }
}

#[cfg(windows)]
fn fixture_geometry() -> VertexColorGeometry {
    VertexColorGeometry::new(
        Geometry::from_positions(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .with_indices(vec![0, 1, 0])
            .unwrap(),
        vec![[1, 2, 3, 4], [5, 6, 7, 8]],
    )
    .unwrap()
}
