//! Geometry, payload, and snapshot contracts for textured and normal uploads.

use super::*;

pub(crate) fn textured_geometry() -> TexturedGeometry {
    TexturedGeometry::new(
        Geometry::from_positions(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .with_indices(vec![0, 1, 0])
            .unwrap(),
        vec![[0.0, 1.0], [1.0, 0.0]],
    )
    .unwrap()
}

pub(crate) fn normal_geometry() -> NormalGeometry {
    NormalGeometry::new(
        Geometry::from_positions(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .with_indices(vec![0, 1, 0])
            .unwrap(),
        vec![[3.0, 4.0, -0.0], [f32::MAX, f32::MIN_POSITIVE, -0.0]],
    )
    .unwrap()
}

pub(crate) fn normal_geometry_canonicalizes_with_fixed_f64_recipe_and_positive_zero_bytes() {
    let geometry = normal_geometry();
    assert_eq!(geometry.normals()[0][0].to_bits(), 0x3f19_999a);
    assert_eq!(geometry.normals()[0][1].to_bits(), 0x3f4c_cccd);
    assert_eq!(geometry.normals()[0][2].to_bits(), 0);
    assert_eq!(geometry.normals()[1], [1.0, 0.0, 0.0]);
    assert!(
        geometry
            .normals()
            .iter()
            .flat_map(|normal| normal.iter())
            .filter(|&&component| component == 0.0)
            .all(|component| component.to_bits() == 0)
    );

    let diagonal = canonicalize_normal([1.0, -1.0, -0.0]).unwrap();
    assert_eq!(diagonal[0].to_bits(), 0x3f35_04f3);
    assert_eq!(diagonal[1].to_bits(), 0xbf35_04f3);
    assert_eq!(diagonal[2].to_bits(), 0);
}

pub(crate) fn normal_geometry_rejects_closed_stream_and_normalization_contract_violations() {
    assert_eq!(
        NormalGeometry::new(Geometry::from_positions(Vec::new()), Vec::new()).unwrap_err(),
        NormalGeometryError::EmptyPositions
    );
    assert_eq!(
        NormalGeometry::new(Geometry::from_positions(vec![[0.0; 3]]), vec![[0.0; 3]]).unwrap_err(),
        NormalGeometryError::MissingIndices
    );
    assert_eq!(
        NormalGeometry::new(
            Geometry::from_positions(vec![[0.0; 3], [1.0; 3]])
                .with_indices(vec![0, 1])
                .unwrap(),
            vec![[0.0, 0.0, 1.0]; 2],
        )
        .unwrap_err(),
        NormalGeometryError::NonTriangleIndexCount { index_count: 2 }
    );
    let indexed = Geometry::from_positions(vec![[0.0; 3]])
        .with_indices(vec![0, 0, 0])
        .unwrap();
    assert_eq!(
        NormalGeometry::new(indexed.clone(), Vec::new()).unwrap_err(),
        NormalGeometryError::NormalCountMismatch {
            positions: 1,
            normals: 0,
        }
    );
    for normal in [
        [0.0, 0.0, 0.0],
        [f32::NAN, 0.0, 1.0],
        [f32::INFINITY, 0.0, 1.0],
    ] {
        assert_eq!(
            NormalGeometry::new(indexed.clone(), vec![normal]).unwrap_err(),
            NormalGeometryError::NonNormalizableNormal { vertex: 0 }
        );
    }
}

pub(crate) fn normal_payload_uses_only_canonical_metadata_bytes() {
    let payload = NormalMeshPayload::from_geometry(&normal_geometry()).unwrap();
    assert_eq!(payload.position_count, 2);
    assert_eq!(payload.index_count, 3);
    assert_eq!(payload.normals.len(), 24);
    assert_eq!(
        payload.normals,
        [
            0x9a, 0x99, 0x19, 0x3f, 0xcd, 0xcc, 0x4c, 0x3f, 0, 0, 0, 0, 0, 0, 0x80, 0x3f, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]
    );
}

pub(crate) fn normal_snapshot_debug_and_publication_are_opaque_and_atomic() {
    let debug = format!(
        "{:?}",
        NormalIndexedMeshSnapshotDebugProbe {
            generation: 11,
            position_count: 2,
            index_count: 3,
        }
    );
    assert_eq!(
        debug,
        "NormalIndexedMeshSnapshot { generation: 11, position_count: 2, index_count: 3 }"
    );
    for forbidden in ["UploadedBuffer", "normals", "metadata", "gate", "state"] {
        assert!(
            !debug.contains(forbidden),
            "snapshot debug leaked {forbidden}"
        );
    }

    use RetainedUploadState::{Missing, Pending, Ready};
    for publication in [
        NormalSnapshotPublication::new(Pending, Missing, Missing, true),
        NormalSnapshotPublication::new(Ready, Pending, Pending, false),
        NormalSnapshotPublication::new(Ready, Ready, Pending, false),
        NormalSnapshotPublication::new(Ready, Ready, Ready, true),
    ] {
        assert!(
            !publication.can_publish(),
            "partial/failed normal upload published: {publication:?}"
        );
    }
    assert!(NormalSnapshotPublication::new(Ready, Ready, Ready, false).can_publish());
}

pub(crate) fn normal_upload_failure_mapping_preserves_stream_role_and_acceptance_boundary() {
    let rejected = BufferUploadError::ForeignDevice;
    assert_eq!(
        normal_position_start_error(rejected.clone()),
        NormalIndexedMeshUploadStartError::PositionUpload(rejected.clone())
    );
    assert_eq!(
        normal_start_failure(NormalGeometryStream::Index, rejected.clone()),
        NormalIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(rejected.clone())
    );
    assert_eq!(
        normal_start_failure(NormalGeometryStream::Normal, rejected.clone()),
        NormalIndexedMeshUploadFailure::NormalStartAfterPositionAndIndexAccepted(rejected.clone())
    );
    for (stream, completion, observation, unknown) in [
        (
            NormalGeometryStream::Position,
            NormalIndexedMeshUploadFailure::PositionCompletion(CompletionFailure::DeviceLost),
            NormalIndexedMeshUploadFailure::PositionObservation(rejected.clone()),
            NormalIndexedMeshUploadFailure::PositionUnknownCompletion,
        ),
        (
            NormalGeometryStream::Index,
            NormalIndexedMeshUploadFailure::IndexCompletion(CompletionFailure::DeviceLost),
            NormalIndexedMeshUploadFailure::IndexObservation(rejected.clone()),
            NormalIndexedMeshUploadFailure::IndexUnknownCompletion,
        ),
        (
            NormalGeometryStream::Normal,
            NormalIndexedMeshUploadFailure::NormalCompletion(CompletionFailure::DeviceLost),
            NormalIndexedMeshUploadFailure::NormalObservation(rejected.clone()),
            NormalIndexedMeshUploadFailure::NormalUnknownCompletion,
        ),
    ] {
        assert_eq!(
            normal_completion_failure(stream, CompletionFailure::DeviceLost),
            completion
        );
        assert_eq!(
            normal_observation_failure(stream, rejected.clone()),
            observation
        );
        assert_eq!(normal_unknown_completion_failure(stream), unknown);
    }
}

#[cfg(windows)]
pub(crate) fn normal_fault_commit_helper_accepts_only_full_hex_sha_values() {
    assert!(normal_fault_commit_is_valid(&"0".repeat(40)));
    assert!(normal_fault_commit_is_valid(
        "0123456789abcdef0123456789abcdef01234567"
    ));
    assert!(!normal_fault_commit_is_valid("working-tree"));
    assert!(!normal_fault_commit_is_valid(&"0".repeat(39)));
    assert!(!normal_fault_commit_is_valid(&"g".repeat(40)));
}

pub(crate) fn srgba8_image_keeps_encoded_bytes_and_rejects_invalid_shape() {
    assert_eq!(
        Srgba8Image::new([0, 1], vec![]).unwrap_err(),
        Srgba8ImageError::ZeroWidth
    );
    assert_eq!(
        Srgba8Image::new([1, 0], vec![]).unwrap_err(),
        Srgba8ImageError::ZeroHeight
    );
    assert_eq!(
        Srgba8Image::new([2, 1], vec![0; 7]).unwrap_err(),
        Srgba8ImageError::IncorrectByteLength {
            expected: 8,
            actual: 7
        }
    );
    let encoded = vec![0, 12, 64, 255, 128, 200, 255, 17];
    let image = Srgba8Image::new([2, 1], encoded.clone()).unwrap();
    assert_eq!(image.extent(), [2, 1]);
    assert_eq!(image.pixels(), encoded);
}

pub(crate) fn srgba8_image_debug_and_error_are_explicit_about_encoded_domain() {
    let image = Srgba8Image::new([1, 1], vec![1, 2, 3, 4]).unwrap();
    assert_eq!(
        format!("{image:?}"),
        "Srgba8Image { extent: [1, 1], pixels: [1, 2, 3, 4] }"
    );
    assert_eq!(
        format!(
            "{}",
            Srgba8ImageError::IncorrectByteLength {
                expected: 4,
                actual: 3
            }
        ),
        "sRGBA8 image has 3 bytes; expected 4"
    );
}

pub(crate) fn textured_geometry_rejects_closed_stream_contract_violations() {
    assert_eq!(
        TexturedGeometry::new(Geometry::from_positions(Vec::new()), Vec::new()).unwrap_err(),
        TexturedGeometryError::EmptyPositions
    );
    assert_eq!(
        TexturedGeometry::new(Geometry::from_positions(vec![[0.0; 3]]), vec![[0.0, 0.0]],)
            .unwrap_err(),
        TexturedGeometryError::MissingIndices
    );
    assert_eq!(
        TexturedGeometry::new(
            Geometry::from_positions(vec![[0.0; 3]])
                .with_indices(vec![0])
                .unwrap(),
            Vec::new(),
        )
        .unwrap_err(),
        TexturedGeometryError::TextureCoordinateCountMismatch {
            positions: 1,
            texture_coordinates: 0,
        }
    );
    assert_eq!(
        TexturedGeometry::new(
            Geometry::from_positions(vec![[0.0; 3]])
                .with_indices(vec![0])
                .unwrap(),
            vec![[f32::NAN, 0.0]],
        )
        .unwrap_err(),
        TexturedGeometryError::NonFiniteTextureCoordinate {
            vertex: 0,
            component: 0,
        }
    );
}

pub(crate) fn textured_payload_has_exact_three_little_endian_streams() {
    let payload = TexturedMeshPayload::from_geometry(&textured_geometry()).unwrap();
    assert_eq!(payload.position_count, 2);
    assert_eq!(payload.index_count, 3);
    assert_eq!(payload.positions.len(), 24);
    assert_eq!(payload.indices, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        payload.texture_coordinates,
        [0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 0, 0,]
    );
}

pub(crate) fn textured_snapshot_debug_is_opaque_metadata_only() {
    let debug = format!(
        "{:?}",
        TexturedIndexedMeshSnapshotDebugProbe {
            generation: 9,
            position_count: 2,
            index_count: 3,
        }
    );
    assert_eq!(
        debug,
        "TexturedIndexedMeshSnapshot { generation: 9, position_count: 2, index_count: 3 }"
    );
    for forbidden in [
        "UploadedBuffer",
        "texture_coordinates",
        "metadata",
        "usage",
        "descriptor",
        "identity",
        "gate",
        "state",
    ] {
        assert!(
            !debug.contains(forbidden),
            "snapshot debug leaked {forbidden}"
        );
    }
}

pub(crate) fn textured_three_stream_publication_never_exposes_partial_or_failed_output() {
    use RetainedUploadState::{Missing, Pending, Ready};
    for publication in [
        TexturedSnapshotPublication::new(Pending, Missing, Missing, true),
        TexturedSnapshotPublication::new(Pending, Pending, Missing, true),
        TexturedSnapshotPublication::new(Ready, Pending, Pending, false),
        TexturedSnapshotPublication::new(Ready, Ready, Pending, false),
        TexturedSnapshotPublication::new(Ready, Ready, Ready, true),
    ] {
        assert!(
            !publication.can_publish(),
            "partial/failed upload published: {publication:?}"
        );
    }
    assert!(TexturedSnapshotPublication::new(Ready, Ready, Ready, false).can_publish());
}

pub(crate) fn textured_upload_failure_mapping_preserves_stream_and_acceptance_boundary() {
    let rejected = BufferUploadError::ForeignDevice;
    assert_eq!(
        textured_position_start_error(rejected.clone()),
        TexturedIndexedMeshUploadStartError::PositionUpload(rejected.clone())
    );
    assert_eq!(
        textured_start_failure(TexturedGeometryStream::Index, rejected.clone()),
        TexturedIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(rejected.clone())
    );
    assert_eq!(
        textured_start_failure(TexturedGeometryStream::TextureCoordinate, rejected.clone()),
        TexturedIndexedMeshUploadFailure::TextureCoordinateStartAfterPositionAndIndexAccepted(
            rejected.clone()
        )
    );
    for (stream, expected_completion, expected_observation, expected_unknown) in [
        (
            TexturedGeometryStream::Position,
            TexturedIndexedMeshUploadFailure::PositionCompletion(CompletionFailure::DeviceLost),
            TexturedIndexedMeshUploadFailure::PositionObservation(rejected.clone()),
            TexturedIndexedMeshUploadFailure::PositionUnknownCompletion,
        ),
        (
            TexturedGeometryStream::Index,
            TexturedIndexedMeshUploadFailure::IndexCompletion(CompletionFailure::DeviceLost),
            TexturedIndexedMeshUploadFailure::IndexObservation(rejected.clone()),
            TexturedIndexedMeshUploadFailure::IndexUnknownCompletion,
        ),
        (
            TexturedGeometryStream::TextureCoordinate,
            TexturedIndexedMeshUploadFailure::TextureCoordinateCompletion(
                CompletionFailure::DeviceLost,
            ),
            TexturedIndexedMeshUploadFailure::TextureCoordinateObservation(rejected.clone()),
            TexturedIndexedMeshUploadFailure::TextureCoordinateUnknownCompletion,
        ),
    ] {
        assert_eq!(
            textured_completion_failure(stream, CompletionFailure::DeviceLost),
            expected_completion
        );
        assert_eq!(
            textured_observation_failure(stream, rejected.clone()),
            expected_observation
        );
        assert_eq!(
            textured_unknown_completion_failure(stream),
            expected_unknown
        );
    }
}

fn normal_fault_commit_is_valid(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
}
