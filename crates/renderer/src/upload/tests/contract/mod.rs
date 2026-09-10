//! Indexed mesh payload serialization and start-rejection contracts.

use super::*;

pub(crate) fn serializes_positions_as_exact_little_endian_bits_and_indices_in_order() {
    let geometry = Geometry::from_positions(vec![[f32::from_bits(0x8000_0000), 1.0, f32::NAN]])
        .with_indices(vec![0])
        .unwrap();
    let payload = IndexedMeshPayload::from_geometry(&geometry).unwrap();
    assert_eq!(
        payload.positions,
        [
            0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0xc0, 0x7f,
        ]
    );
    assert_eq!(payload.indices, [0, 0, 0, 0]);
}

pub(crate) fn payload_rejects_empty_or_non_indexed_geometry_before_upload() {
    assert_eq!(
        IndexedMeshPayload::from_geometry(&Geometry::from_positions(Vec::new())).unwrap_err(),
        IndexedMeshUploadStartError::EmptyPositions
    );
    assert_eq!(
        IndexedMeshPayload::from_geometry(&Geometry::from_positions(vec![[0.0; 3]])).unwrap_err(),
        IndexedMeshUploadStartError::NonIndexedGeometry
    );
}
