//! Linear texture and indexed-snapshot contract bodies called by stable test wrappers.

use super::super::indexed::{IndexedMeshPayload, IndexedMeshSnapshotDebug};
use super::super::texture::{BaseColorTextureSnapshotDebug, rgba8_byte_len};
use super::super::*;
use crate::Geometry;

pub(super) fn rgba8_image_requires_nonzero_extent_and_exact_tight_length_contract() {
    assert_eq!(
        Rgba8Image::new([0, 1], vec![]).unwrap_err(),
        Rgba8ImageError::ZeroWidth
    );
    assert_eq!(
        Rgba8Image::new([1, 0], vec![]).unwrap_err(),
        Rgba8ImageError::ZeroHeight
    );
    assert_eq!(
        Rgba8Image::new([2, 3], vec![0; 23]).unwrap_err(),
        Rgba8ImageError::IncorrectByteLength {
            expected: 24,
            actual: 23,
        }
    );
    let image = Rgba8Image::new([2, 3], vec![7; 24]).unwrap();
    assert_eq!(image.extent(), [2, 3]);
    assert_eq!(image.pixels(), vec![7; 24]);
}

pub(super) fn rgba8_image_checked_length_handles_largest_constructible_dimensions_contract() {
    assert_eq!(
        rgba8_byte_len([u32::MAX, u32::MAX]),
        Err(Rgba8ImageError::ByteLengthOverflow)
    );
}

pub(super) fn base_color_texture_snapshot_debug_is_exact_safe_metadata_only_contract() {
    let debug = format!(
        "{:?}",
        BaseColorTextureSnapshotDebug {
            generation: 17,
            extent: [3, 2],
        }
    );
    assert_eq!(
        debug,
        "BaseColorTextureSnapshot { generation: 17, extent: [3, 2] }"
    );
    for forbidden in [
        "CopyDestination",
        "UploadedTexture",
        "usage",
        "Ready",
        "descriptor",
        "identity",
        "gate",
        "state",
    ] {
        assert!(
            !debug.contains(forbidden),
            "snapshot debug must not expose {forbidden}: {debug}"
        );
    }
}

pub(super) fn indexed_mesh_snapshot_debug_is_exact_safe_metadata_only_contract() {
    let debug = format!(
        "{:?}",
        IndexedMeshSnapshotDebug {
            generation: 23,
            position_count: 3,
            index_count: 3,
        }
    );
    assert_eq!(
        debug,
        "IndexedMeshSnapshot { generation: 23, position_count: 3, index_count: 3 }"
    );
    for forbidden in [
        "UploadedBuffer",
        "positions:",
        "indices:",
        "metadata",
        "usage",
        "descriptor",
        "identity",
        "gate",
        "state",
    ] {
        assert!(
            !debug.contains(forbidden),
            "snapshot debug must not expose {forbidden}: {debug}"
        );
    }
}

pub(super) fn mesh_snapshot_retains_original_cpu_metadata_contract() {
    let geometry = Geometry::from_positions(vec![[1.0, 2.0, 3.0]])
        .with_indices(vec![0])
        .unwrap();
    let payload = IndexedMeshPayload::from_geometry(&geometry).unwrap();
    assert_eq!(geometry.positions(), &[[1.0, 2.0, 3.0]]);
    assert_eq!(geometry.indices(), &[0]);
    assert_eq!(payload.position_count, 1);
    assert_eq!(payload.index_count, 1);
}
