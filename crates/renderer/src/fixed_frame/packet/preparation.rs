//! Associates shared prepared draws with device-affine native snapshots.
//!
//! P*V*M, clip, material, and order semantics are already closed by
//! [`PreparedBasicScene`]. This layer only proves each supplied immutable GPU
//! snapshot is the matching device-local realization before packet ownership.

use fluxel_rendergraph::DeviceIdentity;

use crate::{IndexedMeshSnapshot, prepared_scene::PreparedBasicScene};

use super::{PacketDraw, RenderPacketBuildError, RenderPacketDrawBuildError};

pub(super) fn prepare_draws(
    device: DeviceIdentity,
    prepared: &PreparedBasicScene,
    snapshots: &[IndexedMeshSnapshot],
) -> Result<Vec<PacketDraw>, RenderPacketBuildError> {
    if prepared.draws().len() != snapshots.len() {
        return Err(RenderPacketBuildError::SnapshotCountMismatch {
            draws: prepared.draws().len(),
            snapshots: snapshots.len(),
        });
    }
    prepared
        .draws()
        .iter()
        .zip(snapshots)
        .enumerate()
        .map(|(index, (draw, snapshot))| {
            let reason = |reason| RenderPacketBuildError::Draw { index, reason };
            if snapshot.positions().buffer().device_identity() != device
                || snapshot.indices().buffer().device_identity() != device
            {
                return Err(reason(RenderPacketDrawBuildError::ForeignSnapshotDevice));
            }
            if !positions_equal_by_bits(draw.positions(), snapshot.position_metadata()) {
                return Err(reason(RenderPacketDrawBuildError::PositionMetadataMismatch));
            }
            if draw.indices() != snapshot.index_metadata() {
                return Err(reason(RenderPacketDrawBuildError::IndexMetadataMismatch));
            }
            Ok(PacketDraw {
                snapshot: snapshot.clone(),
                uniform: draw.uniform().clone(),
            })
        })
        .collect()
}

fn positions_equal_by_bits(left: &[[f32; 3]], right: &[[f32; 3]]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        })
}

#[cfg(test)]
mod tests;
