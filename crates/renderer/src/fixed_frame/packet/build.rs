//! Validates positional draw-to-snapshot mapping and builds owned packet data.

use super::*;
use crate::{DrawList, frame_uniform::FrameUniform};

impl super::super::FixedFrameRenderer {
    /// Copies an ordered draw list into an opaque device-affine packet.
    ///
    /// `snapshots[i]` must be the completed immutable upload of draw item `i`.
    /// Exact CPU metadata comparison prevents stale or guessed associations.
    pub fn lower_draw_list(
        &self,
        list: &DrawList<'_>,
        snapshots: &[IndexedMeshSnapshot],
        extent: [u32; 2],
    ) -> Result<RenderPacket, RenderPacketBuildError> {
        if list.is_empty() {
            return Err(RenderPacketBuildError::EmptyDrawList);
        }
        if list.len() != snapshots.len() {
            return Err(RenderPacketBuildError::SnapshotCountMismatch {
                draws: list.len(),
                snapshots: snapshots.len(),
            });
        }
        if extent[0] == 0 || extent[1] == 0 {
            return Err(RenderPacketBuildError::InvalidExtent);
        }

        let camera = list.camera().clone();
        let mut draws = Vec::with_capacity(list.len());
        for (index, (item, snapshot)) in list.iter().zip(snapshots).enumerate() {
            let reason = |reason| RenderPacketBuildError::Draw { index, reason };
            if snapshot.positions().buffer().device_identity() != self.device.identity()
                || snapshot.indices().buffer().device_identity() != self.device.identity()
            {
                return Err(reason(RenderPacketDrawBuildError::ForeignSnapshotDevice));
            }
            let geometry = item.mesh().geometry();
            if !positions_equal_by_bits(geometry.positions(), snapshot.position_metadata()) {
                return Err(reason(RenderPacketDrawBuildError::PositionMetadataMismatch));
            }
            if geometry.indices() != snapshot.index_metadata() {
                return Err(reason(RenderPacketDrawBuildError::IndexMetadataMismatch));
            }
            if snapshot.index_count() == 0 || !snapshot.index_count().is_multiple_of(3) {
                return Err(reason(RenderPacketDrawBuildError::InvalidIndexCount));
            }
            let uniform = FrameUniform::new(&camera, item.mesh().material())
                .map_err(|_| reason(RenderPacketDrawBuildError::InvalidCameraMaterial))?;
            super::super::renderer::validate_clip(
                snapshot.position_metadata(),
                snapshot.index_metadata(),
                uniform.view_projection(),
            )
            .map_err(|error| reason(map_clip_error(error)))?;
            draws.push(PacketDraw {
                snapshot: snapshot.clone(),
                uniform,
            });
        }

        Ok(RenderPacket {
            device: self.device.identity(),
            extent,
            _camera: camera,
            draws,
        })
    }
}

fn positions_equal_by_bits(left: &[[f32; 3]], right: &[[f32; 3]]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        })
}

fn map_clip_error(error: super::super::DrawStartError) -> RenderPacketDrawBuildError {
    use super::super::DrawStartError;
    match error {
        DrawStartError::NonFinitePosition => RenderPacketDrawBuildError::NonFinitePosition,
        DrawStartError::NonFiniteClipPosition => RenderPacketDrawBuildError::NonFiniteClipPosition,
        DrawStartError::ClipWNonPositive => RenderPacketDrawBuildError::ClipWNonPositive,
        DrawStartError::ClipOutOfBounds => RenderPacketDrawBuildError::ClipOutOfBounds,
        _ => RenderPacketDrawBuildError::InvalidIndexCount,
    }
}
