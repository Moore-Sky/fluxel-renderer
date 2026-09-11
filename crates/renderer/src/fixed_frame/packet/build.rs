//! Validates positional draw-to-snapshot mapping and builds owned packet data.

use super::*;
use crate::DrawList;

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
        let draws = super::preparation::prepare_draws(self.device.identity(), list, snapshots)?;

        Ok(RenderPacket {
            device: self.device.identity(),
            extent,
            _camera: camera,
            draws,
        })
    }
}

pub(super) fn map_clip_error(error: super::super::DrawStartError) -> RenderPacketDrawBuildError {
    use super::super::DrawStartError;
    match error {
        DrawStartError::NonFinitePosition => RenderPacketDrawBuildError::NonFinitePosition,
        DrawStartError::NonFiniteClipPosition => RenderPacketDrawBuildError::NonFiniteClipPosition,
        DrawStartError::ClipWNonPositive => RenderPacketDrawBuildError::ClipWNonPositive,
        DrawStartError::ClipOutOfBounds => RenderPacketDrawBuildError::ClipOutOfBounds,
        _ => RenderPacketDrawBuildError::InvalidIndexCount,
    }
}
