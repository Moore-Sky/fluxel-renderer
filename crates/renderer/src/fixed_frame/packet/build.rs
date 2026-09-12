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
        let prepared = crate::prepared_scene::PreparedBasicScene::prepare(list)
            .map_err(map_preparation_error)?;
        let draws =
            super::preparation::prepare_draws(self.device.identity(), &prepared, snapshots)?;

        Ok(RenderPacket {
            device: self.device.identity(),
            extent,
            _camera: camera,
            draws,
        })
    }
}

fn map_preparation_error(
    error: crate::prepared_scene::PreparedBasicSceneError,
) -> RenderPacketBuildError {
    use crate::prepared_scene::{PreparedBasicDrawError, PreparedBasicSceneError};
    match error {
        PreparedBasicSceneError::Empty => RenderPacketBuildError::EmptyDrawList,
        PreparedBasicSceneError::PreparationGraph => RenderPacketBuildError::PreparationGraph,
        PreparedBasicSceneError::Draw { index, reason } => RenderPacketBuildError::Draw {
            index,
            reason: match reason {
                PreparedBasicDrawError::InvalidIndexCount => {
                    RenderPacketDrawBuildError::InvalidIndexCount
                }
                PreparedBasicDrawError::InvalidCameraMaterial => {
                    RenderPacketDrawBuildError::InvalidCameraMaterial
                }
                PreparedBasicDrawError::ModelTransformProductNonFinite => {
                    RenderPacketDrawBuildError::ModelTransformProductNonFinite
                }
                PreparedBasicDrawError::NonFinitePosition => {
                    RenderPacketDrawBuildError::NonFinitePosition
                }
                PreparedBasicDrawError::NonFiniteClipPosition => {
                    RenderPacketDrawBuildError::NonFiniteClipPosition
                }
                PreparedBasicDrawError::ClipWNonPositive => {
                    RenderPacketDrawBuildError::ClipWNonPositive
                }
                PreparedBasicDrawError::ClipOutOfBounds => {
                    RenderPacketDrawBuildError::ClipOutOfBounds
                }
            },
        },
    }
}
