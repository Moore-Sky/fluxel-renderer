//! Unit contracts for the visible-frame lifecycle helpers.

use super::OneShotSurface;
use crate::fixed_frame::{RasterRecipe, visible::visible_camera_recipe};
use fluxel_rhi::adapter::fixed_artifacts::RasterKernel;

#[test]
fn acquired_permission_is_consumed_once() {
    let surface = OneShotSurface::new("acquired-image");
    assert_eq!(surface.take(), Some("acquired-image"));
    assert_eq!(surface.take(), None);
}

#[test]
fn visible_recipe_keeps_the_camera_material_kernel() {
    assert_eq!(
        visible_camera_recipe().kernel(),
        RasterKernel::IndexedPositionFloat32x3CameraMaterial
    );
    assert_eq!(visible_camera_recipe(), RasterRecipe::LEGACY_UNLIT);
}
