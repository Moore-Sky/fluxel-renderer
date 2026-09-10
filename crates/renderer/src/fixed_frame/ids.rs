//! Stable private graph identifiers for the closed raster recipes.

use super::*;

pub(super) fn position_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0001)
}
pub(super) fn index_binding() -> BufferBindingId {
    BufferBindingId::new(0x0210_0002)
}
#[cfg(all(test, windows))]
pub(super) fn fixed_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0210_0001)
}
pub(super) fn uniform_binding() -> BufferBindingId {
    BufferBindingId::new(0x0220_0003)
}
pub(super) fn camera_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0220_0001)
}
pub(super) fn camera_bindings() -> BindingSetId {
    BindingSetId::new(0x0220_0001)
}
pub(super) fn texture_binding() -> TextureBindingId {
    TextureBindingId::new(0x0230_0004)
}
pub(super) fn textured_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0230_0001)
}
pub(super) fn textured_bindings() -> BindingSetId {
    BindingSetId::new(0x0230_0001)
}
pub(super) fn texture_coordinate_binding() -> BufferBindingId {
    BufferBindingId::new(0x0240_0003)
}
pub(super) fn normal_binding() -> BufferBindingId {
    BufferBindingId::new(0x0270_0004)
}
pub(super) fn normal_lambert_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0270_0001)
}
pub(super) fn normal_lambert_bindings() -> BindingSetId {
    BindingSetId::new(0x0270_0001)
}
