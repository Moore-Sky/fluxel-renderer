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
/// The renderer-owned binding identity for one acquired presentable image.
///
/// The RHI owns the corresponding one-shot acquisition token; this opaque id
/// only selects it for the fixed visible-frame graph.
#[cfg(windows)]
pub(super) fn surface_binding() -> fluxel_rendergraph::SurfaceBindingId {
    fluxel_rendergraph::SurfaceBindingId::new(0x0220_0004)
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
pub(super) fn vertex_color_binding() -> BufferBindingId {
    BufferBindingId::new(0x0280_0004)
}
pub(super) fn vertex_color_pipeline() -> RasterPipelineId {
    RasterPipelineId::new(0x0280_0001)
}
pub(super) fn vertex_color_bindings() -> BindingSetId {
    BindingSetId::new(0x0280_0001)
}
