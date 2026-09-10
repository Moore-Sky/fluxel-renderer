//! Binds caller-owned fixed-frame snapshots to RenderGraph import slots.
//!
//! This adapter exposes only opaque RHI resources and their actual incoming states;
//! graph declaration remains in `graph`, while native handle access remains in RHI.
//! Optional resources and their states are constructed as pairs, so resolving an
//! optional import can rely on both values being present together.

use super::*;

pub(super) struct CameraResources {
    pub(in crate::fixed_frame) device: DeviceIdentity,
    pub(in crate::fixed_frame) positions: Buffer,
    pub(in crate::fixed_frame) indices: Buffer,
    pub(in crate::fixed_frame) uniform: Buffer,
    pub(in crate::fixed_frame) texture_coordinates: Option<Buffer>,
    pub(in crate::fixed_frame) normals: Option<Buffer>,
    pub(in crate::fixed_frame) colors: Option<Buffer>,
    pub(in crate::fixed_frame) texture: Option<Texture>,
    pub(in crate::fixed_frame) position_state: ResourceAccessState,
    pub(in crate::fixed_frame) index_state: ResourceAccessState,
    pub(in crate::fixed_frame) texture_coordinate_state: Option<ResourceAccessState>,
    pub(in crate::fixed_frame) normal_state: Option<ResourceAccessState>,
    pub(in crate::fixed_frame) color_state: Option<ResourceAccessState>,
    pub(in crate::fixed_frame) texture_state: Option<ResourceAccessState>,
}

impl FrameResourceProvider<RasterBackend> for CameraResources {
    fn texture(
        &self,
        id: fluxel_rendergraph::TextureBindingId,
    ) -> Result<fluxel_rendergraph::BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        let Some(texture) = self.texture.as_ref() else {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingTexture,
                "camera frame has no texture imports",
            ));
        };
        if id != texture_binding() {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingTexture,
                "unknown camera frame texture",
            ));
        }
        Ok(fluxel_rendergraph::BoundTexture {
            device: self.device,
            identity: texture.identity(),
            physical: texture.clone(),
            descriptor: texture.descriptor().texture,
            usage: texture.allowed_usage(),
            initial_state: self
                .texture_state
                .expect("texture state accompanies texture"),
            lease: texture.lease().into(),
        })
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let buffer = if id == position_binding() {
            &self.positions
        } else if id == index_binding() {
            &self.indices
        } else if id == uniform_binding() {
            &self.uniform
        } else if id == texture_coordinate_binding() {
            self.texture_coordinates.as_ref().ok_or_else(|| {
                missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "camera frame has no texture-coordinate import",
                )
            })?
        } else if id == normal_binding() {
            self.normals.as_ref().ok_or_else(|| {
                missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "camera frame has no normal import",
                )
            })?
        } else if id == vertex_color_binding() {
            self.colors.as_ref().ok_or_else(|| {
                missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "camera frame has no vertex-color import",
                )
            })?
        } else {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingBuffer,
                "unknown camera frame buffer",
            ));
        };
        let initial_state = if id == position_binding() {
            self.position_state
        } else if id == index_binding() {
            self.index_state
        } else if id == texture_coordinate_binding() {
            self.texture_coordinate_state
                .expect("texture-coordinate state accompanies its buffer")
        } else if id == normal_binding() {
            self.normal_state
                .expect("normal state accompanies its buffer")
        } else if id == vertex_color_binding() {
            self.color_state
                .expect("vertex-color state accompanies its buffer")
        } else {
            ResourceAccessState::CopyDestination
        };
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor: buffer.descriptor().buffer,
            usage: buffer.allowed_usage(),
            initial_state,
            lease: buffer.lease().into(),
        })
    }
}
