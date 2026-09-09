//! Safe owned-resource creation and lifetime contracts.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, PhysicalResourceIdentity, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};

use crate::{Backend, Device};

/// Host visibility policy for an owned resource.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MemoryPolicy {
    /// The resource is not exposed for host mapping by this API.
    #[default]
    DeviceOnly,
}

/// Description of an owned buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferDescriptor {
    /// Logical buffer shape.
    pub buffer: BufferDesc,
    /// Operations the resource must permit.
    pub usage: BufferUsage,
    /// Host visibility policy.
    pub memory: MemoryPolicy,
}

/// Description of an owned texture.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureDescriptor {
    /// Logical texture shape and format.
    pub texture: TextureDesc,
    /// Operations the resource must permit.
    pub usage: TextureUsage,
    /// Host visibility policy.
    pub memory: MemoryPolicy,
}

/// The resource kind involved in a creation error.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
}

/// A validated reason why a resource descriptor was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidResourceReason {
    /// A buffer has zero size.
    ZeroSize,
    /// No operation was requested.
    EmptyUsage,
    /// A texture extent contains zero.
    ZeroExtent,
    /// Mip-level count is zero or exceeds the extent's full chain.
    InvalidMipLevels,
    /// Array layers are zero or incompatible with the dimension.
    InvalidArrayLayers,
    /// Sample count is unsupported or incompatible with the descriptor.
    InvalidSampleCount,
    /// Extent components are incompatible with the texture dimension.
    InvalidDimension,
    /// This resource slice supports only two-dimensional textures.
    UnsupportedDimension,
    /// The resource exceeds a limit reported by the selected device.
    ExceedsDeviceLimit,
    /// The requested operation cannot be used with this resource shape or format.
    IncompatibleUsage,
    /// Presentation is reserved for acquired surface images.
    PresentRequiresSurface,
}

/// Why an owned resource could not be created.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResourceCreateError {
    /// The safe descriptor contract was invalid.
    InvalidDescriptor {
        /// Resource kind.
        resource: ResourceKind,
        /// Stable rejection reason.
        reason: InvalidResourceReason,
    },
    /// The native backend failed after portable validation succeeded.
    NativeFailure {
        /// Backend performing the operation.
        backend: Backend,
        /// Native diagnostic captured at the private boundary.
        reason: String,
    },
}

impl fmt::Display for ResourceCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor { resource, reason } => {
                write!(f, "invalid {resource:?} descriptor: {reason:?}")
            }
            Self::NativeFailure { backend, reason } => {
                write!(f, "{backend:?} resource creation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for ResourceCreateError {}

struct BufferShared {
    _native: crate::imp::OwnedBuffer,
    descriptor: BufferDescriptor,
    allowed_usage: BufferUsage,
    identity: PhysicalResourceIdentity,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// One opaque owned native buffer.
pub struct Buffer(Arc<BufferShared>);

/// A cloneable lifetime token for a buffer.
#[derive(Clone)]
pub struct BufferLease(Arc<BufferShared>);

struct TextureShared {
    _native: crate::imp::OwnedTexture,
    descriptor: TextureDescriptor,
    allowed_usage: TextureUsage,
    identity: PhysicalResourceIdentity,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// One opaque owned native texture.
pub struct Texture(Arc<TextureShared>);

/// A cloneable lifetime token for a texture.
#[derive(Clone)]
pub struct TextureLease(Arc<TextureShared>);

macro_rules! resource_accessors {
    ($resource:ident, $lease:ident, $shared:ident, $desc:ty, $usage:ty) => {
        impl $resource {
            /// Returns the descriptor validated at creation.
            pub fn descriptor(&self) -> $desc {
                self.0.descriptor
            }
            /// Returns operations proven by the final native creation facts.
            pub fn allowed_usage(&self) -> $usage {
                self.0.allowed_usage
            }
            /// Returns this physical generation's opaque identity.
            pub fn identity(&self) -> PhysicalResourceIdentity {
                self.0.identity
            }
            /// Returns the owning device identity.
            pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
                self.0.device
            }
            /// Acquires a token that keeps the native object and device alive.
            pub fn lease(&self) -> $lease {
                $lease(Arc::clone(&self.0))
            }
        }
        impl fmt::Debug for $resource {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($resource))
                    .field("descriptor", &self.0.descriptor)
                    .field("allowed_usage", &self.0.allowed_usage)
                    .field("identity", &self.0.identity)
                    .finish_non_exhaustive()
            }
        }
        impl fmt::Debug for $lease {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($lease))
                    .field(&self.0.identity)
                    .finish()
            }
        }
    };
}

resource_accessors!(
    Buffer,
    BufferLease,
    BufferShared,
    BufferDescriptor,
    BufferUsage
);
resource_accessors!(
    Texture,
    TextureLease,
    TextureShared,
    TextureDescriptor,
    TextureUsage
);

impl Device {
    /// Creates a native buffer after validating and lowering its portable contract.
    pub fn create_buffer(
        &self,
        descriptor: BufferDescriptor,
    ) -> Result<Buffer, ResourceCreateError> {
        validate_buffer(descriptor)?;
        let (native, allowed_usage) = crate::imp::create_buffer(&self.inner, descriptor)?;
        Ok(Buffer(Arc::new(BufferShared {
            _native: native,
            descriptor,
            allowed_usage,
            identity: PhysicalResourceIdentity::new(crate::next_identity()),
            device: self.identity,
        })))
    }

    /// Creates a native texture after validating and lowering its portable contract.
    pub fn create_texture(
        &self,
        descriptor: TextureDescriptor,
    ) -> Result<Texture, ResourceCreateError> {
        validate_texture(descriptor)?;
        let extent = descriptor.texture.extent;
        if descriptor.texture.dimension == TextureDimension::D2
            && (extent.width > self.capabilities.max_texture_dimension_2d
                || extent.height > self.capabilities.max_texture_dimension_2d)
        {
            return Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::ExceedsDeviceLimit,
            ));
        }
        let (native, allowed_usage) = crate::imp::create_texture(&self.inner, descriptor)?;
        Ok(Texture(Arc::new(TextureShared {
            _native: native,
            descriptor,
            allowed_usage,
            identity: PhysicalResourceIdentity::new(crate::next_identity()),
            device: self.identity,
        })))
    }
}

fn invalid(resource: ResourceKind, reason: InvalidResourceReason) -> ResourceCreateError {
    ResourceCreateError::InvalidDescriptor { resource, reason }
}

fn validate_buffer(desc: BufferDescriptor) -> Result<(), ResourceCreateError> {
    if desc.buffer.size == 0 {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::ZeroSize,
        ));
    }
    if !buffer_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    Ok(())
}

fn buffer_has_usage(usage: BufferUsage) -> bool {
    [
        BufferUsageKind::Uniform,
        BufferUsageKind::StorageRead,
        BufferUsageKind::StorageWrite,
        BufferUsageKind::Vertex,
        BufferUsageKind::Index,
        BufferUsageKind::Indirect,
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

fn texture_has_usage(usage: TextureUsage) -> bool {
    [
        TextureUsageKind::Sampled,
        TextureUsageKind::StorageRead,
        TextureUsageKind::StorageWrite,
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::DepthStencilAttachment,
        TextureUsageKind::CopySource,
        TextureUsageKind::CopyDestination,
        TextureUsageKind::Present,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

fn validate_texture(desc: TextureDescriptor) -> Result<(), ResourceCreateError> {
    let d = desc.texture;
    if !texture_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    if desc.usage.contains(TextureUsageKind::Present) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::PresentRequiresSurface,
        ));
    }
    if d.extent.width == 0 || d.extent.height == 0 || d.extent.depth == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::ZeroExtent,
        ));
    }
    if d.array_layers == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidArrayLayers,
        ));
    }
    if d.dimension != TextureDimension::D2 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::UnsupportedDimension,
        ));
    }
    let dimensions_valid = match d.dimension {
        TextureDimension::D1 => false,
        TextureDimension::D2 => d.extent.depth == 1,
        TextureDimension::D3 => false,
        _ => false,
    };
    if !dimensions_valid {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidDimension,
        ));
    }
    let max_dimension = d.extent.width.max(d.extent.height).max(d.extent.depth);
    let max_mips = u32::BITS - max_dimension.leading_zeros();
    if d.mip_levels == 0 || d.mip_levels > max_mips {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidMipLevels,
        ));
    }
    if !matches!(d.sample_count, 1 | 2 | 4 | 8 | 16)
        || (d.sample_count > 1
            && (d.dimension != TextureDimension::D2 || d.mip_levels != 1 || d.array_layers != 1))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidSampleCount,
        ));
    }
    let depth = d.format == TextureFormat::Depth32Float;
    let color_attachment = desc.usage.contains(TextureUsageKind::ColorAttachment);
    let depth_attachment = desc
        .usage
        .contains(TextureUsageKind::DepthStencilAttachment);
    let storage = desc.usage.contains(TextureUsageKind::StorageRead)
        || desc.usage.contains(TextureUsageKind::StorageWrite);
    let copy = desc.usage.contains(TextureUsageKind::CopySource)
        || desc.usage.contains(TextureUsageKind::CopyDestination);
    if (depth && (color_attachment || storage))
        || (!depth && depth_attachment)
        || (d.sample_count > 1 && (storage || copy))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::IncompatibleUsage,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::{Extent3d, TextureDimension};

    #[test]
    fn rejects_zero_buffer_and_empty_usage() {
        let empty = BufferDescriptor {
            buffer: BufferDesc { size: 0 },
            usage: BufferUsage::empty(),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_buffer(empty),
            Err(invalid(
                ResourceKind::Buffer,
                InvalidResourceReason::ZeroSize
            ))
        );
        assert_eq!(
            validate_buffer(BufferDescriptor {
                buffer: BufferDesc { size: 4 },
                ..empty
            }),
            Err(invalid(
                ResourceKind::Buffer,
                InvalidResourceReason::EmptyUsage
            ))
        );
    }

    #[test]
    fn rejects_surface_and_incompatible_texture_contracts() {
        let base = TextureDescriptor {
            texture: TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
            },
            usage: TextureUsage::empty().with(TextureUsageKind::Present),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_texture(base),
            Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::PresentRequiresSurface
            ))
        );
        let depth_color = TextureDescriptor {
            texture: TextureDesc {
                format: TextureFormat::Depth32Float,
                ..base.texture
            },
            usage: TextureUsage::empty().with(TextureUsageKind::ColorAttachment),
            ..base
        };
        assert_eq!(
            validate_texture(depth_color),
            Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::IncompatibleUsage
            ))
        );
    }
}
