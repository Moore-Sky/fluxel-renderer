//! Validates portable resource descriptors and declared subresource ranges.

use super::super::super::*;
use super::super::initialization::{buffer_bounds, texture_keys};

pub(super) fn validate_range(
    resource: &ResourceDecl,
    range: DeclRange,
) -> Result<(), &'static str> {
    match (resource.kind, range) {
        (ResourceKind::Buffer(d), DeclRange::Buffer(BufferRange::Whole)) => {
            if d.size > 0 {
                Ok(())
            } else {
                Err("zero-sized buffer")
            }
        }
        (ResourceKind::Buffer(d), DeclRange::Buffer(BufferRange::Bytes { offset, size })) => {
            if size == 0 || offset.checked_add(size).is_none_or(|e| e > d.size) {
                Err("buffer range is empty or out of bounds")
            } else {
                Ok(())
            }
        }
        (ResourceKind::Texture(d), DeclRange::Texture(TextureRange::Whole)) => valid_tex_desc(d),
        (
            ResourceKind::Texture(d),
            DeclRange::Texture(TextureRange::Subresources {
                base_mip_level,
                mip_level_count,
                base_array_layer,
                array_layer_count,
                aspect,
            }),
        ) => {
            valid_tex_desc(d)?;
            if mip_level_count == 0
                || array_layer_count == 0
                || base_mip_level
                    .checked_add(mip_level_count)
                    .is_none_or(|e| e > d.mip_levels)
                || base_array_layer
                    .checked_add(array_layer_count)
                    .is_none_or(|e| e > d.array_layers)
            {
                return Err("texture range is empty or out of bounds");
            }
            let depth = d.format == TextureFormat::Depth32Float;
            match aspect {
                TextureAspect::All => Ok(()),
                TextureAspect::Color if !depth => Ok(()),
                TextureAspect::Depth if depth => Ok(()),
                _ => Err("texture aspect is incompatible with format"),
            }
        }
        _ => Err("range kind mismatch"),
    }
}
pub(super) fn valid_tex_desc(d: crate::rhi::TextureDesc) -> Result<(), &'static str> {
    if d.extent.width == 0
        || d.extent.height == 0
        || d.extent.depth == 0
        || d.mip_levels == 0
        || d.array_layers == 0
        || d.sample_count == 0
    {
        return Err("texture descriptor contains a zero value");
    }
    match d.dimension {
        TextureDimension::D1 if d.extent.height != 1 || d.extent.depth != 1 => {
            return Err("D1 textures require height and depth of one");
        }
        TextureDimension::D2 if d.extent.depth != 1 => {
            return Err("D2 textures require depth of one");
        }
        TextureDimension::D3 if d.array_layers != 1 => {
            return Err("D3 textures cannot declare array layers");
        }
        _ => {}
    }
    let largest = d.extent.width.max(d.extent.height).max(d.extent.depth);
    let max_mips = u32::BITS - largest.leading_zeros();
    if d.mip_levels > max_mips {
        return Err("texture mip count exceeds its extent");
    }
    if !d.sample_count.is_power_of_two()
        || (d.sample_count > 1 && (d.dimension != TextureDimension::D2 || d.mip_levels != 1))
    {
        return Err("multisample texture descriptor is not portable");
    }
    Ok(())
}
pub(in crate::compile) fn ranges_overlap(
    resource: &ResourceDecl,
    a: DeclRange,
    b: DeclRange,
) -> bool {
    match (a, b) {
        (DeclRange::Buffer(_), DeclRange::Buffer(_)) => {
            let (a0, a1) = buffer_bounds(resource, a);
            let (b0, b1) = buffer_bounds(resource, b);
            a0 < b1 && b0 < a1
        }
        (DeclRange::Texture(_), DeclRange::Texture(_)) => {
            let keys: HashSet<_> = texture_keys(resource, a).into_iter().collect();
            texture_keys(resource, b)
                .into_iter()
                .any(|key| keys.contains(&key))
        }
        _ => false,
    }
}
