//! Validates graph ownership, descriptors, access shapes, and same-pass conflicts.

use super::super::*;
use super::initialization::{buffer_bounds, texture_keys};

pub(in crate::compile) fn validate_references<F>(
    graph: &RenderGraph<F>,
    resources: &HashMap<ResourceId, &ResourceDecl>,
    pass_ids: &HashSet<PassId>,
) -> Result<(), CompileError> {
    for resource in &graph.resources {
        match resource.kind {
            ResourceKind::Texture(desc) => valid_tex_desc(desc).map_err(|detail| {
                err(
                    CompileErrorKind::InvalidSubresourceRange,
                    Vec::new(),
                    Some(resource.id),
                    detail,
                    None,
                )
            })?,
            ResourceKind::Buffer(desc) if desc.size == 0 => {
                return Err(err(
                    CompileErrorKind::InvalidSubresourceRange,
                    Vec::new(),
                    Some(resource.id),
                    "zero-sized buffer descriptor",
                    None,
                ));
            }
            ResourceKind::Buffer(_) => {}
        }
        match resource.origin {
            ResourceOrigin::TextureImport(_, contract)
                if !texture_boundary_state(
                    contract.descriptor.format,
                    contract.initial_state,
                    true,
                ) =>
            {
                return Err(err(
                    CompileErrorKind::MissingImportContract,
                    Vec::new(),
                    Some(resource.id),
                    "texture import state is incompatible with its format",
                    None,
                ));
            }
            ResourceOrigin::BufferImport(_, contract)
                if !buffer_boundary_state(contract.initial_state, true) =>
            {
                return Err(err(
                    CompileErrorKind::MissingImportContract,
                    Vec::new(),
                    Some(resource.id),
                    "buffer import state is incompatible with a buffer",
                    None,
                ));
            }
            ResourceOrigin::Surface(_, contract)
                if contract.descriptor.dimension != TextureDimension::D2
                    || contract.descriptor.mip_levels != 1
                    || contract.descriptor.array_layers != 1
                    || contract.descriptor.sample_count != 1 =>
            {
                return Err(err(
                    CompileErrorKind::InvalidSubresourceRange,
                    Vec::new(),
                    Some(resource.id),
                    "surface descriptor must be a single-sampled, single-mip, single-layer D2 texture",
                    None,
                ));
            }
            _ => {}
        }
    }
    for pass in &graph.passes {
        let produced: HashSet<_> = pass
            .accesses
            .iter()
            .filter_map(|access| {
                access
                    .output_version
                    .map(|version| (access.resource, version))
            })
            .collect();
        if let Some(access) = pass
            .accesses
            .iter()
            .find(|access| produced.contains(&(access.resource, access.input_version)))
        {
            return Err(err(
                CompileErrorKind::ConflictingAccess,
                vec![pass.id],
                Some(access.resource),
                "a pass cannot consume a successor version that it produces",
                None,
            ));
        }
        for access in &pass.accesses {
            let Some(resource) = resources.get(&access.resource) else {
                return Err(err(
                    CompileErrorKind::StaleOrForeignVersion,
                    vec![pass.id],
                    Some(access.resource),
                    "resource belongs to another graph",
                    None,
                ));
            };
            validate_range(resource, access.range).map_err(|d| {
                err(
                    CompileErrorKind::InvalidSubresourceRange,
                    vec![pass.id],
                    Some(access.resource),
                    d,
                    None,
                )
            })?;
            match (resource.kind, access.range) {
                (ResourceKind::Texture(_), DeclRange::Texture(_))
                | (ResourceKind::Buffer(_), DeclRange::Buffer(_)) => {}
                _ => {
                    return Err(err(
                        CompileErrorKind::StaleOrForeignVersion,
                        vec![pass.id],
                        Some(access.resource),
                        "resource handle kind mismatch",
                        None,
                    ));
                }
            }
            if let ResourceKind::Texture(desc) = resource.kind {
                let aspect = match access.range {
                    DeclRange::Texture(TextureRange::Whole) => TextureAspect::All,
                    DeclRange::Texture(TextureRange::Subresources { aspect, .. }) => aspect,
                    _ => unreachable!(),
                };
                match access.semantic {
                    AccessSemantic::ColorAttachment { .. }
                        if desc.format == TextureFormat::Depth32Float
                            || !matches!(aspect, TextureAspect::All | TextureAspect::Color) =>
                    {
                        return Err(err(
                            CompileErrorKind::InvalidSubresourceRange,
                            vec![pass.id],
                            Some(access.resource),
                            "color attachment requires a color aspect and format",
                            None,
                        ));
                    }
                    AccessSemantic::DepthStencilAttachment { depth, stencil }
                        if desc.format != TextureFormat::Depth32Float
                            || !depth
                            || stencil
                            || !matches!(aspect, TextureAspect::All | TextureAspect::Depth) =>
                    {
                        return Err(err(
                            CompileErrorKind::InvalidSubresourceRange,
                            vec![pass.id],
                            Some(access.resource),
                            "Depth32Float attachments require depth operations only",
                            None,
                        ));
                    }
                    _ => {}
                }
            }
        }
        for (index, left) in pass.accesses.iter().enumerate() {
            for right in pass.accesses.iter().skip(index + 1) {
                if left.resource == right.resource
                    && left.input_version == right.input_version
                    && (left.output_version.is_some() || right.output_version.is_some())
                    && ranges_overlap(resources[&left.resource], left.range, right.range)
                {
                    return Err(err(
                        CompileErrorKind::ConflictingAccess,
                        vec![pass.id],
                        Some(left.resource),
                        "overlapping same-pass accesses must use one read-write declaration",
                        None,
                    ));
                }
                if left.resource == right.resource
                    && left.output_version.is_none()
                    && right.output_version.is_none()
                    && ranges_overlap(resources[&left.resource], left.range, right.range)
                    && crate::plan::required_state(left) != crate::plan::required_state(right)
                {
                    return Err(err(
                        CompileErrorKind::ConflictingAccess,
                        vec![pass.id],
                        Some(left.resource),
                        "overlapping same-pass reads require incompatible resource states",
                        None,
                    ));
                }
            }
        }
        let mut attachment_indices = HashSet::new();
        let mut attachment_samples = None;
        for access in &pass.accesses {
            if matches!(
                access.semantic,
                AccessSemantic::ColorAttachment { .. }
                    | AccessSemantic::DepthStencilAttachment { .. }
            ) {
                let ResourceKind::Texture(desc) = resources[&access.resource].kind else {
                    unreachable!()
                };
                if attachment_samples
                    .replace(desc.sample_count)
                    .is_some_and(|samples| samples != desc.sample_count)
                {
                    return Err(err(
                        CompileErrorKind::ConflictingAccess,
                        vec![pass.id],
                        Some(access.resource),
                        "raster attachments must use the same sample count",
                        None,
                    ));
                }
                if let AccessSemantic::ColorAttachment { index } = access.semantic {
                    if !attachment_indices.insert(index) {
                        return Err(err(
                            CompileErrorKind::ConflictingAccess,
                            vec![pass.id],
                            Some(access.resource),
                            "duplicate color attachment index",
                            None,
                        ));
                    }
                }
            }
        }
    }
    for order in &graph.orders {
        if !pass_ids.contains(&order.before) || !pass_ids.contains(&order.after) {
            return Err(err(
                CompileErrorKind::StaleOrForeignVersion,
                vec![order.before, order.after],
                None,
                "explicit order references a foreign pass",
                None,
            ));
        }
    }
    for root in &graph.roots {
        let ok = match *root {
            RootDecl::Texture(_, r, _, _)
            | RootDecl::Buffer(_, r, _, _)
            | RootDecl::Present(_, r, _, _) => resources.contains_key(&r),
            RootDecl::SideEffect(p, _) => pass_ids.contains(&p),
        };
        if !ok {
            return Err(err(
                CompileErrorKind::InvalidExportOrPresent,
                Vec::new(),
                None,
                "root references another graph",
                None,
            ));
        }
    }
    Ok(())
}
fn validate_range(resource: &ResourceDecl, range: DeclRange) -> Result<(), &'static str> {
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
fn valid_tex_desc(d: crate::rhi::TextureDesc) -> Result<(), &'static str> {
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
