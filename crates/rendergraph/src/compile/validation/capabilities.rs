//! Rejects retained graph semantics unsupported by the selected device.

use super::super::*;

pub(in crate::compile) fn validate_capabilities<F>(
    graph: &RenderGraph<F>,
    resources: &HashMap<ResourceId, &ResourceDecl>,
    retained: &HashSet<PassId>,
    caps: &DeviceCapabilities,
) -> Result<(), CompileError> {
    for resource in &graph.resources {
        match resource.origin {
            ResourceOrigin::TextureImport(_, contract)
                if contract.initial_state != ResourceAccessState::Undefined
                    && !texture_state_supported(
                        caps,
                        contract.descriptor.format,
                        contract.initial_state,
                    ) =>
            {
                return Err(unsupported_root(
                    resource.id,
                    "texture import state is unsupported for its format",
                    caps,
                ));
            }
            ResourceOrigin::BufferImport(_, contract)
                if contract.initial_state != ResourceAccessState::Undefined
                    && !buffer_state_supported(caps, contract.initial_state) =>
            {
                return Err(unsupported_root(
                    resource.id,
                    "buffer import state is unsupported",
                    caps,
                ));
            }
            _ => {}
        }
    }
    let needs_present = graph
        .roots
        .iter()
        .any(|root| matches!(root, RootDecl::Present(_, _, _, _)));
    let needs_queue = !retained.is_empty()
        || graph.roots.iter().any(|root| {
            matches!(
                root,
                RootDecl::Texture(_, _, _, _)
                    | RootDecl::Buffer(_, _, _, _)
                    | RootDecl::Present(_, _, _, _)
            )
        });
    let single_queue = caps.queues.iter().any(|queue| {
        (!needs_present || queue.capabilities.present)
            && graph
                .passes
                .iter()
                .filter(|pass| retained.contains(&pass.id))
                .all(|pass| match pass.kind {
                    PassKind::Raster => queue.capabilities.raster,
                    PassKind::Compute => queue.capabilities.compute,
                    PassKind::Copy => queue.capabilities.copy,
                })
    });
    if needs_queue && !single_queue {
        let passes = graph
            .passes
            .iter()
            .filter(|pass| retained.contains(&pass.id))
            .map(|pass| pass.id)
            .collect();
        return Err(err(
            CompileErrorKind::UnsupportedSemanticRequirement,
            passes,
            None,
            "the current single-queue compiler requires one logical queue that supports every retained pass kind and presentation requirement",
            Some(caps.clone()),
        ));
    }
    for p in graph.passes.iter().filter(|p| retained.contains(&p.id)) {
        let queue = caps.queues.iter().any(|q| match p.kind {
            PassKind::Raster => q.capabilities.raster,
            PassKind::Compute => q.capabilities.compute,
            PassKind::Copy => q.capabilities.copy,
        });
        if !queue {
            return Err(unsupported(
                p.id,
                None,
                "no queue supports this pass kind",
                caps,
            ));
        }
        let color_indices = p
            .accesses
            .iter()
            .filter_map(|a| {
                if let AccessSemantic::ColorAttachment { index } = a.semantic {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
        let required_color_slots = color_indices.iter().max().map_or(0, |index| index + 1);
        if required_color_slots > caps.limits.max_color_attachments {
            return Err(unsupported(
                p.id,
                None,
                "color attachment count exceeds limit",
                caps,
            ));
        }
        for a in &p.accesses {
            let r = resources[&a.resource];
            match (r.kind, a.semantic) {
                (ResourceKind::Buffer(_), AccessSemantic::BufferRead(BufferReadUse::Storage))
                    if !caps.buffers.storage_read =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer reads unsupported",
                        caps,
                    ));
                }
                (ResourceKind::Buffer(_), AccessSemantic::BufferRead(BufferReadUse::Indirect))
                    if !caps.buffers.indirect_read =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "indirect buffer reads unsupported",
                        caps,
                    ));
                }
                (ResourceKind::Buffer(_), AccessSemantic::BufferWrite(BufferWriteUse::Storage))
                    if !caps.buffers.storage_write =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer writes unsupported",
                        caps,
                    ));
                }
                (
                    ResourceKind::Buffer(_),
                    AccessSemantic::BufferReadWrite(BufferReadWriteUse::Storage),
                ) if !caps.buffers.storage_read || !caps.buffers.storage_write => {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer read-write unsupported",
                        caps,
                    ));
                }
                (ResourceKind::Texture(d), semantic) => {
                    let Some(f) = caps.texture_formats.iter().find(|f| f.format == d.format) else {
                        return Err(unsupported(
                            p.id,
                            Some(r.id),
                            "missing texture format capabilities",
                            caps,
                        ));
                    };
                    let ok = match semantic {
                        AccessSemantic::TextureRead(TextureReadUse::Sampled) => f.sampled,
                        AccessSemantic::TextureRead(TextureReadUse::Storage) => f.storage_read,
                        AccessSemantic::TextureRead(TextureReadUse::CopySource) => f.copy_source,
                        AccessSemantic::TextureWrite(TextureWriteUse::Storage) => f.storage_write,
                        AccessSemantic::TextureReadWrite(TextureReadWriteUse::Storage) => {
                            f.storage_read && f.storage_write
                        }
                        AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
                            f.copy_destination
                        }
                        AccessSemantic::ColorAttachment { .. } => {
                            f.color_attachment
                                && f.attachment_sample_counts.contains(&d.sample_count)
                        }
                        AccessSemantic::DepthStencilAttachment { .. } => {
                            f.depth_stencil_attachment
                                && f.attachment_sample_counts.contains(&d.sample_count)
                        }
                        _ => true,
                    };
                    if !ok {
                        return Err(unsupported(
                            p.id,
                            Some(r.id),
                            "texture semantic unsupported for format",
                            caps,
                        ));
                    }
                    if matches!(r.origin, ResourceOrigin::Surface(_, _)) {
                        let Some(s) = &caps.surface else {
                            return Err(unsupported(
                                p.id,
                                Some(r.id),
                                "surface capabilities unavailable",
                                caps,
                            ));
                        };
                        let ok = match semantic {
                            AccessSemantic::ColorAttachment { .. } => s.color_attachment,
                            AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
                                s.copy_destination
                            }
                            _ => false,
                        };
                        if !ok {
                            return Err(unsupported(
                                p.id,
                                Some(r.id),
                                "surface destination semantic unsupported",
                                caps,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
