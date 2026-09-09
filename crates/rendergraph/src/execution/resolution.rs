//! Resolves and validates graph resources for one frame.

use std::collections::{HashMap, HashSet};

use crate::{
    CompiledGraph,
    backend::{
        BoundBuffer, BoundTexture, ExecutionBackend, ExecutionError, FrameBindingError,
        FrameBindingErrorKind, FrameResourceProvider,
    },
    handles::ResourceId,
    internal::{ResourceKind, ResourceOrigin, RootDecl},
    plan::{BufferUsage, ResourceUsageSummary, TextureUsage},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc},
};

use super::super::{
    FrameExecution,
    recording::{PhysicalResource, PhysicalResources},
};

pub(super) fn live_resources<F>(graph: &CompiledGraph<F>) -> HashSet<ResourceId> {
    let mut live = HashSet::new();
    for pass in &graph.passes {
        live.extend(pass.accesses.iter().map(|access| access.resource));
    }
    for root in &graph.roots {
        match root {
            RootDecl::Texture(_, resource, _, _)
            | RootDecl::Buffer(_, resource, _, _)
            | RootDecl::Present(_, resource, _, _) => {
                live.insert(*resource);
            }
            RootDecl::SideEffect(..) => {}
        }
    }
    live
}

pub(super) struct ResolvedFrameResources<B: ExecutionBackend> {
    pub(super) physical: PhysicalResources<B>,
    pub(super) resource_leases: HashMap<ResourceId, B::Lease>,
    pub(super) retained: Vec<B::Lease>,
}

pub(super) fn resolve_resources<B, R, F, M>(
    graph: &CompiledGraph<F>,
    execution: &FrameExecution<F, M>,
    provider: &R,
    backend: &mut B,
    live: &HashSet<ResourceId>,
) -> Result<ResolvedFrameResources<B>, ExecutionError<B::Error>>
where
    B: ExecutionBackend,
    R: FrameResourceProvider<B>,
{
    let texture_bindings: HashMap<_, _> = execution.inputs.textures.iter().copied().collect();
    let buffer_bindings: HashMap<_, _> = execution.inputs.buffers.iter().copied().collect();
    let surface_slots: HashSet<_> = execution
        .inputs
        .surfaces
        .iter()
        .map(|(slot, _)| *slot)
        .collect();
    let mut physical = HashMap::new();
    let mut resource_leases = HashMap::new();
    let mut retained = Vec::new();
    // Texture and buffer identities live in independent backend namespaces.
    let mut texture_identities = HashMap::new();
    let mut buffer_identities = HashMap::new();
    for resource in graph
        .resources
        .iter()
        .filter(|resource| live.contains(&resource.id))
    {
        match (resource.kind, resource.origin) {
            (ResourceKind::Texture(descriptor), ResourceOrigin::Transient) => {
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Texture(usage)) => usage,
                    _ => unreachable!("live transient texture has a texture usage requirement"),
                };
                let bound = backend
                    .create_transient_texture(descriptor, usage)
                    .map_err(ExecutionError::Backend)?;
                validate_bound_texture(
                    backend,
                    resource.id,
                    descriptor,
                    ResourceAccessState::Undefined,
                    usage,
                    None,
                    &bound,
                )?;
                reject_alias(&mut texture_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Texture {
                        physical: bound.physical,
                        descriptor,
                    },
                );
            }
            (ResourceKind::Buffer(descriptor), ResourceOrigin::Transient) => {
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Buffer(usage)) => usage,
                    _ => unreachable!("live transient buffer has a buffer usage requirement"),
                };
                let bound = backend
                    .create_transient_buffer(descriptor, usage)
                    .map_err(ExecutionError::Backend)?;
                validate_bound_buffer(
                    backend,
                    resource.id,
                    descriptor,
                    ResourceAccessState::Undefined,
                    usage,
                    None,
                    &bound,
                )?;
                reject_alias(&mut buffer_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Buffer {
                        physical: bound.physical,
                        descriptor,
                    },
                );
            }
            (ResourceKind::Texture(descriptor), ResourceOrigin::TextureImport(slot, contract)) => {
                if surface_slots.contains(&slot) {
                    return Err(frame_error(
                        FrameBindingErrorKind::ConflictingSlotBinding,
                        Some(slot),
                        None,
                        "ordinary texture import slot also has a surface binding",
                    ));
                }
                let id = texture_bindings.get(&slot).copied().ok_or_else(|| {
                    frame_error(
                        FrameBindingErrorKind::MissingTexture,
                        Some(slot),
                        None,
                        "required texture import was not bound",
                    )
                })?;
                let bound = provider.texture(id).map_err(ExecutionError::FrameBinding)?;
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Texture(usage)) => usage,
                    _ => unreachable!("live imported texture has a texture usage requirement"),
                };
                validate_bound_texture(
                    backend,
                    resource.id,
                    descriptor,
                    contract.initial_state,
                    usage,
                    Some(slot),
                    &bound,
                )?;
                reject_alias(&mut texture_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Texture {
                        physical: bound.physical,
                        descriptor,
                    },
                );
            }
            (ResourceKind::Buffer(descriptor), ResourceOrigin::BufferImport(slot, contract)) => {
                let id = buffer_bindings.get(&slot).copied().ok_or_else(|| {
                    frame_error(
                        FrameBindingErrorKind::MissingBuffer,
                        None,
                        Some(slot),
                        "required buffer import was not bound",
                    )
                })?;
                let bound = provider.buffer(id).map_err(ExecutionError::FrameBinding)?;
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Buffer(usage)) => usage,
                    _ => unreachable!("live imported buffer has a buffer usage requirement"),
                };
                validate_bound_buffer(
                    backend,
                    resource.id,
                    descriptor,
                    contract.initial_state,
                    usage,
                    Some(slot),
                    &bound,
                )?;
                reject_alias(&mut buffer_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Buffer {
                        physical: bound.physical,
                        descriptor,
                    },
                );
            }
            (_, ResourceOrigin::Surface(_, _)) => unreachable!("surface plans are rejected first"),
            _ => unreachable!("validated resource kind and origin match"),
        }
    }
    Ok(ResolvedFrameResources {
        physical,
        resource_leases,
        retained,
    })
}

fn reject_alias<E>(
    identities: &mut HashMap<crate::PhysicalResourceIdentity, ResourceId>,
    physical: crate::PhysicalResourceIdentity,
    logical: ResourceId,
) -> Result<(), ExecutionError<E>> {
    if let Some(previous) = identities.insert(physical, logical) {
        if previous != logical {
            return Err(frame_error(
                FrameBindingErrorKind::AliasedPhysicalResource,
                None,
                None,
                "distinct logical resources resolved to one physical generation",
            ));
        }
    }
    Ok(())
}

fn validate_bound_texture<B: ExecutionBackend>(
    backend: &B,
    resource: ResourceId,
    descriptor: TextureDesc,
    state: ResourceAccessState,
    required_usage: TextureUsage,
    texture_slot: Option<crate::ImportTextureSlot>,
    bound: &BoundTexture<B::Texture, B::Lease>,
) -> Result<(), ExecutionError<B::Error>> {
    if bound.device != backend.device_identity() {
        return Err(frame_error(
            FrameBindingErrorKind::DeviceMismatch,
            texture_slot,
            None,
            "texture belongs to another backend device",
        ));
    }
    if bound.descriptor != descriptor {
        return Err(frame_error(
            FrameBindingErrorKind::DescriptorMismatch,
            texture_slot,
            None,
            "texture descriptor differs from the compiled import contract",
        ));
    }
    if bound.initial_state != state {
        return Err(frame_error(
            FrameBindingErrorKind::InitialStateMismatch,
            texture_slot,
            None,
            "texture incoming state differs from the compiled import contract",
        ));
    }
    if !bound.usage.contains_all(required_usage) {
        return Err(ExecutionError::FrameBinding(FrameBindingError {
            kind: FrameBindingErrorKind::UsageMismatch,
            texture_slot,
            buffer_slot: None,
            resource: Some(resource),
            surface_binding: None,
            detail: format!(
                "texture allowed operations do not cover the compiled requirement: required {required_usage:?}, actual {:?}",
                bound.usage
            ),
        }));
    }
    Ok(())
}

fn validate_bound_buffer<B: ExecutionBackend>(
    backend: &B,
    resource: ResourceId,
    descriptor: BufferDesc,
    state: ResourceAccessState,
    required_usage: BufferUsage,
    buffer_slot: Option<crate::ImportBufferSlot>,
    bound: &BoundBuffer<B::Buffer, B::Lease>,
) -> Result<(), ExecutionError<B::Error>> {
    if bound.device != backend.device_identity() {
        return Err(frame_error(
            FrameBindingErrorKind::DeviceMismatch,
            None,
            buffer_slot,
            "buffer belongs to another backend device",
        ));
    }
    if bound.descriptor != descriptor {
        return Err(frame_error(
            FrameBindingErrorKind::DescriptorMismatch,
            None,
            buffer_slot,
            "buffer descriptor differs from the compiled import contract",
        ));
    }
    if bound.initial_state != state {
        return Err(frame_error(
            FrameBindingErrorKind::InitialStateMismatch,
            None,
            buffer_slot,
            "buffer incoming state differs from the compiled import contract",
        ));
    }
    if !bound.usage.contains_all(required_usage) {
        return Err(ExecutionError::FrameBinding(FrameBindingError {
            kind: FrameBindingErrorKind::UsageMismatch,
            texture_slot: None,
            buffer_slot,
            resource: Some(resource),
            surface_binding: None,
            detail: format!(
                "buffer allowed operations do not cover the compiled requirement: required {required_usage:?}, actual {:?}",
                bound.usage
            ),
        }));
    }
    Ok(())
}

fn frame_error<E>(
    kind: FrameBindingErrorKind,
    texture_slot: Option<crate::ImportTextureSlot>,
    buffer_slot: Option<crate::ImportBufferSlot>,
    detail: impl Into<String>,
) -> ExecutionError<E> {
    ExecutionError::FrameBinding(FrameBindingError {
        kind,
        texture_slot,
        buffer_slot,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    })
}
