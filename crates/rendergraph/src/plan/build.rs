//! Lowers retained declarations into the backend-neutral immutable execution plan.
//!
//! Queue selection uses the same capability predicate already validated by compilation,
//! while resource requirements include both pass accesses and import/export boundary
//! states so physical objects authorize the entire frame contract.

use std::collections::HashMap;

use crate::{
    handles::PassId,
    internal::{
        AccessDecl, AccessDetails, AccessSemantic, DeclRange, PassDecl, ResourceDecl, ResourceKind,
        ResourceOrigin, RootDecl,
    },
    pass::PassKind,
    rhi::{DeviceCapabilities, ResourceAccessState},
};

use super::{
    BufferUsage, BufferUsageKind, PlannedColorAttachment, PlannedDepthStencilAttachment,
    PlannedPass, RasterPassPlan, ResourceRequirement, ResourceUsageSummary, TextureUsage,
    TextureUsageKind, execution_plan, state::ResourceStates,
};

pub(crate) fn build_execution_plan<F>(
    passes: &[PassDecl<F>],
    execution_order: &[PassId],
    resources: &[ResourceDecl],
    roots: &[RootDecl],
    capabilities: &DeviceCapabilities,
) -> super::ExecutionPlan {
    let pass_by_id: HashMap<_, _> = passes.iter().map(|pass| (pass.id, pass)).collect();
    let resource_by_id: HashMap<_, _> = resources
        .iter()
        .map(|resource| (resource.id, resource))
        .collect();
    let needs_present = roots
        .iter()
        .any(|root| matches!(root, RootDecl::Present(_, _, _, _)));
    let queue = capabilities
        .queues
        .iter()
        .find(|queue| {
            (!needs_present || queue.capabilities.present)
                && execution_order.iter().all(|id| {
                    let pass = pass_by_id[id];
                    match pass.kind {
                        PassKind::Raster => queue.capabilities.raster,
                        PassKind::Compute => queue.capabilities.compute,
                        PassKind::Copy => queue.capabilities.copy,
                    }
                })
        })
        .map(|queue| queue.id);

    let mut states = ResourceStates::new(resources, passes, roots);
    let mut planned_passes = Vec::with_capacity(execution_order.len());
    for pass_id in execution_order {
        let pass = pass_by_id[pass_id];
        let mut transitions = Vec::new();
        for access in &pass.accesses {
            transitions.extend(states.transition(
                resource_by_id[&access.resource],
                access.range,
                required_state(access),
                access.mode,
            ));
        }
        planned_passes.push(PlannedPass {
            pass: pass.id,
            kind: pass.kind,
            transitions,
            raster: raster_plan(pass),
        });
    }

    let mut final_transitions = Vec::new();
    for root in roots {
        let (resource, state) = match *root {
            RootDecl::Texture(_, resource, _, contract) => (resource, contract.final_state),
            RootDecl::Buffer(_, resource, _, contract) => (resource, contract.final_state),
            RootDecl::Present(_, resource, _, _) => (resource, ResourceAccessState::Present),
            RootDecl::SideEffect(..) => continue,
        };
        let declaration = resource_by_id[&resource];
        let range = match declaration.kind {
            ResourceKind::Texture(_) => DeclRange::Texture(crate::access::TextureRange::Whole),
            ResourceKind::Buffer(_) => DeclRange::Buffer(crate::access::BufferRange::Whole),
        };
        // Exports are state requirements, not a graph access, so they do not
        // introduce a new write hazard of their own.
        final_transitions.extend(states.transition(
            declaration,
            range,
            state,
            crate::access::AccessMode::Read,
        ));
    }

    let resource_requirements = resource_requirements(resources, passes, roots);
    execution_plan(
        queue,
        planned_passes,
        final_transitions,
        resource_requirements,
    )
}

fn resource_requirements<F>(
    resources: &[ResourceDecl],
    passes: &[PassDecl<F>],
    roots: &[RootDecl],
) -> Vec<ResourceRequirement> {
    let mut usages: HashMap<_, _> = resources
        .iter()
        .map(|resource| {
            let usage = match resource.kind {
                ResourceKind::Texture(_) => ResourceUsageSummary::Texture(TextureUsage::default()),
                ResourceKind::Buffer(_) => ResourceUsageSummary::Buffer(BufferUsage::default()),
            };
            (resource.id, usage)
        })
        .collect();
    let mut live = HashMap::new();

    for pass in passes {
        for access in &pass.accesses {
            add_usage(
                usages
                    .get_mut(&access.resource)
                    .expect("resource usage exists"),
                required_state(access),
            );
            live.insert(access.resource, ());
        }
    }
    for root in roots {
        let (resource, state) = match *root {
            RootDecl::Texture(_, resource, _, contract) => (resource, contract.final_state),
            RootDecl::Buffer(_, resource, _, contract) => (resource, contract.final_state),
            RootDecl::Present(_, resource, _, _) => (resource, ResourceAccessState::Present),
            RootDecl::SideEffect(..) => continue,
        };
        add_usage(
            usages.get_mut(&resource).expect("resource usage exists"),
            state,
        );
        live.insert(resource, ());
    }

    for resource in resources
        .iter()
        .filter(|resource| live.contains_key(&resource.id))
    {
        let initial_state = match resource.origin {
            ResourceOrigin::TextureImport(_, contract) => Some(contract.initial_state),
            ResourceOrigin::BufferImport(_, contract) => Some(contract.initial_state),
            ResourceOrigin::Transient | ResourceOrigin::Surface(_, _) => None,
        };
        // The imported incoming state is an actual operation requirement, not
        // metadata: the bound physical object must have been created for it.
        if let Some(state) = initial_state {
            add_usage(
                usages.get_mut(&resource.id).expect("resource usage exists"),
                state,
            );
        }
    }

    resources
        .iter()
        .filter(|resource| live.contains_key(&resource.id))
        .map(|resource| ResourceRequirement {
            resource: resource.id,
            usage: usages.remove(&resource.id).expect("resource usage exists"),
        })
        .collect()
}

fn add_usage(summary: &mut ResourceUsageSummary, state: ResourceAccessState) {
    match summary {
        ResourceUsageSummary::Texture(usage) => match state {
            ResourceAccessState::Undefined => {}
            ResourceAccessState::ColorAttachmentRead
            | ResourceAccessState::ColorAttachmentWrite
            | ResourceAccessState::ColorAttachmentReadWrite => {
                usage.insert(TextureUsageKind::ColorAttachment)
            }
            ResourceAccessState::DepthStencilRead
            | ResourceAccessState::DepthStencilWrite
            | ResourceAccessState::DepthStencilReadWrite => {
                usage.insert(TextureUsageKind::DepthStencilAttachment)
            }
            ResourceAccessState::ShaderSampledRead => usage.insert(TextureUsageKind::Sampled),
            ResourceAccessState::ShaderStorageRead => usage.insert(TextureUsageKind::StorageRead),
            ResourceAccessState::ShaderStorageWrite => usage.insert(TextureUsageKind::StorageWrite),
            ResourceAccessState::ShaderStorageReadWrite => {
                usage.insert(TextureUsageKind::StorageRead);
                usage.insert(TextureUsageKind::StorageWrite);
            }
            ResourceAccessState::CopySource => usage.insert(TextureUsageKind::CopySource),
            ResourceAccessState::CopyDestination => usage.insert(TextureUsageKind::CopyDestination),
            ResourceAccessState::Present => usage.insert(TextureUsageKind::Present),
            ResourceAccessState::UniformRead
            | ResourceAccessState::VertexRead
            | ResourceAccessState::IndexRead
            | ResourceAccessState::IndirectRead => {}
        },
        ResourceUsageSummary::Buffer(usage) => match state {
            ResourceAccessState::Undefined => {}
            ResourceAccessState::ShaderStorageRead => usage.insert(BufferUsageKind::StorageRead),
            ResourceAccessState::ShaderStorageWrite => usage.insert(BufferUsageKind::StorageWrite),
            ResourceAccessState::ShaderStorageReadWrite => {
                usage.insert(BufferUsageKind::StorageRead);
                usage.insert(BufferUsageKind::StorageWrite);
            }
            ResourceAccessState::UniformRead => usage.insert(BufferUsageKind::Uniform),
            ResourceAccessState::VertexRead => usage.insert(BufferUsageKind::Vertex),
            ResourceAccessState::IndexRead => usage.insert(BufferUsageKind::Index),
            ResourceAccessState::IndirectRead => usage.insert(BufferUsageKind::Indirect),
            ResourceAccessState::CopySource => usage.insert(BufferUsageKind::CopySource),
            ResourceAccessState::CopyDestination => usage.insert(BufferUsageKind::CopyDestination),
            ResourceAccessState::ColorAttachmentRead
            | ResourceAccessState::ColorAttachmentWrite
            | ResourceAccessState::ColorAttachmentReadWrite
            | ResourceAccessState::DepthStencilRead
            | ResourceAccessState::DepthStencilWrite
            | ResourceAccessState::DepthStencilReadWrite
            | ResourceAccessState::ShaderSampledRead
            | ResourceAccessState::Present => {}
        },
    }
}

fn raster_plan<F>(pass: &PassDecl<F>) -> Option<RasterPassPlan> {
    if pass.kind != PassKind::Raster {
        return None;
    }
    let mut plan = RasterPassPlan::default();
    for access in &pass.accesses {
        match access.details {
            AccessDetails::None => {}
            AccessDetails::ColorAttachment(descriptor) => {
                plan.colors.push(PlannedColorAttachment {
                    resource: access.resource,
                    descriptor,
                });
            }
            AccessDetails::DepthStencilAttachment(descriptor) => {
                plan.depth_stencil = Some(PlannedDepthStencilAttachment {
                    resource: access.resource,
                    descriptor,
                });
            }
        }
    }
    plan.colors
        .sort_by_key(|attachment| attachment.descriptor.index);
    Some(plan)
}

pub(crate) fn required_state(access: &AccessDecl) -> ResourceAccessState {
    use crate::access::{
        BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureReadUse, TextureReadWriteUse,
        TextureWriteUse,
    };
    match access.semantic {
        AccessSemantic::TextureRead(TextureReadUse::Sampled) => {
            ResourceAccessState::ShaderSampledRead
        }
        AccessSemantic::TextureRead(TextureReadUse::Storage) => {
            ResourceAccessState::ShaderStorageRead
        }
        AccessSemantic::TextureRead(TextureReadUse::CopySource) => ResourceAccessState::CopySource,
        AccessSemantic::TextureWrite(TextureWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageWrite
        }
        AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
            ResourceAccessState::CopyDestination
        }
        AccessSemantic::TextureReadWrite(TextureReadWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageReadWrite
        }
        AccessSemantic::BufferRead(BufferReadUse::Uniform) => ResourceAccessState::UniformRead,
        AccessSemantic::BufferRead(BufferReadUse::Storage) => {
            ResourceAccessState::ShaderStorageRead
        }
        AccessSemantic::BufferRead(BufferReadUse::Vertex) => ResourceAccessState::VertexRead,
        AccessSemantic::BufferRead(BufferReadUse::Index) => ResourceAccessState::IndexRead,
        AccessSemantic::BufferRead(BufferReadUse::Indirect) => ResourceAccessState::IndirectRead,
        AccessSemantic::BufferRead(BufferReadUse::CopySource) => ResourceAccessState::CopySource,
        AccessSemantic::BufferWrite(BufferWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageWrite
        }
        AccessSemantic::BufferWrite(BufferWriteUse::CopyDestination) => {
            ResourceAccessState::CopyDestination
        }
        AccessSemantic::BufferReadWrite(BufferReadWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageReadWrite
        }
        AccessSemantic::ColorAttachment { .. } if access.read_required => {
            ResourceAccessState::ColorAttachmentReadWrite
        }
        AccessSemantic::ColorAttachment { .. } => ResourceAccessState::ColorAttachmentWrite,
        AccessSemantic::DepthStencilAttachment { .. } if access.read_required => {
            ResourceAccessState::DepthStencilReadWrite
        }
        AccessSemantic::DepthStencilAttachment { .. } => ResourceAccessState::DepthStencilWrite,
    }
}
