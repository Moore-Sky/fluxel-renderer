//! Derives range-partitioned resource state and visibility transitions.
//!
//! Buffer partitions are fixed from every declared access and exported boundary before
//! lowering starts, so later transitions cannot skip a range boundary. Equal native
//! states still require a transition when either access writes: that transition carries
//! memory ordering/visibility semantics rather than a state-layout change.

use std::collections::{BTreeSet, HashMap};

use crate::{
    access::{AccessMode, BufferRange, TextureAspect, TextureRange},
    handles::ResourceId,
    internal::{DeclRange, PassDecl, ResourceDecl, ResourceKind, ResourceOrigin, RootDecl},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc, TextureFormat},
};

use super::{PlannedResourceRange, PlannedTransition};

enum ResourceState {
    Texture(TextureState),
    Buffer(BufferState),
}

pub(super) struct ResourceStates {
    states: HashMap<ResourceId, ResourceState>,
}

impl ResourceStates {
    pub(super) fn new<F>(
        resources: &[ResourceDecl],
        passes: &[PassDecl<F>],
        roots: &[RootDecl],
    ) -> Self {
        let mut states = HashMap::new();
        for resource in resources {
            let initial = match resource.origin {
                ResourceOrigin::TextureImport(_, contract) => contract.initial_state,
                ResourceOrigin::BufferImport(_, contract) => contract.initial_state,
                ResourceOrigin::Transient | ResourceOrigin::Surface(_, _) => {
                    ResourceAccessState::Undefined
                }
            };
            let state = match resource.kind {
                ResourceKind::Texture(descriptor) => {
                    ResourceState::Texture(TextureState::new(descriptor, initial))
                }
                ResourceKind::Buffer(descriptor) => {
                    let mut boundaries = BTreeSet::from([0, descriptor.size]);
                    for access in passes
                        .iter()
                        .flat_map(|pass| pass.accesses.iter())
                        .filter(|access| access.resource == resource.id)
                    {
                        if let DeclRange::Buffer(range) = access.range {
                            let (start, end) = buffer_bounds(descriptor, range);
                            boundaries.insert(start);
                            boundaries.insert(end);
                        }
                    }
                    if roots.iter().any(|root| match root {
                        RootDecl::Buffer(_, id, _, _) => *id == resource.id,
                        _ => false,
                    }) {
                        boundaries.insert(0);
                        boundaries.insert(descriptor.size);
                    }
                    ResourceState::Buffer(BufferState::new(boundaries, initial))
                }
            };
            states.insert(resource.id, state);
        }
        Self { states }
    }

    pub(super) fn transition(
        &mut self,
        resource: &ResourceDecl,
        range: DeclRange,
        after: ResourceAccessState,
        mode: AccessMode,
    ) -> Vec<PlannedTransition> {
        match (&mut self.states.get_mut(&resource.id).unwrap(), range) {
            (ResourceState::Texture(state), DeclRange::Texture(range)) => {
                state.transition(resource.id, range, after, mode)
            }
            (ResourceState::Buffer(state), DeclRange::Buffer(range)) => {
                state.transition(resource.id, range, after, mode)
            }
            _ => unreachable!("validated ranges match resource kinds"),
        }
    }
}

struct TextureState {
    descriptor: TextureDesc,
    states: HashMap<(u32, u32, u8), TextureSubresourceState>,
}

#[derive(Clone, Copy)]
struct TextureSubresourceState {
    state: ResourceAccessState,
    last_access_wrote: bool,
}

impl TextureState {
    fn new(descriptor: TextureDesc, initial: ResourceAccessState) -> Self {
        let states = texture_keys(descriptor, TextureRange::Whole)
            .into_iter()
            .map(|key| {
                (
                    key,
                    TextureSubresourceState {
                        state: initial,
                        // Import contracts describe a state, not a preceding graph access.
                        last_access_wrote: false,
                    },
                )
            })
            .collect();
        Self { descriptor, states }
    }

    fn transition(
        &mut self,
        resource: ResourceId,
        range: TextureRange,
        after: ResourceAccessState,
        mode: AccessMode,
    ) -> Vec<PlannedTransition> {
        let mut transitions = Vec::new();
        for (mip, layer, aspect) in texture_keys(self.descriptor, range) {
            let previous = self.states[&(mip, layer, aspect)];
            let before = previous.state;
            // Same-state write hazards still need a memory dependency; a state
            // equality check alone is not a sufficient ordering proof.
            if before != after || previous.last_access_wrote || mode != AccessMode::Read {
                transitions.push(PlannedTransition {
                    resource,
                    range: PlannedResourceRange::Texture(TextureRange::Subresources {
                        base_mip_level: mip,
                        mip_level_count: 1,
                        base_array_layer: layer,
                        array_layer_count: 1,
                        aspect: aspect_from_key(aspect),
                    }),
                    before,
                    after,
                });
            }
            self.states.insert(
                (mip, layer, aspect),
                TextureSubresourceState {
                    state: after,
                    last_access_wrote: mode != AccessMode::Read,
                },
            );
        }
        transitions
    }
}

fn texture_keys(descriptor: TextureDesc, range: TextureRange) -> Vec<(u32, u32, u8)> {
    let (base_mip, mip_count, base_layer, layer_count, aspect) = match range {
        TextureRange::Whole => (
            0,
            descriptor.mip_levels,
            0,
            descriptor.array_layers,
            TextureAspect::All,
        ),
        TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        } => (
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        ),
    };
    let aspects: &[u8] = match aspect {
        TextureAspect::All if descriptor.format == TextureFormat::Depth32Float => &[1],
        TextureAspect::All | TextureAspect::Color => &[0],
        TextureAspect::Depth => &[1],
        TextureAspect::Stencil => &[2],
    };
    let mut keys = Vec::new();
    for mip in base_mip..base_mip + mip_count {
        for layer in base_layer..base_layer + layer_count {
            for aspect in aspects {
                keys.push((mip, layer, *aspect));
            }
        }
    }
    keys
}

fn aspect_from_key(aspect: u8) -> TextureAspect {
    match aspect {
        0 => TextureAspect::Color,
        1 => TextureAspect::Depth,
        2 => TextureAspect::Stencil,
        _ => unreachable!(),
    }
}

struct BufferSegment {
    start: u64,
    end: u64,
    state: ResourceAccessState,
    last_access_wrote: bool,
}

struct BufferState {
    size: u64,
    segments: Vec<BufferSegment>,
}

impl BufferState {
    fn new(boundaries: BTreeSet<u64>, initial: ResourceAccessState) -> Self {
        let points: Vec<_> = boundaries.into_iter().collect();
        let segments = points
            .windows(2)
            .map(|window| BufferSegment {
                start: window[0],
                end: window[1],
                state: initial,
                // Import contracts describe a state, not a preceding graph access.
                last_access_wrote: false,
            })
            .collect();
        Self {
            size: *points.last().unwrap(),
            segments,
        }
    }

    fn transition(
        &mut self,
        resource: ResourceId,
        range: BufferRange,
        after: ResourceAccessState,
        mode: AccessMode,
    ) -> Vec<PlannedTransition> {
        let (start, end) = match range {
            BufferRange::Whole => (0, self.size),
            BufferRange::Bytes { offset, size } => (offset, offset + size),
        };
        let mut transitions: Vec<PlannedTransition> = Vec::new();
        for segment in self
            .segments
            .iter_mut()
            .filter(|segment| segment.start >= start && segment.end <= end)
        {
            let needs_transition =
                segment.state != after || segment.last_access_wrote || mode != AccessMode::Read;
            if !needs_transition {
                segment.last_access_wrote = false;
                continue;
            }
            if let Some(previous) = transitions.last_mut() {
                if previous.resource == resource
                    && previous.before == segment.state
                    && previous.after == after
                {
                    if let PlannedResourceRange::Buffer(BufferRange::Bytes { offset, size }) =
                        &mut previous.range
                    {
                        if *offset + *size == segment.start {
                            *size += segment.end - segment.start;
                            segment.state = after;
                            segment.last_access_wrote = mode != AccessMode::Read;
                            continue;
                        }
                    }
                }
            }
            transitions.push(PlannedTransition {
                resource,
                range: PlannedResourceRange::Buffer(BufferRange::Bytes {
                    offset: segment.start,
                    size: segment.end - segment.start,
                }),
                before: segment.state,
                after,
            });
            segment.state = after;
            segment.last_access_wrote = mode != AccessMode::Read;
        }
        transitions
    }
}

fn buffer_bounds(descriptor: BufferDesc, range: BufferRange) -> (u64, u64) {
    match range {
        BufferRange::Whole => (0, descriptor.size),
        BufferRange::Bytes { offset, size } => (offset, offset + size),
    }
}
