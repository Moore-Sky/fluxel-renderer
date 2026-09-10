//! Resolves renderer-owned binding recipes against current-pass declarations.

use std::marker::PhantomData;

use crate::{RecordResult, handles::BindingSetId};

use super::{BindingResource, ResolvedBindings};

pub(crate) trait BindingResolverSink {
    fn resolve_bindings(
        &mut self,
        recipe: BindingSetId,
        resources: &[BindingResource<'_>],
        dynamic_offsets: &[u32],
    ) -> RecordResult<(u64, u64)>;
}

/// Resolver that restricts physical resource resolution to the current pass.
///
/// Binding recipes remain renderer/RHI owned. The resolver validates that every
/// listed graph resource was declared by this pass with a compatible use.
pub struct PassResourceResolver<'a> {
    sink: &'a mut dyn BindingResolverSink,
}

impl<'a> PassResourceResolver<'a> {
    /// Resolves one opaque binding recipe against declared pass resources.
    pub fn resolve_bindings<'b>(
        &'b mut self,
        recipe: BindingSetId,
        resources: &[BindingResource<'_>],
        dynamic_offsets: &[u32],
    ) -> RecordResult<ResolvedBindings<'b>> {
        let (ticket, session) = self
            .sink
            .resolve_bindings(recipe, resources, dynamic_offsets)?;
        Ok(ResolvedBindings {
            ticket,
            session,
            _marker: PhantomData,
        })
    }

    pub(crate) fn new(sink: &'a mut dyn BindingResolverSink) -> Self {
        Self { sink }
    }
}
