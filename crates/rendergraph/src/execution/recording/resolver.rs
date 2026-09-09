//! Binding-resource declaration checks.

use crate::internal::AccessDecl;

pub(super) fn access_mode_matches(access: &AccessDecl, write_expected: bool) -> bool {
    if write_expected {
        access.output_version.is_some()
    } else {
        access.output_version.is_none()
    }
}

use std::collections::HashMap;

use crate::{
    backend::{
        BindingResourceSemantic, ExecutionBackend, RenderObjectProvider, ResolvedBindingResource,
    },
    error::{RecordResult, RecordingErrorKind},
    handles::{BindingSetId, PassId},
    pass::{BindingResolverSink, BindingResource},
};

use super::{PhysicalResource, PhysicalResources, SessionObjects, shared::recording_error};

pub(super) struct ResolverBridge<'a, B: ExecutionBackend, O> {
    pass: PassId,
    device: crate::backend::DeviceIdentity,
    accesses: &'a HashMap<u64, &'a AccessDecl>,
    physical: &'a PhysicalResources<B>,
    objects: &'a O,
    session: &'a SessionObjects<B>,
}

impl<'a, B: ExecutionBackend, O> ResolverBridge<'a, B, O> {
    pub(super) fn new(
        pass: PassId,
        device: crate::backend::DeviceIdentity,
        accesses: &'a HashMap<u64, &'a AccessDecl>,
        physical: &'a PhysicalResources<B>,
        objects: &'a O,
        session: &'a SessionObjects<B>,
    ) -> Self {
        Self {
            pass,
            device,
            accesses,
            physical,
            objects,
            session,
        }
    }
}

impl<B, O> BindingResolverSink for ResolverBridge<'_, B, O>
where
    B: ExecutionBackend,
    O: RenderObjectProvider<B>,
{
    fn resolve_bindings(
        &mut self,
        recipe: BindingSetId,
        resources: &[BindingResource<'_>],
        dynamic_offsets: &[u32],
    ) -> RecordResult<(u64, u64)> {
        let mut resolved = Vec::with_capacity(resources.len());
        for resource in resources {
            let (handle, texture_expected, write_expected) = match resource {
                BindingResource::TextureRead(handle) => (handle.0, true, false),
                BindingResource::TextureWrite(handle) => (handle.0, true, true),
                BindingResource::TextureReadWrite(handle) => (handle.0, true, true),
                BindingResource::BufferRead(handle) => (handle.0, false, false),
                BindingResource::BufferWrite(handle) => (handle.0, false, true),
                BindingResource::BufferReadWrite(handle) => (handle.0, false, true),
            };
            let access = self.accesses.get(&handle).copied().ok_or_else(|| {
                recording_error(
                    RecordingErrorKind::ForeignOrUndeclaredPassAccess,
                    self.pass,
                    None,
                    "binding resource was not declared by this pass",
                )
            })?;
            if !access_mode_matches(access, write_expected) {
                return Err(recording_error(
                    RecordingErrorKind::DeclaredUseMismatch,
                    self.pass,
                    Some(access.resource),
                    "binding access mode differs from its declared handle",
                ));
            }
            let semantic = match access.semantic {
                crate::internal::AccessSemantic::TextureRead(use_) => {
                    BindingResourceSemantic::TextureRead(use_)
                }
                crate::internal::AccessSemantic::TextureWrite(use_) => {
                    BindingResourceSemantic::TextureWrite(use_)
                }
                crate::internal::AccessSemantic::TextureReadWrite(use_) => {
                    BindingResourceSemantic::TextureReadWrite(use_)
                }
                crate::internal::AccessSemantic::BufferRead(use_) => {
                    BindingResourceSemantic::BufferRead(use_)
                }
                crate::internal::AccessSemantic::BufferWrite(use_) => {
                    BindingResourceSemantic::BufferWrite(use_)
                }
                crate::internal::AccessSemantic::BufferReadWrite(use_) => {
                    BindingResourceSemantic::BufferReadWrite(use_)
                }
                _ => {
                    return Err(recording_error(
                        RecordingErrorKind::DeclaredUseMismatch,
                        self.pass,
                        Some(access.resource),
                        "binding handle has no bindable resource semantic",
                    ));
                }
            };
            match (
                &self.physical[&access.resource],
                texture_expected,
                access.range,
            ) {
                (
                    PhysicalResource::Texture { physical, .. },
                    true,
                    crate::internal::DeclRange::Texture(range),
                ) => resolved.push(ResolvedBindingResource::Texture {
                    physical,
                    range,
                    semantic,
                }),
                (
                    PhysicalResource::Buffer { physical, .. },
                    false,
                    crate::internal::DeclRange::Buffer(range),
                ) => resolved.push(ResolvedBindingResource::Buffer {
                    physical,
                    range,
                    semantic,
                }),
                _ => {
                    return Err(recording_error(
                        RecordingErrorKind::DeclaredUseMismatch,
                        self.pass,
                        Some(access.resource),
                        "binding resource kind differs from its declared handle",
                    ));
                }
            }
        }
        let binding = self.objects.bindings(recipe, &resolved, dynamic_offsets)?;
        if binding.device != self.device {
            return Err(recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                self.pass,
                None,
                "binding recipe belongs to another device",
            ));
        }
        let mut bindings = self.session.bindings.borrow_mut();
        let ticket = bindings.len() as u64;
        bindings.push(binding);
        Ok((ticket, self.session.session))
    }
}
