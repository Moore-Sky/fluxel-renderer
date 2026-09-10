//! Private pass/root/order declarations and pass-build state.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    handles::{ExportBufferSlot, ExportTextureSlot, PassId, PresentTarget, ResourceId},
    pass::PassKind,
    resource::{ExportBufferContract, ExportTextureContract, PresentContract},
};

use super::{AccessDecl, StoredExecute};

pub(crate) struct PassDecl<F> {
    pub id: PassId,
    pub name: String,
    pub kind: PassKind,
    pub accesses: Vec<AccessDecl>,
    pub recipe: StoredExecute<F>,
    pub marker: std::marker::PhantomData<fn(&F)>,
}

impl<F> Clone for PassDecl<F> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            name: self.name.clone(),
            kind: self.kind,
            accesses: self.accesses.clone(),
            recipe: self.recipe.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<F> std::fmt::Debug for PassDecl<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PassDecl")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("accesses", &self.accesses)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RootDecl {
    Texture(ExportTextureSlot, ResourceId, u32, ExportTextureContract),
    Buffer(ExportBufferSlot, ResourceId, u32, ExportBufferContract),
    Present(PresentTarget, ResourceId, u32, PresentContract),
    SideEffect(PassId, crate::graph::SideEffectReason),
}

#[derive(Clone, Debug)]
pub(crate) struct OrderDecl {
    pub before: PassId,
    pub after: PassId,
    pub reason: String,
}

pub(crate) struct PassBuildState {
    pub pass: PassId,
    pub accesses: Vec<AccessDecl>,
}

static NEXT_ACCESS_ID: AtomicU64 = AtomicU64::new(1);

impl PassBuildState {
    pub fn new(pass: PassId) -> Self {
        Self {
            pass,
            accesses: Vec::new(),
        }
    }

    pub fn push(&mut self, mut access: AccessDecl) -> u64 {
        let id = NEXT_ACCESS_ID.fetch_add(1, Ordering::Relaxed);
        access.handle = id;
        self.accesses.push(access);
        id
    }
}
