#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    access::{
        AccessMode, BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureRange,
        TextureReadUse, TextureReadWriteUse, TextureWriteUse, WriteCoverage,
    },
    handles::{
        ExportBufferSlot, ExportTextureSlot, ImportBufferSlot, ImportTextureSlot, PassId,
        PresentTarget, ResourceId,
    },
    pass::{
        ColorAttachmentDesc, ComputeCommands, CopyCommands, DepthStencilAttachmentDesc, PassKind,
        PassResourceResolver, RasterCommands,
    },
    recipe::{ComputeExecute, CopyExecute, PassData, RasterExecute},
    resource::{
        ExportBufferContract, ExportTextureContract, ImportBufferContract, ImportTextureContract,
        PresentContract, SurfaceTextureContract,
    },
    rhi::{BufferDesc, TextureDesc},
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ResourceKind {
    Texture(TextureDesc),
    Buffer(BufferDesc),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ResourceOrigin {
    Transient,
    TextureImport(ImportTextureSlot, ImportTextureContract),
    Surface(ImportTextureSlot, SurfaceTextureContract),
    BufferImport(ImportBufferSlot, ImportBufferContract),
}

#[derive(Clone, Debug)]
pub(crate) struct ResourceDecl {
    pub id: ResourceId,
    pub name: String,
    pub kind: ResourceKind,
    pub origin: ResourceOrigin,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DeclRange {
    Texture(TextureRange),
    Buffer(BufferRange),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AccessSemantic {
    TextureRead(TextureReadUse),
    TextureWrite(TextureWriteUse),
    TextureReadWrite(TextureReadWriteUse),
    BufferRead(BufferReadUse),
    BufferWrite(BufferWriteUse),
    BufferReadWrite(BufferReadWriteUse),
    ColorAttachment { index: u32 },
    DepthStencilAttachment { depth: bool, stencil: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum AccessDetails {
    None,
    ColorAttachment(ColorAttachmentDesc),
    DepthStencilAttachment(DepthStencilAttachmentDesc),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AccessDecl {
    pub handle: u64,
    pub resource: ResourceId,
    pub input_version: u32,
    pub output_version: Option<u32>,
    pub mode: AccessMode,
    pub range: DeclRange,
    pub semantic: AccessSemantic,
    pub details: AccessDetails,
    pub read_required: bool,
    pub coverage: WriteCoverage,
    pub invalidate_before: bool,
    pub discard_after: bool,
}

pub(crate) trait RasterRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut RasterCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}
pub(crate) trait ComputeRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut ComputeCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}
pub(crate) trait CopyRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut CopyCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}

struct Retained<D, E> {
    data: D,
    execute: E,
}
impl<F, D, E> RasterRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: RasterExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut RasterCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}
impl<F, D, E> ComputeRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: ComputeExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut ComputeCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}
impl<F, D, E> CopyRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: CopyExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut CopyCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}

pub(crate) enum StoredExecute<F> {
    Raster(Arc<dyn RasterRecipe<F>>),
    Compute(Arc<dyn ComputeRecipe<F>>),
    Copy(Arc<dyn CopyRecipe<F>>),
}

pub(crate) fn retain_raster<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: RasterExecute<F, D>,
{
    StoredExecute::Raster(Arc::new(Retained { data, execute }))
}
pub(crate) fn retain_compute<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: ComputeExecute<F, D>,
{
    StoredExecute::Compute(Arc::new(Retained { data, execute }))
}
pub(crate) fn retain_copy<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: CopyExecute<F, D>,
{
    StoredExecute::Copy(Arc::new(Retained { data, execute }))
}

impl<F> Clone for StoredExecute<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Raster(value) => Self::Raster(Arc::clone(value)),
            Self::Compute(value) => Self::Compute(Arc::clone(value)),
            Self::Copy(value) => Self::Copy(Arc::clone(value)),
        }
    }
}

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
