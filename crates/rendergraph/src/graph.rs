//! Editable render graph declarations and the compile entry point.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    compile::{CompileResult, ExplicitOrder},
    handles::{
        BufferVersion, ExportBufferSlot, ExportTextureSlot, ImportBufferSlot, ImportTextureSlot,
        PassId, PresentTarget, ResourceId, TextureVersion,
    },
    internal::{
        OrderDecl, PassBuildState, PassDecl, ResourceDecl, ResourceKind, ResourceOrigin, RootDecl,
        retain_compute, retain_copy, retain_raster,
    },
    pass::{ComputePassBuilder, CopyPassBuilder, RasterPassBuilder},
    recipe::{
        ComputeExecute, ComputeSetup, CopyExecute, CopySetup, PassData, RasterExecute, RasterSetup,
    },
    resource::{
        ExportBufferContract, ExportTextureContract, ImportBufferContract, ImportTextureContract,
        ImportedBuffer, ImportedTexture, PresentContract, SurfaceTextureContract,
    },
    rhi::{BufferDesc, DeviceCapabilities, TextureDesc},
};

/// Reason for retaining a pass that has an observable non-resource effect.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SideEffectReason {
    /// The pass participates in an external protocol not represented by a GPU resource.
    ExternalProtocol(String),
    /// The pass records diagnostic or profiling work that must not be culled.
    Diagnostic(String),
}

/// Reason for ordering two passes that do not share a declared GPU resource.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplicitOrderReason {
    /// An external protocol requires the ordering.
    ExternalProtocol(String),
    /// A diagnostic boundary requires the ordering.
    Diagnostic(String),
}

/// One declared pass and the values its setup callback exposes downstream.
#[derive(Debug)]
pub struct DeclaredPass<O> {
    /// Stable pass identity used by diagnostics and explicit order edges.
    pub id: PassId,
    /// Setup output, normally one or more successor resource versions.
    pub output: O,
}

/// Mutable authoring surface for a retained render graph.
///
/// Compiling borrows this declaration graph and produces an immutable snapshot;
/// later edits do not change older compiled graphs.
pub struct RenderGraph<F = ()> {
    pub(crate) resources: Vec<ResourceDecl>,
    pub(crate) passes: Vec<PassDecl<F>>,
    pub(crate) roots: Vec<RootDecl>,
    pub(crate) orders: Vec<OrderDecl>,
    _frame_data: std::marker::PhantomData<fn(&F)>,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

impl<F: 'static> RenderGraph<F> {
    /// Creates an empty editable render graph.
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            roots: Vec::new(),
            orders: Vec::new(),
            _frame_data: std::marker::PhantomData,
        }
    }

    /// Creates an uninitialized transient texture version.
    pub fn create_texture(&mut self, name: impl Into<String>, desc: TextureDesc) -> TextureVersion {
        let id = ResourceId(next_id());
        self.resources.push(ResourceDecl {
            id,
            name: name.into(),
            kind: ResourceKind::Texture(desc),
            origin: ResourceOrigin::Transient,
        });
        TextureVersion::new(id, 0)
    }

    /// Creates an uninitialized transient buffer version.
    pub fn create_buffer(&mut self, name: impl Into<String>, desc: BufferDesc) -> BufferVersion {
        let id = ResourceId(next_id());
        self.resources.push(ResourceDecl {
            id,
            name: name.into(),
            kind: ResourceKind::Buffer(desc),
            origin: ResourceOrigin::Transient,
        });
        BufferVersion::new(id, 0)
    }

    /// Declares a retained external texture slot.
    pub fn import_texture_slot(
        &mut self,
        name: impl Into<String>,
        contract: ImportTextureContract,
    ) -> ImportedTexture {
        let id = ResourceId(next_id());
        let slot = ImportTextureSlot(next_id());
        self.resources.push(ResourceDecl {
            id,
            name: name.into(),
            kind: ResourceKind::Texture(contract.descriptor),
            origin: ResourceOrigin::TextureImport(slot, contract),
        });
        ImportedTexture {
            slot,
            version: TextureVersion::new(id, 0),
        }
    }

    /// Declares a retained surface-texture slot that may become a presentation root.
    pub fn import_surface_texture_slot(
        &mut self,
        name: impl Into<String>,
        contract: SurfaceTextureContract,
    ) -> ImportedTexture {
        let id = ResourceId(next_id());
        let slot = ImportTextureSlot(next_id());
        self.resources.push(ResourceDecl {
            id,
            name: name.into(),
            kind: ResourceKind::Texture(contract.descriptor),
            origin: ResourceOrigin::Surface(slot, contract),
        });
        ImportedTexture {
            slot,
            version: TextureVersion::new(id, 0),
        }
    }

    /// Declares a retained external buffer slot.
    pub fn import_buffer_slot(
        &mut self,
        name: impl Into<String>,
        contract: ImportBufferContract,
    ) -> ImportedBuffer {
        let id = ResourceId(next_id());
        let slot = ImportBufferSlot(next_id());
        self.resources.push(ResourceDecl {
            id,
            name: name.into(),
            kind: ResourceKind::Buffer(contract.descriptor),
            origin: ResourceOrigin::BufferImport(slot, contract),
        });
        ImportedBuffer {
            slot,
            version: BufferVersion::new(id, 0),
        }
    }

    /// Adds a repeatable raster pass.
    ///
    /// Setup runs while authoring the graph. The graph retains `data` and calls
    /// `execute` once per future frame, with access only to the handles declared
    /// by setup.
    pub fn add_raster_pass<O, D: PassData>(
        &mut self,
        name: impl Into<String>,
        setup: impl RasterSetup<O, D>,
        execute: impl RasterExecute<F, D>,
    ) -> DeclaredPass<O> {
        let id = PassId(next_id());
        let mut builder = RasterPassBuilder::new(PassBuildState::new(id));
        let (output, data) = setup(&mut builder);
        let state = builder.into_state();
        self.passes.push(PassDecl {
            id,
            name: name.into(),
            kind: crate::pass::PassKind::Raster,
            accesses: state.accesses,
            recipe: retain_raster(data, execute),
            marker: std::marker::PhantomData,
        });
        DeclaredPass { id, output }
    }

    /// Adds a repeatable compute pass.
    pub fn add_compute_pass<O, D: PassData>(
        &mut self,
        name: impl Into<String>,
        setup: impl ComputeSetup<O, D>,
        execute: impl ComputeExecute<F, D>,
    ) -> DeclaredPass<O> {
        let id = PassId(next_id());
        let mut builder = ComputePassBuilder::new(PassBuildState::new(id));
        let (output, data) = setup(&mut builder);
        let state = builder.into_state();
        self.passes.push(PassDecl {
            id,
            name: name.into(),
            kind: crate::pass::PassKind::Compute,
            accesses: state.accesses,
            recipe: retain_compute(data, execute),
            marker: std::marker::PhantomData,
        });
        DeclaredPass { id, output }
    }

    /// Adds a repeatable copy pass.
    pub fn add_copy_pass<O, D: PassData>(
        &mut self,
        name: impl Into<String>,
        setup: impl CopySetup<O, D>,
        execute: impl CopyExecute<F, D>,
    ) -> DeclaredPass<O> {
        let id = PassId(next_id());
        let mut builder = CopyPassBuilder::new(PassBuildState::new(id));
        let (output, data) = setup(&mut builder);
        let state = builder.into_state();
        self.passes.push(PassDecl {
            id,
            name: name.into(),
            kind: crate::pass::PassKind::Copy,
            accesses: state.accesses,
            recipe: retain_copy(data, execute),
            marker: std::marker::PhantomData,
        });
        DeclaredPass { id, output }
    }

    /// Marks a texture version as an external output and graph root.
    pub fn export_texture(
        &mut self,
        version: TextureVersion,
        contract: ExportTextureContract,
    ) -> ExportTextureSlot {
        let slot = ExportTextureSlot(next_id());
        self.roots.push(RootDecl::Texture(
            slot,
            version.resource,
            version.version,
            contract,
        ));
        slot
    }

    /// Marks a buffer version as an external output and graph root.
    pub fn export_buffer(
        &mut self,
        version: BufferVersion,
        contract: ExportBufferContract,
    ) -> ExportBufferSlot {
        let slot = ExportBufferSlot(next_id());
        self.roots.push(RootDecl::Buffer(
            slot,
            version.resource,
            version.version,
            contract,
        ));
        slot
    }

    /// Marks a surface-owned texture version as a presentation root.
    ///
    /// Presentation readiness is not whole-frame completion.
    pub fn present(&mut self, version: TextureVersion, contract: PresentContract) -> PresentTarget {
        let target = PresentTarget(next_id());
        self.roots.push(RootDecl::Present(
            target,
            version.resource,
            version.version,
            contract,
        ));
        target
    }

    /// Retains a pass because of a declared non-resource side effect.
    pub fn mark_side_effect(&mut self, pass: PassId, reason: SideEffectReason) {
        self.roots.push(RootDecl::SideEffect(pass, reason));
    }

    /// Adds an ordering edge for passes with no shared declared GPU resource.
    pub fn depends_on(
        &mut self,
        before: PassId,
        after: PassId,
        reason: ExplicitOrderReason,
    ) -> ExplicitOrder {
        let reason = match reason {
            ExplicitOrderReason::ExternalProtocol(value)
            | ExplicitOrderReason::Diagnostic(value) => value,
        };
        self.orders.push(OrderDecl {
            before,
            after,
            reason: reason.clone(),
        });
        ExplicitOrder {
            before,
            after,
            reason,
        }
    }

    /// Compiles an immutable graph snapshot for the supplied device capabilities.
    pub fn compile(&self, capabilities: &DeviceCapabilities) -> CompileResult<F> {
        crate::compile::compile_graph(self, capabilities)
    }
}

impl<F: 'static> Default for RenderGraph<F> {
    fn default() -> Self {
        Self::new()
    }
}
