//! Instantiates immutable graphs and executes them as serial single-queue frame transactions.
//!
//! This layer owns frame inputs, resource resolution, portable transition lowering, callback
//! recording, and lease retirement; native objects and backend-specific synchronization remain
//! behind [`ExecutionBackend`](crate::backend::ExecutionBackend). One execution is bound to the
//! compiled graph identity and one backend lock spans resolution through submission, preventing
//! re-entrant callbacks and partial submissions. Recording and submission collaborate here by
//! retaining every resolved lease until the backend reports GPU completion.

mod recording;
mod run;
mod submission;

use std::{marker::PhantomData, rc::Rc};

use crate::{
    CompiledGraph,
    handles::{
        BufferBindingId, ImportBufferSlot, ImportTextureSlot, SurfaceBindingId, TextureBindingId,
    },
};

pub use run::{ExecutedFrame, ExportedBuffer, ExportedTexture, FrameExecutor, FrameExports};
pub use submission::FrameSubmission;

/// Owner-thread recording mode for thread-bound backends and frame data.
///
/// The `Rc` marker makes [`FrameExecution`] instantiated in this mode `!Send`
/// even when its frame data is `Send`. This keeps a frame tied to the thread
/// that owns an immediate-context backend; use [`SendMode`] for work that may
/// cross threads.
///
/// ```compile_fail
/// use fluxel_rendergraph::{FrameExecution, Local};
///
/// fn requires_send<T: Send>() {}
/// requires_send::<FrameExecution<(), Local>>();
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Local(PhantomData<Rc<()>>);

/// Cross-thread recording mode for frame data shared by independent jobs.
///
/// `FrameExecution<F, SendMode>` is `Send` when `F` is `Send`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SendMode;

/// Owned values supplied to one independent execution of a compiled graph.
///
/// Resource identities refer to renderer-owned registry entries whose leases
/// must outlive GPU completion. They are not native API handles.
pub struct FrameInputs<F> {
    pub(crate) frame_data: F,
    pub(crate) textures: Vec<(ImportTextureSlot, TextureBindingId)>,
    pub(crate) buffers: Vec<(ImportBufferSlot, BufferBindingId)>,
    pub(crate) surfaces: Vec<(ImportTextureSlot, SurfaceBindingId)>,
}

impl<F> FrameInputs<F> {
    /// Creates one owned frame-input set.
    pub fn new(frame_data: F) -> Self {
        Self {
            frame_data,
            textures: Vec::new(),
            buffers: Vec::new(),
            surfaces: Vec::new(),
        }
    }

    /// Binds a caller-owned physical texture to one retained import slot.
    pub fn bind_texture(
        &mut self,
        slot: ImportTextureSlot,
        texture: TextureBindingId,
    ) -> &mut Self {
        self.textures.retain(|(bound, _)| *bound != slot);
        self.textures.push((slot, texture));
        self
    }

    /// Binds a caller-owned physical buffer to one retained import slot.
    pub fn bind_buffer(&mut self, slot: ImportBufferSlot, buffer: BufferBindingId) -> &mut Self {
        self.buffers.retain(|(bound, _)| *bound != slot);
        self.buffers.push((slot, buffer));
        self
    }

    /// Binds one acquired presentation image to a surface import slot.
    pub fn bind_surface(
        &mut self,
        slot: ImportTextureSlot,
        surface: SurfaceBindingId,
    ) -> &mut Self {
        self.surfaces.retain(|(bound, _)| *bound != slot);
        self.surfaces.push((slot, surface));
        self
    }
}

/// One independent, not-yet-submitted instantiation of a compiled plan.
///
/// CPU completion is distinct from the GPU completion returned after submit.
/// Its mode parameter carries the thread-affinity contract selected by the
/// constructor: [`Local`] is owner-thread-only and [`SendMode`] may move when
/// its frame data may move.
pub struct FrameExecution<F, M> {
    pub(crate) graph_identity: u64,
    pub(crate) inputs: FrameInputs<F>,
    pub(crate) marker: PhantomData<M>,
}

impl<F> CompiledGraph<F> {
    /// Instantiates an owner-thread frame run from owned inputs.
    pub fn instantiate_local(&self, inputs: FrameInputs<F>) -> FrameExecution<F, Local>
    where
        F: 'static,
    {
        FrameExecution {
            graph_identity: self.identity,
            inputs,
            marker: PhantomData,
        }
    }

    /// Instantiates a frame run whose data may be shared by Send recording jobs.
    pub fn instantiate_send(&self, inputs: FrameInputs<F>) -> FrameExecution<F, SendMode>
    where
        F: Send + Sync + 'static,
    {
        FrameExecution {
            graph_identity: self.identity,
            inputs,
            marker: PhantomData,
        }
    }
}

impl<F, M> FrameExecution<F, M> {
    /// Returns immutable frame data owned by this execution instance.
    pub fn frame_data(&self) -> &F {
        &self.inputs.frame_data
    }
}

#[cfg(test)]
mod tests;
