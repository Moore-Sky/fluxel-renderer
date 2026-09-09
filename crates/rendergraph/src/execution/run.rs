//! Frame-execution orchestration.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    CompiledGraph,
    backend::{ExecutionBackend, ExecutionError, FrameResourceProvider, RenderObjectProvider},
    internal::{ResourceOrigin, RootDecl},
};

use super::{
    FrameExecution,
    recording::{RecordPassRequest, record_pass},
    submission::{FrameSubmission, RetirementInbox, drain_retirement_inbox, try_lock_backend},
};

#[path = "exports.rs"]
mod exports;
#[path = "resolution.rs"]
mod resolution;
#[path = "transitions.rs"]
mod transitions;

pub use exports::{ExecutedFrame, ExportedBuffer, ExportedTexture, FrameExports};

use exports::build_exports;
use resolution::{live_resources, resolve_resources};
use transitions::emit_transitions;

/// Serial single-queue executor for one backend device.
pub struct FrameExecutor<B: ExecutionBackend> {
    backend: Arc<Mutex<B>>,
    retirement_inbox: RetirementInbox<B>,
}

impl<B: ExecutionBackend> FrameExecutor<B> {
    /// Creates an executor owning one backend device instance.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            retirement_inbox: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Tries to borrow the backend for diagnostics or explicit test control.
    ///
    /// This returns `None` while a frame is executing, so providers and pass
    /// callbacks cannot accidentally deadlock by re-entering the executor.
    pub fn try_backend(&self) -> Option<std::sync::MutexGuard<'_, B>> {
        try_lock_backend(&self.backend).ok()
    }

    /// Polls the backend retirement queue.
    pub fn collect_retired(&self) -> Result<usize, ExecutionError<B::Error>> {
        let mut backend =
            try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        drain_retirement_inbox(&mut *backend, &self.retirement_inbox);
        backend.collect_retired().map_err(ExecutionError::Backend)
    }

    /// Executes one frame instance through the retained single-queue plan.
    pub fn execute<F, M, R, O>(
        &self,
        graph: &CompiledGraph<F>,
        execution: FrameExecution<F, M>,
        resources: &R,
        objects: &O,
    ) -> Result<ExecutedFrame<B>, ExecutionError<B::Error>>
    where
        R: FrameResourceProvider<B>,
        O: RenderObjectProvider<B>,
    {
        if execution.graph_identity != graph.identity {
            return Err(ExecutionError::WrongCompiledGraph);
        }
        let live = live_resources(graph);
        if graph.resources.iter().any(|resource| {
            live.contains(&resource.id) && matches!(resource.origin, ResourceOrigin::Surface(_, _))
        }) || graph
            .roots
            .iter()
            .any(|root| matches!(root, RootDecl::Present(_, _, _, _)))
        {
            return Err(ExecutionError::UnsupportedExecutionFeature(
                "surface and present are deferred to the 0.3 line",
            ));
        }
        let queue =
            graph
                .execution_plan()
                .queue()
                .ok_or(ExecutionError::UnsupportedExecutionFeature(
                    "an empty graph has no submission queue",
                ))?;
        let mut backend =
            try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        if graph.capability_fingerprint != backend.capabilities().fingerprint() {
            return Err(ExecutionError::CapabilityMismatch);
        }
        drain_retirement_inbox(&mut *backend, &self.retirement_inbox);
        backend.collect_retired().map_err(ExecutionError::Backend)?;

        let resolution::ResolvedFrameResources {
            physical,
            resource_leases,
            mut retained,
        } = resolve_resources(graph, &execution, resources, &mut *backend, &live)?;
        let exports = build_exports(graph, &physical, &resource_leases);
        let mut encoder = backend
            .begin_encoder(queue)
            .map_err(ExecutionError::Backend)?;
        let passes: HashMap<_, _> = graph.passes.iter().map(|pass| (pass.id, pass)).collect();
        for planned in graph.execution_plan().passes() {
            emit_transitions(&mut *backend, &mut encoder, &physical, &planned.transitions)?;
            record_pass(RecordPassRequest {
                backend: &mut *backend,
                encoder: &mut encoder,
                pass: passes[&planned.pass],
                planned,
                physical: &physical,
                objects,
                frame: &execution.inputs.frame_data,
                retained: &mut retained,
            })?;
        }
        emit_transitions(
            &mut *backend,
            &mut encoder,
            &physical,
            graph.execution_plan().final_transitions(),
        )?;
        let command_buffer = backend
            .finish_encoder(encoder)
            .map_err(ExecutionError::Backend)?;
        let completion = backend
            .submit(queue, command_buffer)
            .map_err(ExecutionError::Backend)?;
        drop(backend);

        Ok(ExecutedFrame {
            exports,
            submission: FrameSubmission::new(
                Arc::clone(&self.backend),
                Arc::clone(&self.retirement_inbox),
                completion,
                retained,
            ),
        })
    }
}
