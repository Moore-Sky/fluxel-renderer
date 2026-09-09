//! Compute-command argument checks.

/// Returns whether the current execution milestone accepts the dispatch size.
///
/// The graph currently permits every representable non-negative workgroup
/// count; this boundary exists so future backend limits remain compute-local.
pub(super) fn valid_dispatch(_groups: [u32; 3]) -> bool {
    true
}

use crate::{
    backend::{ExecutionBackend, RenderObjectProvider},
    error::{RecordResult, RecordingErrorKind},
    handles::ComputePipelineId,
    pass::ComputeCommandSink,
};

use super::shared::{CommandBridge, recording_error};

pub(super) struct ComputeBridge<'a, B: ExecutionBackend, O> {
    common: CommandBridge<'a, B, O>,
}

impl<'a, B: ExecutionBackend, O> ComputeBridge<'a, B, O> {
    pub(super) fn new(common: CommandBridge<'a, B, O>) -> Self {
        Self { common }
    }
    pub(super) fn finish(
        &mut self,
        callback: RecordResult,
    ) -> Result<(), crate::backend::ExecutionError<B::Error>> {
        self.common.finish(callback)
    }
}

impl<B, O> ComputeCommandSink for ComputeBridge<'_, B, O>
where
    B: ExecutionBackend,
    O: RenderObjectProvider<B>,
{
    fn set_pipeline(&mut self, pipeline: ComputePipelineId) -> RecordResult {
        let pipeline = self.common.objects.compute_pipeline(pipeline)?;
        if pipeline.device != self.common.device {
            return Err(recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                self.common.pass,
                None,
                "compute pipeline belongs to another device",
            ));
        }
        self.common
            .backend
            .set_compute_pipeline(self.common.encoder, &pipeline.physical)
            .map_err(|error| self.common.fail_backend(error))?;
        self.common.session.leases.borrow_mut().push(pipeline.lease);
        Ok(())
    }
    fn set_bindings(&mut self, ticket: u64, session: u64) -> RecordResult {
        self.common.set_bindings(ticket, session)
    }
    fn dispatch(&mut self, groups: [u32; 3]) -> RecordResult {
        if !valid_dispatch(groups) {
            return Err(recording_error(
                RecordingErrorKind::InvalidCommandArgument,
                self.common.pass,
                None,
                "dispatch group count is unsupported",
            ));
        }
        self.common
            .backend
            .dispatch(self.common.encoder, groups)
            .map_err(|error| self.common.fail_backend(error))
    }
}
