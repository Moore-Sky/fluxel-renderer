//! Raster-command argument checks.

use std::ops::Range;

use crate::pass::Viewport;

pub(super) fn valid_viewport(viewport: Viewport) -> bool {
    viewport.x.is_finite()
        && viewport.y.is_finite()
        && viewport.width.is_finite()
        && viewport.height.is_finite()
        && viewport.min_depth.is_finite()
        && viewport.max_depth.is_finite()
        && viewport.width > 0.0
        && viewport.height > 0.0
        && (0.0..=1.0).contains(&viewport.min_depth)
        && (0.0..=1.0).contains(&viewport.max_depth)
        && viewport.min_depth <= viewport.max_depth
}

pub(super) fn valid_scissor(scissor: ScissorRect) -> bool {
    scissor.width != 0
        && scissor.height != 0
        && scissor.x.checked_add(scissor.width).is_some()
        && scissor.y.checked_add(scissor.height).is_some()
}

fn non_empty(range: &Range<u32>) -> bool {
    range.start < range.end
}

use crate::{
    access::{BufferRange, BufferReadUse},
    backend::{ExecutionBackend, RenderObjectProvider},
    error::{RecordResult, RecordingErrorKind},
    handles::{BufferRead, RasterPipelineId, ResourceId},
    internal::{AccessDecl, AccessSemantic, DeclRange},
    pass::{RasterCommandSink, ScissorRect},
    rhi::IndexFormat,
};

use super::{
    PhysicalResource, PhysicalResources,
    shared::{CommandBridge, recording_error},
};

pub(super) struct RasterBridge<'a, B: ExecutionBackend, O> {
    common: CommandBridge<'a, B, O>,
}

impl<'a, B: ExecutionBackend, O> RasterBridge<'a, B, O> {
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

impl<B, O> RasterCommandSink for RasterBridge<'_, B, O>
where
    B: ExecutionBackend,
    O: RenderObjectProvider<B>,
{
    fn set_pipeline(&mut self, pipeline: RasterPipelineId) -> RecordResult {
        let pipeline = self.common.objects.raster_pipeline(pipeline)?;
        if pipeline.device != self.common.device {
            return Err(recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                self.common.pass,
                None,
                "raster pipeline belongs to another device",
            ));
        }
        self.common
            .backend
            .set_raster_pipeline(self.common.encoder, &pipeline.physical)
            .map_err(|error| self.common.fail_backend(error))?;
        self.common.session.leases.borrow_mut().push(pipeline.lease);
        Ok(())
    }
    fn set_bindings(&mut self, ticket: u64, session: u64) -> RecordResult {
        self.common.set_bindings(ticket, session)
    }
    fn set_vertex_buffer(&mut self, slot: u32, buffer: &BufferRead) -> RecordResult {
        let access = self.common.access(buffer.0)?;
        if !matches!(
            access.semantic,
            AccessSemantic::BufferRead(BufferReadUse::Vertex)
        ) {
            return Err(recording_error(
                RecordingErrorKind::DeclaredUseMismatch,
                self.common.pass,
                Some(access.resource),
                "vertex buffer command requires a Vertex read declaration",
            ));
        }
        let (physical, offset) = buffer_physical(self.common.physical, access);
        self.common
            .backend
            .set_vertex_buffer(self.common.encoder, slot, physical, offset)
            .map_err(|error| self.common.fail_backend(error))
    }
    fn set_index_buffer(&mut self, buffer: &BufferRead, format: IndexFormat) -> RecordResult {
        let access = self.common.access(buffer.0)?;
        if !matches!(
            access.semantic,
            AccessSemantic::BufferRead(BufferReadUse::Index)
        ) {
            return Err(recording_error(
                RecordingErrorKind::DeclaredUseMismatch,
                self.common.pass,
                Some(access.resource),
                "index buffer command requires an Index read declaration",
            ));
        }
        let (physical, offset) = buffer_physical(self.common.physical, access);
        self.common
            .backend
            .set_index_buffer(self.common.encoder, physical, offset, format)
            .map_err(|error| self.common.fail_backend(error))
    }
    fn set_viewport(&mut self, viewport: Viewport) -> RecordResult {
        if !valid_viewport(viewport) {
            return Err(recording_error(
                RecordingErrorKind::InvalidCommandArgument,
                self.common.pass,
                None,
                "viewport is non-finite or has invalid bounds",
            ));
        }
        self.common
            .backend
            .set_viewport(self.common.encoder, viewport)
            .map_err(|error| self.common.fail_backend(error))
    }
    fn set_scissor(&mut self, scissor: ScissorRect) -> RecordResult {
        if !valid_scissor(scissor) {
            return Err(recording_error(
                RecordingErrorKind::InvalidCommandArgument,
                self.common.pass,
                None,
                "scissor rectangle must be non-empty and not overflow",
            ));
        }
        self.common
            .backend
            .set_scissor(self.common.encoder, scissor)
            .map_err(|error| self.common.fail_backend(error))
    }
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> RecordResult {
        if !non_empty(&vertices) || !non_empty(&instances) {
            return Err(recording_error(
                RecordingErrorKind::InvalidCommandArgument,
                self.common.pass,
                None,
                "draw vertex and instance ranges must be non-empty",
            ));
        }
        self.common
            .backend
            .draw(self.common.encoder, vertices, instances)
            .map_err(|error| self.common.fail_backend(error))
    }
    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> RecordResult {
        if !non_empty(&indices) || !non_empty(&instances) {
            return Err(recording_error(
                RecordingErrorKind::InvalidCommandArgument,
                self.common.pass,
                None,
                "indexed draw index and instance ranges must be non-empty",
            ));
        }
        self.common
            .backend
            .draw_indexed(self.common.encoder, indices, base_vertex, instances)
            .map_err(|error| self.common.fail_backend(error))
    }
}

pub(super) fn texture<B: ExecutionBackend>(
    physical: &PhysicalResources<B>,
    resource: ResourceId,
) -> &B::Texture {
    match &physical[&resource] {
        PhysicalResource::Texture { physical, .. } => physical,
        PhysicalResource::Buffer { .. } => unreachable!(),
    }
}

fn buffer_physical<'a, B: ExecutionBackend>(
    physical: &'a PhysicalResources<B>,
    access: &AccessDecl,
) -> (&'a B::Buffer, u64) {
    match &physical[&access.resource] {
        PhysicalResource::Buffer { physical, .. } => {
            let offset = match access.range {
                DeclRange::Buffer(BufferRange::Whole) => 0,
                DeclRange::Buffer(BufferRange::Bytes { offset, .. }) => offset,
                _ => unreachable!(),
            };
            (physical, offset)
        }
        PhysicalResource::Texture { .. } => unreachable!(),
    }
}
