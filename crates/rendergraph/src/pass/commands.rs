//! Typed recording command facades and their private sinks.

use std::ops::Range;

use crate::{
    RecordResult,
    access::{BufferCopyRegion, TextureCopyRegion},
    handles::{
        BufferRead, BufferWrite, ComputePipelineId, RasterPipelineId, TextureRead, TextureWrite,
    },
    rhi::IndexFormat,
};

use super::{ResolvedBindings, ScissorRect, Viewport};

/// Commands available while recording one raster pass.
///
/// Resources are referenced only through access handles produced by that
/// pass's setup callback.
pub struct RasterCommands<'a> {
    sink: &'a mut dyn RasterCommandSink,
}

pub(crate) trait RasterCommandSink {
    fn set_pipeline(&mut self, pipeline: RasterPipelineId) -> RecordResult;
    fn set_bindings(&mut self, ticket: u64, session: u64) -> RecordResult;
    fn set_vertex_buffer(&mut self, slot: u32, buffer: &BufferRead) -> RecordResult;
    fn set_index_buffer(&mut self, buffer: &BufferRead, format: IndexFormat) -> RecordResult;
    fn set_viewport(&mut self, viewport: Viewport) -> RecordResult;
    fn set_scissor(&mut self, scissor: ScissorRect) -> RecordResult;
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> RecordResult;
    fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> RecordResult;
}

impl<'a> RasterCommands<'a> {
    /// Selects a renderer-owned raster pipeline.
    pub fn set_pipeline(&mut self, pipeline: RasterPipelineId) -> RecordResult {
        self.sink.set_pipeline(pipeline)
    }

    /// Applies bindings validated by the current pass resource resolver.
    pub fn set_bindings(&mut self, bindings: &ResolvedBindings<'_>) -> RecordResult {
        self.sink.set_bindings(bindings.ticket, bindings.session)
    }

    /// Selects a declared vertex-buffer read for one input slot.
    pub fn set_vertex_buffer(&mut self, slot: u32, buffer: &BufferRead) -> RecordResult {
        self.sink.set_vertex_buffer(slot, buffer)
    }

    /// Selects a declared index-buffer read.
    pub fn set_index_buffer(&mut self, buffer: &BufferRead, format: IndexFormat) -> RecordResult {
        self.sink.set_index_buffer(buffer, format)
    }

    /// Sets the dynamic viewport for subsequent raster commands.
    pub fn set_viewport(&mut self, viewport: Viewport) -> RecordResult {
        self.sink.set_viewport(viewport)
    }

    /// Sets the dynamic scissor rectangle for subsequent raster commands.
    pub fn set_scissor(&mut self, scissor: ScissorRect) -> RecordResult {
        self.sink.set_scissor(scissor)
    }

    /// Records a non-indexed draw.
    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) -> RecordResult {
        self.sink.draw(vertices, instances)
    }

    /// Records an indexed draw.
    pub fn draw_indexed(
        &mut self,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> RecordResult {
        self.sink.draw_indexed(indices, base_vertex, instances)
    }

    pub(crate) fn new(sink: &'a mut dyn RasterCommandSink) -> Self {
        Self { sink }
    }
}

/// Commands available while recording one compute pass.
pub struct ComputeCommands<'a> {
    sink: &'a mut dyn ComputeCommandSink,
}

pub(crate) trait ComputeCommandSink {
    fn set_pipeline(&mut self, pipeline: ComputePipelineId) -> RecordResult;
    fn set_bindings(&mut self, ticket: u64, session: u64) -> RecordResult;
    fn dispatch(&mut self, groups: [u32; 3]) -> RecordResult;
}

impl<'a> ComputeCommands<'a> {
    /// Selects a renderer-owned compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: ComputePipelineId) -> RecordResult {
        self.sink.set_pipeline(pipeline)
    }

    /// Applies bindings validated by the current pass resource resolver.
    pub fn set_bindings(&mut self, bindings: &ResolvedBindings<'_>) -> RecordResult {
        self.sink.set_bindings(bindings.ticket, bindings.session)
    }

    /// Records one compute dispatch.
    pub fn dispatch(&mut self, groups: [u32; 3]) -> RecordResult {
        self.sink.dispatch(groups)
    }

    pub(crate) fn new(sink: &'a mut dyn ComputeCommandSink) -> Self {
        Self { sink }
    }
}

/// Commands available while recording one copy pass.
pub struct CopyCommands<'a> {
    sink: &'a mut dyn CopyCommandSink,
}

pub(crate) trait CopyCommandSink {
    fn copy_texture(
        &mut self,
        source: &TextureRead,
        destination: &TextureWrite,
        region: TextureCopyRegion,
    ) -> RecordResult;
    fn copy_buffer(
        &mut self,
        source: &BufferRead,
        destination: &BufferWrite,
        region: BufferCopyRegion,
    ) -> RecordResult;
}

impl<'a> CopyCommands<'a> {
    /// Copies between declared texture source and destination accesses.
    pub fn copy_texture(
        &mut self,
        source: &TextureRead,
        destination: &TextureWrite,
        region: TextureCopyRegion,
    ) -> RecordResult {
        self.sink.copy_texture(source, destination, region)
    }

    /// Copies between declared buffer source and destination accesses.
    pub fn copy_buffer(
        &mut self,
        source: &BufferRead,
        destination: &BufferWrite,
        region: BufferCopyRegion,
    ) -> RecordResult {
        self.sink.copy_buffer(source, destination, region)
    }

    pub(crate) fn new(sink: &'a mut dyn CopyCommandSink) -> Self {
        Self { sink }
    }
}
