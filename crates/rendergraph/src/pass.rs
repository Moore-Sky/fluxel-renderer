//! Pass setup builders and future command-recording contexts.

use std::{marker::PhantomData, ops::Range};

use crate::{
    RecordResult,
    access::{
        BufferCopyRegion, BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse,
        TextureCopyRegion, TextureRange, TextureReadUse, TextureReadWriteUse, TextureWriteUse,
        WriteCoverage,
    },
    handles::{
        BindingSetId, BufferRead, BufferReadWrite, BufferVersion, BufferWrite, ComputePipelineId,
        RasterPipelineId, TextureRead, TextureReadWrite, TextureVersion, TextureWrite,
    },
    internal::{AccessDecl, AccessDetails, AccessSemantic, DeclRange, PassBuildState},
    rhi::IndexFormat,
};

/// Logical kind of commands recorded by one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// Rasterization and attachment work.
    Raster,
    /// Compute-dispatch work.
    Compute,
    /// Resource-copy work.
    Copy,
}

/// Contents operation performed when an attachment begins its pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    /// Preserves and reads the previous contents.
    Load,
    /// Initializes the selected range to one clear value.
    Clear(T),
    /// Does not preserve previous contents and does not imply initialization.
    DontCare,
}

/// Contents operation performed when an attachment ends its pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Makes valid produced contents available to later accesses.
    Store,
    /// Invalidates the selected contents after the pass.
    Discard,
}

/// Initial and final content operations for one attachment aspect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttachmentOps<T> {
    /// Initial content operation.
    pub load: LoadOp<T>,
    /// Final content operation.
    pub store: StoreOp,
    /// Definite shader/draw write coverage of the selected range.
    ///
    /// `Clear` initializes the full range regardless of this value. `Load`
    /// inherits valid input contents. `DontCare` needs `Full` before a later
    /// read can rely on newly initialized contents.
    pub write_coverage: WriteCoverage,
}

/// Declaration of one color attachment and its content operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachmentDesc {
    /// Zero-based color attachment index.
    pub index: u32,
    /// Attachment view selected from the logical texture.
    pub range: TextureRange,
    /// Initial and final color-content operations.
    pub operations: AttachmentOps<[f32; 4]>,
}

/// Declaration of one depth-stencil attachment and its content operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthStencilAttachmentDesc {
    /// Attachment view selected from the logical texture.
    pub range: TextureRange,
    /// Depth-aspect operations, or `None` when unused.
    pub depth: Option<AttachmentOps<f32>>,
    /// Stencil-aspect operations, or `None` when unused.
    pub stencil: Option<AttachmentOps<u32>>,
}

/// Dynamic raster viewport supplied while recording a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Viewport width in pixels.
    pub width: f32,
    /// Viewport height in pixels.
    pub height: f32,
    /// Minimum depth value.
    pub min_depth: f32,
    /// Maximum depth value.
    pub max_depth: f32,
}

/// Dynamic integer scissor rectangle supplied while recording a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Rectangle width in pixels.
    pub width: u32,
    /// Rectangle height in pixels.
    pub height: u32,
}

/// One declared resource supplied to a renderer-owned opaque binding recipe.
#[derive(Clone, Copy, Debug)]
pub enum BindingResource<'a> {
    /// A declared sampled or read-only texture.
    TextureRead(&'a TextureRead),
    /// A declared writable texture.
    TextureWrite(&'a TextureWrite),
    /// A declared read-modify-write texture.
    TextureReadWrite(&'a TextureReadWrite),
    /// A declared read-only buffer.
    BufferRead(&'a BufferRead),
    /// A declared writable buffer.
    BufferWrite(&'a BufferWrite),
    /// A declared read-modify-write buffer.
    BufferReadWrite(&'a BufferReadWrite),
}

/// A pass-validated opaque binding object ready for one command context.
pub struct ResolvedBindings<'a> {
    pub(crate) ticket: u64,
    pub(crate) session: u64,
    _marker: PhantomData<&'a mut ()>,
}

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

macro_rules! resource_declaration_methods {
    () => {
        /// Declares a texture read without creating a new logical version.
        pub fn read_texture(
            &mut self,
            input: &TextureVersion,
            usage: TextureReadUse,
            range: TextureRange,
        ) -> TextureRead {
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource: input.resource,
                input_version: input.version,
                output_version: None,
                mode: crate::access::AccessMode::Read,
                range: DeclRange::Texture(range),
                semantic: AccessSemantic::TextureRead(usage),
                details: AccessDetails::None,
                read_required: true,
                coverage: WriteCoverage::Unknown,
                invalidate_before: false,
                discard_after: false,
            });
            TextureRead(handle)
        }

        /// Declares a texture write and returns its successor version.
        pub fn write_texture(
            &mut self,
            input: TextureVersion,
            usage: TextureWriteUse,
            range: TextureRange,
            coverage: WriteCoverage,
        ) -> (TextureVersion, TextureWrite) {
            let output_version = input.version.saturating_add(1);
            let resource = input.resource;
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource,
                input_version: input.version,
                output_version: Some(output_version),
                mode: crate::access::AccessMode::Write,
                range: DeclRange::Texture(range),
                semantic: AccessSemantic::TextureWrite(usage),
                details: AccessDetails::None,
                read_required: false,
                coverage,
                invalidate_before: false,
                discard_after: false,
            });
            (
                TextureVersion::new(resource, output_version),
                TextureWrite(handle),
            )
        }

        /// Declares a texture read-modify-write and returns its successor version.
        pub fn read_write_texture(
            &mut self,
            input: TextureVersion,
            usage: TextureReadWriteUse,
            range: TextureRange,
        ) -> (TextureVersion, TextureReadWrite) {
            let output_version = input.version.saturating_add(1);
            let resource = input.resource;
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource,
                input_version: input.version,
                output_version: Some(output_version),
                mode: crate::access::AccessMode::ReadWrite,
                range: DeclRange::Texture(range),
                semantic: AccessSemantic::TextureReadWrite(usage),
                details: AccessDetails::None,
                read_required: true,
                coverage: WriteCoverage::Unknown,
                invalidate_before: false,
                discard_after: false,
            });
            (
                TextureVersion::new(resource, output_version),
                TextureReadWrite(handle),
            )
        }

        /// Declares a buffer read without creating a new logical version.
        pub fn read_buffer(
            &mut self,
            input: &BufferVersion,
            usage: BufferReadUse,
            range: BufferRange,
        ) -> BufferRead {
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource: input.resource,
                input_version: input.version,
                output_version: None,
                mode: crate::access::AccessMode::Read,
                range: DeclRange::Buffer(range),
                semantic: AccessSemantic::BufferRead(usage),
                details: AccessDetails::None,
                read_required: true,
                coverage: WriteCoverage::Unknown,
                invalidate_before: false,
                discard_after: false,
            });
            BufferRead(handle)
        }

        /// Declares a buffer write and returns its successor version.
        pub fn write_buffer(
            &mut self,
            input: BufferVersion,
            usage: BufferWriteUse,
            range: BufferRange,
            coverage: WriteCoverage,
        ) -> (BufferVersion, BufferWrite) {
            let output_version = input.version.saturating_add(1);
            let resource = input.resource;
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource,
                input_version: input.version,
                output_version: Some(output_version),
                mode: crate::access::AccessMode::Write,
                range: DeclRange::Buffer(range),
                semantic: AccessSemantic::BufferWrite(usage),
                details: AccessDetails::None,
                read_required: false,
                coverage,
                invalidate_before: false,
                discard_after: false,
            });
            (
                BufferVersion::new(resource, output_version),
                BufferWrite(handle),
            )
        }

        /// Declares a buffer read-modify-write and returns its successor version.
        pub fn read_write_buffer(
            &mut self,
            input: BufferVersion,
            usage: BufferReadWriteUse,
            range: BufferRange,
        ) -> (BufferVersion, BufferReadWrite) {
            let output_version = input.version.saturating_add(1);
            let resource = input.resource;
            let handle = self.state.push(AccessDecl {
                handle: 0,
                resource,
                input_version: input.version,
                output_version: Some(output_version),
                mode: crate::access::AccessMode::ReadWrite,
                range: DeclRange::Buffer(range),
                semantic: AccessSemantic::BufferReadWrite(usage),
                details: AccessDetails::None,
                read_required: true,
                coverage: WriteCoverage::Unknown,
                invalidate_before: false,
                discard_after: false,
            });
            (
                BufferVersion::new(resource, output_version),
                BufferReadWrite(handle),
            )
        }
    };
}

/// Setup-only declaration surface for a raster pass.
pub struct RasterPassBuilder {
    pub(crate) state: PassBuildState,
}

impl RasterPassBuilder {
    resource_declaration_methods!();

    /// Declares one color attachment; its load operation determines whether old
    /// contents participate in the pass dependency.
    pub fn color_attachment(
        &mut self,
        input: TextureVersion,
        desc: ColorAttachmentDesc,
    ) -> TextureVersion {
        let output_version = input.version.saturating_add(1);
        let resource = input.resource;
        let (read_required, coverage, invalidate_before) = match desc.operations.load {
            LoadOp::Load => (true, desc.operations.write_coverage, false),
            LoadOp::Clear(_) => (false, WriteCoverage::Full, false),
            LoadOp::DontCare => (false, desc.operations.write_coverage, true),
        };
        self.state.push(AccessDecl {
            handle: 0,
            resource,
            input_version: input.version,
            output_version: Some(output_version),
            mode: if read_required {
                crate::access::AccessMode::ReadWrite
            } else {
                crate::access::AccessMode::Write
            },
            range: DeclRange::Texture(desc.range),
            semantic: AccessSemantic::ColorAttachment { index: desc.index },
            details: AccessDetails::ColorAttachment(desc),
            read_required,
            coverage,
            invalidate_before,
            discard_after: desc.operations.store == StoreOp::Discard,
        });
        TextureVersion::new(resource, output_version)
    }

    /// Declares one depth-stencil attachment; load operations determine which
    /// aspects read previous contents.
    pub fn depth_stencil_attachment(
        &mut self,
        input: TextureVersion,
        desc: DepthStencilAttachmentDesc,
    ) -> TextureVersion {
        let output_version = input.version.saturating_add(1);
        let resource = input.resource;
        let read_required = desc
            .depth
            .map(|ops| matches!(ops.load, LoadOp::Load))
            .unwrap_or(false)
            || desc
                .stencil
                .map(|ops| matches!(ops.load, LoadOp::Load))
                .unwrap_or(false);
        let clear = desc
            .depth
            .map(|ops| matches!(ops.load, LoadOp::Clear(_)))
            .unwrap_or(false)
            || desc
                .stencil
                .map(|ops| matches!(ops.load, LoadOp::Clear(_)))
                .unwrap_or(false);
        let full = desc
            .depth
            .map(|ops| ops.write_coverage == WriteCoverage::Full)
            .unwrap_or(false)
            || desc
                .stencil
                .map(|ops| ops.write_coverage == WriteCoverage::Full)
                .unwrap_or(false);
        let dont_care = desc
            .depth
            .map(|ops| matches!(ops.load, LoadOp::DontCare))
            .unwrap_or(false)
            || desc
                .stencil
                .map(|ops| matches!(ops.load, LoadOp::DontCare))
                .unwrap_or(false);
        let discard = desc
            .depth
            .map(|ops| ops.store == StoreOp::Discard)
            .unwrap_or(false)
            || desc
                .stencil
                .map(|ops| ops.store == StoreOp::Discard)
                .unwrap_or(false);
        self.state.push(AccessDecl {
            handle: 0,
            resource,
            input_version: input.version,
            output_version: Some(output_version),
            mode: if read_required {
                crate::access::AccessMode::ReadWrite
            } else {
                crate::access::AccessMode::Write
            },
            range: DeclRange::Texture(desc.range),
            semantic: AccessSemantic::DepthStencilAttachment {
                depth: desc.depth.is_some(),
                stencil: desc.stencil.is_some(),
            },
            details: AccessDetails::DepthStencilAttachment(desc),
            read_required,
            coverage: if clear || full {
                WriteCoverage::Full
            } else {
                WriteCoverage::Unknown
            },
            invalidate_before: dont_care,
            discard_after: discard,
        });
        TextureVersion::new(resource, output_version)
    }
}

/// Setup-only declaration surface for a compute pass.
pub struct ComputePassBuilder {
    pub(crate) state: PassBuildState,
}

impl ComputePassBuilder {
    resource_declaration_methods!();
}

/// Setup-only declaration surface for a copy pass.
pub struct CopyPassBuilder {
    pub(crate) state: PassBuildState,
}

impl CopyPassBuilder {
    /// Declares a texture copy source.
    pub fn read_texture(&mut self, input: &TextureVersion, range: TextureRange) -> TextureRead {
        let handle = self.state.push(AccessDecl {
            handle: 0,
            resource: input.resource,
            input_version: input.version,
            output_version: None,
            mode: crate::access::AccessMode::Read,
            range: DeclRange::Texture(range),
            semantic: AccessSemantic::TextureRead(TextureReadUse::CopySource),
            details: AccessDetails::None,
            read_required: true,
            coverage: WriteCoverage::Unknown,
            invalidate_before: false,
            discard_after: false,
        });
        TextureRead(handle)
    }

    /// Declares a texture copy destination and returns its successor version.
    pub fn write_texture(
        &mut self,
        input: TextureVersion,
        range: TextureRange,
        coverage: WriteCoverage,
    ) -> (TextureVersion, TextureWrite) {
        let output_version = input.version.saturating_add(1);
        let resource = input.resource;
        let handle = self.state.push(AccessDecl {
            handle: 0,
            resource,
            input_version: input.version,
            output_version: Some(output_version),
            mode: crate::access::AccessMode::Write,
            range: DeclRange::Texture(range),
            semantic: AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination),
            details: AccessDetails::None,
            read_required: false,
            coverage,
            invalidate_before: false,
            discard_after: false,
        });
        (
            TextureVersion::new(resource, output_version),
            TextureWrite(handle),
        )
    }

    /// Declares a buffer copy source.
    pub fn read_buffer(&mut self, input: &BufferVersion, range: BufferRange) -> BufferRead {
        let handle = self.state.push(AccessDecl {
            handle: 0,
            resource: input.resource,
            input_version: input.version,
            output_version: None,
            mode: crate::access::AccessMode::Read,
            range: DeclRange::Buffer(range),
            semantic: AccessSemantic::BufferRead(BufferReadUse::CopySource),
            details: AccessDetails::None,
            read_required: true,
            coverage: WriteCoverage::Unknown,
            invalidate_before: false,
            discard_after: false,
        });
        BufferRead(handle)
    }

    /// Declares a buffer copy destination and returns its successor version.
    pub fn write_buffer(
        &mut self,
        input: BufferVersion,
        range: BufferRange,
        coverage: WriteCoverage,
    ) -> (BufferVersion, BufferWrite) {
        let output_version = input.version.saturating_add(1);
        let resource = input.resource;
        let handle = self.state.push(AccessDecl {
            handle: 0,
            resource,
            input_version: input.version,
            output_version: Some(output_version),
            mode: crate::access::AccessMode::Write,
            range: DeclRange::Buffer(range),
            semantic: AccessSemantic::BufferWrite(BufferWriteUse::CopyDestination),
            details: AccessDetails::None,
            read_required: false,
            coverage,
            invalidate_before: false,
            discard_after: false,
        });
        (
            BufferVersion::new(resource, output_version),
            BufferWrite(handle),
        )
    }
}

macro_rules! builder_state_methods {
    ($builder:ident) => {
        impl $builder {
            pub(crate) fn new(state: PassBuildState) -> Self {
                Self { state }
            }
            pub(crate) fn into_state(self) -> PassBuildState {
                self.state
            }
        }
    };
}

builder_state_methods!(RasterPassBuilder);
builder_state_methods!(ComputePassBuilder);
builder_state_methods!(CopyPassBuilder);

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
