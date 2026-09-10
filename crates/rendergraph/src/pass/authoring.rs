//! Pass declaration builders and attachment metadata.

use crate::{
    access::{
        BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureRange,
        TextureReadUse, TextureReadWriteUse, TextureWriteUse, WriteCoverage,
    },
    handles::{
        BufferRead, BufferReadWrite, BufferVersion, BufferWrite, TextureRead, TextureReadWrite,
        TextureVersion, TextureWrite,
    },
    internal::{AccessDecl, AccessDetails, AccessSemantic, DeclRange, PassBuildState},
};

use super::{ColorAttachmentDesc, DepthStencilAttachmentDesc, LoadOp, StoreOp};

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
