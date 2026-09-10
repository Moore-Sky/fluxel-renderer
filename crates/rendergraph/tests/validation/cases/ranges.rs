//! Descriptor, initialization, and range validation cases.

use super::*;

#[test]
fn initialization_tracks_full_unknown_imported_and_partial_contents() {
    let caps = capabilities(true, true, true, true);

    let mut full = RenderGraph::<()>::new();
    let buffer = full.create_buffer("full", BufferDesc { size: 16 });
    let written = full.add_compute_pass(
        "initialize",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    full.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert!(full.compile(&caps).is_ok());

    let mut unknown = RenderGraph::<()>::new();
    let buffer = unknown.create_buffer("unknown", BufferDesc { size: 16 });
    let written = unknown.add_compute_pass(
        "unknown-write",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Unknown,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    unknown.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        unknown.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );

    let mut defined = RenderGraph::<()>::new();
    let imported = defined.import_buffer_slot("defined", import_buffer(InitialContents::Defined));
    let reader = defined.add_compute_pass(
        "read-import",
        |pass| {
            let _ = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    defined.mark_side_effect(reader.id, SideEffectReason::Diagnostic("keep read".into()));
    assert!(defined.compile(&caps).is_ok());

    let mut undefined = RenderGraph::<()>::new();
    let imported =
        undefined.import_buffer_slot("undefined", import_buffer(InitialContents::Undefined));
    let reader = undefined.add_compute_pass(
        "read-import",
        |pass| {
            let _ = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    undefined.mark_side_effect(reader.id, SideEffectReason::Diagnostic("keep read".into()));
    assert_error(
        undefined.compile(&caps),
        CompileErrorKind::ReadBeforeInitialization,
    );

    let mut partial = RenderGraph::<()>::new();
    let buffer = partial.create_buffer("partial", BufferDesc { size: 16 });
    let written = partial.add_compute_pass(
        "partially-initialize",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::Bytes { offset: 0, size: 8 },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    partial.export_buffer(
        written.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        partial.compile(&caps),
        CompileErrorKind::InvalidExportOrPresent,
    );
}

#[test]
fn invalid_buffer_and_texture_ranges_are_rejected() {
    let caps = capabilities(true, true, true, true);

    let mut buffers = RenderGraph::<()>::new();
    let buffer = buffers.create_buffer("buffer", BufferDesc { size: 16 });
    let pass = buffers.add_compute_pass(
        "bad-buffer-range",
        |pass| {
            let (next, _) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::Bytes {
                    offset: 12,
                    size: 8,
                },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    buffers.export_buffer(
        pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        buffers.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );

    let mut textures = RenderGraph::<()>::new();
    let image = textures.create_texture("image", texture());
    let pass = textures.add_compute_pass(
        "bad-texture-range",
        |pass| {
            let (next, _) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::Subresources {
                    base_mip_level: 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: TextureAspect::Color,
                },
                WriteCoverage::Full,
            );
            (next, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    textures.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    assert_error(
        textures.compile(&caps),
        CompileErrorKind::InvalidSubresourceRange,
    );
}
