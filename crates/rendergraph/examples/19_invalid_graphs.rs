//! Compile-only fixture for invalid declarations and one deferred recording shape.
//! It demonstrates compiler errors; compile never calls execute callbacks.

mod common;

use fluxel_rendergraph::pass::BindingResource;
use fluxel_rendergraph::*;

fn report(result: CompileResult) {
    if let Err(error) = result {
        match error.kind {
            CompileErrorKind::ReadBeforeInitialization
            | CompileErrorKind::StaleOrForeignVersion
            | CompileErrorKind::InvalidSubresourceRange => {}
            _ => {}
        }
    }
}

fn read_before_initialization() {
    let mut graph = RenderGraph::<()>::new();
    let uninitialized = graph.create_texture("never-written", common::rgba8_texture(64, 64));
    let pass = graph.add_raster_pass(
        "invalid-read",
        |pass| {
            let read = pass.read_texture(
                &uninitialized,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), read)
        },
        |commands, resolver, data, _frame| {
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(19),
                &[BindingResource::TextureRead(data)],
                &[],
            )?;
            commands.set_pipeline(RasterPipelineId::new(19))?;
            commands.set_bindings(&bindings)?;
            commands.draw(0..3, 0..1)?;
            Ok(())
        },
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain invalid fixture".into()),
    );
    report(graph.compile(&common::single_queue_capabilities()));
}

fn foreign_version() {
    let mut first = RenderGraph::<()>::new();
    let foreign = first.create_texture("from-first-graph", common::rgba8_texture(64, 64));
    let mut second = RenderGraph::new();
    let invalid = second.add_raster_pass(
        "foreign-version",
        move |pass| {
            let read = pass.read_texture(&foreign, TextureReadUse::Sampled, TextureRange::whole());
            ((), read)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    second.mark_side_effect(
        invalid.id,
        SideEffectReason::Diagnostic("retain foreign-version fixture".into()),
    );
    report(second.compile(&common::single_queue_capabilities()));
}

fn invalid_subresource_range() {
    let mut graph = RenderGraph::new();
    let texture = graph.import_texture_slot(
        "one-mip",
        common::imported_texture(64, 64, ResourceAccessState::ShaderSampledRead),
    );
    let invalid = graph.add_raster_pass(
        "out-of-range-mip",
        |pass| {
            let read = pass.read_texture(
                &texture.version,
                TextureReadUse::Sampled,
                TextureRange::Subresources {
                    base_mip_level: 1,
                    mip_level_count: 1,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: TextureAspect::Color,
                },
            );
            ((), read)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        invalid.id,
        SideEffectReason::Diagnostic("retain invalid-range fixture".into()),
    );
    report(graph.compile(&common::single_queue_capabilities()));
}

fn foreign_access_during_recording() {
    let mut graph = RenderGraph::<()>::new();
    let source = graph.import_buffer_slot(
        "vertices",
        common::imported_buffer(1024, ResourceAccessState::VertexRead),
    );
    let producer = graph.add_raster_pass(
        "declares-access",
        |pass| {
            let access =
                pass.read_buffer(&source.version, BufferReadUse::Vertex, BufferRange::whole());
            (access, ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    let foreign_access = producer.output;
    let invalid = graph.add_raster_pass(
        "misuses-foreign-access",
        |_| ((), ()),
        move |commands, _resolver, _data, _frame| {
            // This shape currently type-checks; recording validation must return
            // `ForeignOrUndeclaredPassAccess` rather than letting it reach an RHI.
            commands.set_vertex_buffer(0, &foreign_access)?;
            Ok(())
        },
    );
    graph.mark_side_effect(
        invalid.id,
        SideEffectReason::Diagnostic("retain foreign-access fixture".into()),
    );
}

fn unknown_write_does_not_initialize() {
    let mut graph = RenderGraph::new();
    let buffer = graph.create_buffer("unknown-write", common::buffer(1024));
    let written = graph.add_compute_pass(
        "partial-or-conditional-write",
        |pass| {
            let (buffer, write) = pass.write_buffer(
                buffer,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Unknown,
            );
            (buffer, write)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    let invalid = graph.add_compute_pass(
        "read-after-unknown-write",
        |pass| {
            let read = pass.read_buffer(
                &written.output,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), read)
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        invalid.id,
        SideEffectReason::Diagnostic("retain unknown-coverage fixture".into()),
    );
    report(graph.compile(&common::single_queue_capabilities()));
}

fn main() {
    read_before_initialization();
    foreign_version();
    invalid_subresource_range();
    foreign_access_during_recording();
    unknown_write_does_not_initialize();
}
