//! Compile-only fixture for mip/layer/aspect-range declarations; run with `cargo run --example 11_subresource`.

mod common;

use fluxel_rendergraph::pass::BindingResource;
use fluxel_rendergraph::*;

struct WriteData {
    target: TextureWrite,
}

struct ReadData {
    source: TextureRead,
}

fn main() {
    let mut graph = RenderGraph::new();
    let mut desc = common::rgba8_texture(1024, 1024);
    desc.mip_levels = 4;
    desc.array_layers = 2;
    let texture = graph.import_texture_slot(
        "mipped-array",
        ImportTextureContract {
            descriptor: desc,
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let first_mip_layer_zero = TextureRange::Subresources {
        base_mip_level: 0,
        mip_level_count: 1,
        base_array_layer: 0,
        array_layer_count: 1,
        aspect: TextureAspect::Color,
    };
    let remaining_mips_layer_one = TextureRange::Subresources {
        base_mip_level: 1,
        mip_level_count: 3,
        base_array_layer: 1,
        array_layer_count: 1,
        aspect: TextureAspect::Color,
    };

    let written = graph.add_compute_pass(
        "write-one-subresource",
        |pass| {
            let (output, target) = pass.write_texture(
                texture.version,
                TextureWriteUse::Storage,
                first_mip_layer_zero,
                WriteCoverage::Full,
            );
            (output, WriteData { target })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(20))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(20),
                &[BindingResource::TextureWrite(&data.target)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([64, 64, 1])
        },
    );

    // This is intentionally a separate, non-overlapping range. It remains
    // valid because the imported X0 was initialized and untouched ranges of X1
    // inherit their prior contents and validity.
    let reader = graph.add_compute_pass(
        "read-other-subresources",
        |pass| {
            let source = pass.read_texture(
                &written.output,
                TextureReadUse::Sampled,
                remaining_mips_layer_one,
            );
            ((), ReadData { source })
        },
        |commands, resolver, data, _frame| {
            commands.set_pipeline(ComputePipelineId::new(21))?;
            let bindings = resolver.resolve_bindings(
                BindingSetId::new(21),
                &[BindingResource::TextureRead(&data.source)],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    graph.mark_side_effect(
        reader.id,
        SideEffectReason::Diagnostic("subresource validation fixture".into()),
    );

    graph.export_texture(
        written.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
