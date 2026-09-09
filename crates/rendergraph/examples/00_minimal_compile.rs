//! Smallest complete declaration-and-compilation example.
//!
//! Run with `cargo run --example 00_minimal_compile`.

use fluxel_rendergraph::{
    BufferCapabilities, BufferDesc, BufferRange, BufferWriteUse, CompileError, DeviceCapabilities,
    ExportBufferContract, QueueCapabilities, QueueDescriptor, QueueId, RenderGraph,
    ResourceAccessState, WriteCoverage,
};

fn main() -> Result<(), CompileError> {
    let capabilities = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, false, false),
        ))
        .buffers(BufferCapabilities::new(false, true, false))
        .build();

    let mut graph = RenderGraph::<()>::new();
    let output = graph.create_buffer("output", BufferDesc { size: 256 });

    let output = graph.add_compute_pass(
        "initialize",
        |pass| {
            let (next, write) = pass.write_buffer(
                output,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, write)
        },
        |_commands, _resolver, _write, _frame| Ok(()),
    );

    graph.export_buffer(
        output.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&capabilities)?;
    println!(
        "retained passes: {}",
        compiled.graph.execution_order().len()
    );
    Ok(())
}
