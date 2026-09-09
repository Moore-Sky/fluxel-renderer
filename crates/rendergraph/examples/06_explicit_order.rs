//! Compile-only example; run with `cargo run --example 06_explicit_order`.
//! Execute callbacks are retained but are not invoked by `compile`.

mod common;

use fluxel_rendergraph::*;

fn main() {
    let mut graph = RenderGraph::new();
    let begin = graph.add_copy_pass("publish-begin", |_| ((), ()), |_, _, _, _| Ok(()));
    let end = graph.add_copy_pass("publish-end", |_| ((), ()), |_, _, _, _| Ok(()));

    graph.mark_side_effect(
        begin.id,
        SideEffectReason::ExternalProtocol("begin external publication".into()),
    );
    graph.mark_side_effect(
        end.id,
        SideEffectReason::ExternalProtocol("finish external publication".into()),
    );
    graph.depends_on(
        begin.id,
        end.id,
        ExplicitOrderReason::ExternalProtocol("external protocol requires begin before end".into()),
    );

    let _result: CompileResult = graph.compile(&common::single_queue_capabilities());
}
