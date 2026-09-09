//! Compile-only diagnostics example; run with `cargo run --example 07_compile_diagnostics`.
//! Execute callbacks are retained but are not invoked by `compile`.
//!
//! This example only proves that callers can type-match stable diagnostic kinds.
//! It demonstrates matching the implemented compiler's stable error kinds.

mod common;

use fluxel_rendergraph::*;

fn inspect(result: CompileResult) {
    match result {
        Ok(output) => {
            let _report: CompileReport = output.report;
        }
        Err(error) => match error.kind {
            CompileErrorKind::StaleOrForeignVersion
            | CompileErrorKind::ReadBeforeInitialization
            | CompileErrorKind::InvalidResourceBranch
            | CompileErrorKind::ConflictingAccess
            | CompileErrorKind::InvalidSubresourceRange
            | CompileErrorKind::DependencyCycle
            | CompileErrorKind::MissingImportContract
            | CompileErrorKind::UnsupportedSemanticRequirement
            | CompileErrorKind::InvalidExportOrPresent => {}
            _ => {}
        },
    }
}

fn main() {
    let graph = RenderGraph::new();
    inspect(graph.compile(&common::single_queue_capabilities()));
}
