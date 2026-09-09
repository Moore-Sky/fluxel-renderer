# Workspace integration examples

This directory is reserved for examples that exercise the assembled renderer
workspace rather than one package in isolation.

Package-specific examples remain beside their crates:

- `crates/rendergraph/examples/` covers graph declaration, compilation, and
  the CPU-only `TestRhi` execution protocol;
- `crates/rhi/examples/` covers native device discovery and opening.

The first root integration example will accompany the headless renderer
vertical slice. An empty placeholder executable is intentionally not presented
as rendering evidence.
