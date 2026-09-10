# Fluxel Renderer

`fluxel-renderer` is a Rust workspace for Fluxel's typed render graph, native
RHI boundary, and renderer layer.

## Workspace

The workspace is organised around three crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | Typed resource declarations, dependency compilation, validation, immutable execution plans, and the CPU-only `TestRhi` protocol. |
| `fluxel-rhi` | Headless DX12/Vulkan ownership plus fixed Raster, Compute, and Copy RenderGraph execution. |
| `fluxel-renderer` | Scene/domain data, immutable GPU snapshots, owned render packets, and fixed headless indexed draws. |

The optional `gpu-upload` slice publishes immutable GPU snapshot generations
after native uploads complete. It can lower an insertion-ordered `DrawList`
and its matching ready indexed snapshots into an owned, opaque, device-affine
`RenderPacket`, then execute all of its legacy unlit indexed draws through one
compiled graph, raster pass, and submission. Existing single-draw paths also
cover closed textured, vertex-color, and fixed-Lambert recipes.
Each legacy-unlit packet draw may carry an independent affine model-to-world
placement; the renderer lowers it into the existing camera/material uniform
contract without exposing a general scene or pipeline API.
The default build remains the portable headless domain model.

The fixed renderer and RHI slices are split into single-responsibility modules.
Seven proven raster paths share one private, closed recipe mapping.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](documents/design-rhi.md)
- [Renderer design](documents/design-renderer.md)
- [Architecture decisions](documents/adr/README.md)
- [Fluxel ecosystem map](documents/fluxel-ecosystem.md)
- [RenderGraph guide](crates/rendergraph/README.md)
- [RHI guide](crates/rhi/README.md)

## Quick verification

Rust MSRV is 1.87 (edition 2024).

```sh
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The graph compiler and `TestRhi` are CPU-only. `fluxel-rhi` provides
headless DX12 and Vulkan fixed Raster, Compute, and Copy execution on Windows
when the native loader and driver are available. CI is a compile/link gate;
real-GPU conformance remains an explicit local release gate.

## Roadmap

This roadmap covers this workspace only. Near-term work is deliberately more
specific than distant direction; no item is a version or schedule commitment.
For ownership outside the renderer workspace, see the
[Fluxel ecosystem map](documents/fluxel-ecosystem.md).

### Current — Renderer submission model

- [x] Establish closed recipes for proven fixed raster combinations.
- [x] Publish immutable GPU snapshots and lower ordered `DrawList` input into
  owned, device-affine render packets.
- [x] Lower one packet through RenderGraph into a compiled plan and one native
  submission.
- [ ] Make render intent, packet construction, recipe selection, and graph
  lowering explicit parts of one renderer submission model.
- [ ] Support deterministic multi-draw ordering and compatible packet grouping.
- [x] Add only the minimal material, shader, and vertex-layout variation that
  packet lowering can describe without a public general-pipeline API.

### Next — Renderer execution

- [ ] Batch compatible render packets while preserving declared ordering.
- [ ] Lower compatible repeated draws as instances where the chosen recipe
  permits it.
- [ ] Reuse compatible renderer-side pipelines and bindings across a frame.
- [ ] Separate frame preparation, graph construction, and submission cleanly.
- [ ] Establish representative renderer-level correctness and performance
  benchmarks.

### Performance evolution

Multi-queue scheduling, parallel command recording, recording caches, transient
resource aliasing, and more aggressive GPU scheduling or memory optimization
enter the roadmap only after representative benchmarks or profiling demonstrate
a concrete need. They are not current feature TODOs.

### Out of scope

Asset identity and caching, file/URL loading, scene graphs, animation,
presentation and platform lifecycle, backend bring-up, and higher-level JS or
UI APIs belong to other Fluxel layers rather than this renderer workspace.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
