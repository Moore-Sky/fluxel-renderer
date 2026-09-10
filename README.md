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
cover closed textured and fixed-Lambert recipes.
Each legacy-unlit packet draw may carry an independent affine model-to-world
placement; the renderer lowers it into the existing camera/material uniform
contract without exposing a general scene or pipeline API.
The default build remains the portable headless domain model.

The fixed renderer and RHI slices are split into single-responsibility modules.
Six proven raster paths share one private, closed recipe mapping.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](documents/design-rhi.md)
- [Renderer design](documents/design-renderer.md)
- [Architecture decisions](documents/adr/README.md)
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

Near-term work is intentionally more specific than distant direction. Items
beyond the current work are goals, not version or schedule commitments.

### Current — Renderer foundation

- [x] DX12/Vulkan owned resources and RenderGraph Copy/Compute/Raster
- [x] Immutable mesh, texture, UV, and normal GPU snapshots
- [x] Indexed draw with Camera and BasicMaterial uniforms
- [x] Linear-clamp sampling and sRGB decode-before-filter
- [x] Fixed Lambert lighting
- [x] Consolidate fixed slices into private closed recipes
- [x] Lower `DrawList` into renderer-owned packets
- [x] Add per-draw transform and object state for legacy-unlit packets
- [ ] Introduce minimal shader, material, and layout variation
- [ ] Add basic PBR and one representative static scene

### Next — Static renderer

- [ ] Load and render a static GLTF scene
- [ ] Compile and reflect shader variants
- [ ] Add stable resource handles, caching, and reuse
- [ ] Add culling, batching, and instancing
- [ ] Complete Surface/Present, resize, and lost-surface recovery on Windows

### Mid-term — Web and dynamic scenes

- [ ] Establish WebGPU/WebGL2-compatible renderer paths
- [ ] Support animation, skinning, morph targets, and richer scene submission

### Long-term — Platform runtime

- [ ] Bring up Metal, Android, and iOS under one renderer/runtime model
- [ ] Keep Windows, Web, Android, and iOS semantics aligned by conformance tests

### Performance evolution

Multi-queue scheduling, parallel command recording, recording caches, transient
resource aliasing, and more aggressive GPU scheduling or memory optimization
enter the roadmap only after representative benchmarks or profiling demonstrate
a concrete need. They are not current feature TODOs.

### Future / TBD

A TypeScript or Three.js-style façade and a declarative UI runtime are possible
long-term directions. They have no current plan, version, schedule, or
compatibility commitment.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
