# Fluxel Renderer

`fluxel-renderer` is a `0.1.3` workspace for Fluxel's typed render graph,
native RHI boundary, and renderer layer. It starts an independent release line
from a pre-migration source snapshot and does not inherit prior version numbers
or Git history.

## Workspace

The workspace is organised around three crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | Typed resource declarations, dependency compilation, validation, immutable execution plans, and the CPU-only `TestRhi` protocol. |
| `fluxel-rhi` | Headless DX12/Vulkan ownership plus Copy and fixed-artifact Compute RenderGraph execution and hardware conformance. |
| `fluxel-renderer` | The renderer-facing layer: scene/domain data, draw-list construction, and its internal shader boundary. |

All three packages are present at the current baseline. The renderer crate
currently stops at its headless domain model; it does not yet lower a
`DrawList` into render-graph or native GPU work.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

## Documentation

- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](documents/design-rhi.md)
- [Renderer design](documents/design-renderer.md)
- [RenderGraph guide](crates/rendergraph/README.md)
- [RHI guide](crates/rhi/README.md)

## Quick verification

Rust MSRV is 1.87 (edition 2024).

```sh
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The graph compiler and `TestRhi` are CPU-only. `fluxel-rhi` provides
headless DX12 and Vulkan Copy and fixed Compute execution on Windows when the
native loader and driver are available. CI is a compile/link gate; real-GPU
C01/C02/C03 and K01/K02 conformance remain explicit local release gates.

## Roadmap

Through `0.1.3`, the workspace has graph planning, CPU protocol validation,
DX12/Vulkan owned resources, Copy correctness, and a deliberately fixed
Compute correctness slice. The RHI parses embedded WGSL into validated Naga IR
and lowers the same artifact to DX12 and Vulkan; K01 and K02 compare readback
with wrapping-integer CPU oracles. It does **not** yet draw raster work, lower
renderer scenes, present a surface, or expose a general shader API.

| Component | Next milestones |
| --- | --- |
| RHI / RenderGraph integration `0.1` | RHI implements Windows DX12/Vulkan resources and commands; both backends execute the same RenderGraph plan through Copy/readback → fixed Compute/readback → Raster/readback conformance |
| Renderer / Shader / Assets `0.1` | Headless Camera/Mesh/Geometry/BasicMaterial/DrawList → minimal shader compile/reflection and resource lifetime/cache/handles |
| Renderer `0.2` | Texture, indexed mesh, basic PBR, and one real static scene |
| RenderGraph `0.2` | WebGPU and WebGL2 compatibility adapters |
| Renderer `0.3` / Assets `0.2` / Shader `0.2` | GLTF, loading/reuse, variants/layout metadata, caches, culling, batching |
| RenderGraph `0.3` / Renderer `0.4` | Surface/Present/resize/lost, then a visible Windows/Web renderer |
| Renderer `0.5` | Animation, skinning, morph, instancing |
| RenderGraph `0.4` / Renderer `0.6` | Metal conformance, then unified Windows/Web/Android/iOS runtime |
| JS/WASM `0.1` | Three-like TypeScript façade with handle/batch synchronization |
| UI `0.1` | Declarative Vue/mini-program-style API plus Rust layout/text/image/event runtime |

Correctness gates are readback plus CPU oracles on both native backends.
Multi-queue, parallel recording, recording caches, and transient aliasing wait
for profiling evidence. The published design documents above describe the
current supported boundary; local draft materials are not release documentation.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
