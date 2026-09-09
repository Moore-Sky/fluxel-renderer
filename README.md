# Fluxel Renderer

`fluxel-renderer` is a `0.2.5` workspace milestone for Fluxel's typed render graph,
native RHI boundary, and renderer layer. It starts an independent release line
from a pre-migration source snapshot and does not inherit prior version numbers
or Git history.

## Workspace

The workspace is organised around three crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | Typed resource declarations, dependency compilation, validation, immutable execution plans, and the CPU-only `TestRhi` protocol. |
| `fluxel-rhi` | Headless DX12/Vulkan ownership plus fixed Raster, Compute, and Copy RenderGraph execution. |
| `fluxel-renderer` | Scene/domain data, immutable mesh/texture GPU snapshots, and fixed unlit headless indexed draws. |

The renderer still does not lower a `DrawList` or general material pipeline.
Its optional `gpu-upload` slice publishes a mesh generation after both native
uploads complete and can submit one closed `f32x3/u32` indexed draw to an
offscreen image, optionally modulated by one immutable RGBA8 base-color texture
through closed integer-load or fixed linear-clamp recipes. The default build
remains the portable headless domain model.

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
headless DX12 and Vulkan fixed Raster, Compute, and Copy execution on Windows
when the native loader and driver are available. CI is a compile/link gate;
real-GPU conformance remains an explicit local release gate.

## Roadmap

The frozen `0.1.4` implementation boundary adds a deliberately fixed
Raster→Compute→Copy slice to graph planning, CPU protocol validation, owned
DX12/Vulkan resources, Copy, and fixed Compute. `RasterBackend` executes the
same compiled plan on both APIs for R01 clear/triangle, R02 indexed
viewport/scissor, and X01 raster pixels packed by a closed compute recipe then
copied out. Texture and buffer readback compare exact bytes with CPU oracles;
they consume each export's reported outgoing state rather than repair it.
Raster pipeline, binding, attachment, encoder, and submission leases remain
alive through terminal completion, with structured failures and private unsafe
HAL boundaries. The 0.1.4 release fixtures pass on an AMD Radeon 780M through
both DX12 and Vulkan with Required validation and exact CPU oracles.

The `0.2.0` renderer slice adds exact little-endian `f32x3`/`u32` serialization,
non-blocking immutable buffer upload, and an opaque indexed-mesh snapshot.
Pending or failed submissions never publish a ready generation; native target
and staging storage remain retained through terminal completion. The uploaded
state is reported as `CopyDestination` for a later graph import rather than
guessed as a draw state. This is not draw lowering, texture/PBR, surface/present,
or a general shader/pipeline API. The `0.2.1` slice consumes that ready snapshot
through a renderer-private RenderGraph import, executes a fixed opaque-color
indexed draw into offscreen `Rgba8Unorm`, and restores both immutable buffers to
their reported `CopyDestination` state after proven completion. The `0.2.2`
fixed draw consumes a `Camera` view / projection and
`BasicMaterial::base_color` through one immutable 80-byte uniform: upload
completes non-blockingly before the owning operation submits Raster. It remains
a single closed offscreen recipe, not `DrawList` lowering, PBR, textures, or a
general pipeline/bind-group API. Snapshot clones share a single-in-flight gate;
an unproven accepted Raster outcome poisons the generation instead of guessing
its state. The `0.2.3` slice adds tight immutable `Rgba8Unorm` upload, an opaque
texture snapshot, and `draw_textured`. The fixed shader derives planar
coordinates from model-space position and performs an integer mip-zero
`textureLoad`; it adds no sampler, UV vertex layout, mip/LOD, sRGB, or PBR API.
Mesh and texture generations are reserved and released or poisoned together.
The `0.2.4` slice adds a separate typed geometry/upload/snapshot path with one
finite `f32x2` texture coordinate per vertex. `draw_textured_uv` feeds those
coordinates through a closed second vertex slot while retaining the same
perspective-center, integer mip-zero `textureLoad` semantics. The earlier
position-derived path remains available and unchanged.

The `0.2.5` slice adds one separate explicit-UV path with a private sampler fixed
to linear min/mag, nearest mip, clamp-to-edge on every axis, and
`textureSampleLevel(..., 0.0)`. Real `Rgba8Unorm` filterability participates in
graph compilation and is rechecked before native creation. This does not expose
a configurable sampler, mip/LOD, sRGB, or general material API.

| Component | Next milestones |
| --- | --- |
| RHI / RenderGraph integration `0.1` | Complete: Windows DX12/Vulkan owned resources and fixed Copy → Compute → Raster→Compute→Copy exact-oracle execution |
| Renderer / Shader / Assets `0.1` | Headless Camera/Mesh/Geometry/BasicMaterial/DrawList → minimal shader compile/reflection and resource lifetime/cache/handles |
| Renderer `0.2` | In progress: GPU-ready mesh/texture/UV snapshots and fixed textured unlit draws → fixed private linear-clamp sampling → basic PBR → one real static scene |
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
