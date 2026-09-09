# Fluxel Renderer design

**Status: 0.2.4 fixed headless explicit-UV indexed snapshot draw**

`fluxel-renderer` is the application-facing layer that will translate scene
inputs into render-graph declarations and renderer/RHI-owned objects. The
0.1 established that ownership boundary with `Camera`, `Geometry`, `Mesh`,
`BasicMaterial`, and `DrawList`. The opt-in 0.2.0 slice adds immutable
indexed-geometry upload and readiness publication. The 0.2.1 slice consumes one
ready snapshot in a closed headless indexed draw; it still does not lower a
`DrawList` or expose a general renderer pipeline.

## Responsibility split

```text
application / scene
        |
        v
fluxel-renderer       Camera, mesh/material inputs, draw-list policy
        |
        +--> fluxel-rendergraph   per-frame access and execution planning
        |
        +--> fluxel-rhi           immutable upload + native resources/commands

fluxel-assets         persistent identity, loading, cache, lifetime
```

The renderer owns scene-to-frame policy: which objects are visible, how they
are ordered, which material/pipeline recipe they need, and which graph passes
are declared. RenderGraph owns the dependency facts after declaration. RHI owns
native objects and unsafe API calls. Assets remains a sibling project because
persistent identity and loading policy must not become frame-graph state.

## Why the first model is headless

Native Copy, Compute, and Raster conformance was established by the 0.1.x fixed
execution slices before renderer draw work began. A windowed façade before
Surface/Present would still hide unimplemented behavior or force scene policy
into the RHI, so the current renderer remains headless.

`Geometry` owns CPU vertex positions and optional validated indices.
`Mesh` pairs geometry with a material. `DrawList` borrows meshes and a
camera for one preparation scope and preserves insertion order. Borrowing makes
the initial lifetime rule explicit and avoids inventing an asset handle system
that belongs to `fluxel-assets`.

The initial types keep fields private and expose constructors plus accessors.
That leaves room to add attributes, transforms, material parameters, and
renderer-owned identifiers without committing callers to exhaustive struct
literals.

## Shader boundary

`shader` is private to the renderer. In 0.1.0 it only establishes
stage-specific module selection for `BasicMaterial`; shader compilation,
reflection, variants, pipeline layouts, and caching are intentionally absent.

Keeping this module private prevents compiler-specific source and reflection
types from becoming public renderer API before the DX12/Vulkan pipeline
requirements are known. The first real shader API must be driven by the
Compute/Raster readback slices and expose only facts the renderer needs.

## Future asset and frame coordination

0.2.0 makes the first part concrete. `IndexedMeshUpload` owns two non-blocking
RHI operations for tightly packed position `f32x3` and index `u32` buffers. A
snapshot generation becomes Ready only when both completions succeed. If the
first submission was accepted and the second cannot start, the failed owning
operation still retains and retires the first; a partial generation is never
published. The cloneable snapshot exposes generation and element counts to the
application, while buffers, leases, and their reported `CopyDestination` state
remain private for the next lowering slice.

Four ownership classes stay distinct:

- asset-owned persistent resources carry source identity, cache state, and hot
  reload generations;
- renderer-owned persistent resources carry pipeline/material recipes and
  renderer caches;
- graph transients belong to one compiled frame plan;
- surface-owned images belong to acquisition and presentation.

The renderer resolves a GPU-ready immutable asset snapshot once while building
a frame. Every pass in that frame uses the same physical generation, and the
snapshot's lease survives until the corresponding GPU completion. Hot reload
or eviction publishes a new generation first; the old generation retires only
after its last lease and GPU use complete. If an asset is not ready, renderer
policy chooses an older generation, fallback, or skip instead of blocking the
frame-building thread.

A resource used in the same frame as its upload must eventually enter through
an explicit graph Copy pass or an external GPU dependency. CPU waiting between
passes is not an ownership mechanism.

The frame coordinator also owns persistent state reconciliation. It queries the
known incoming state when binding an import and commits the exported state only
after accepted submission. A failed submit must not publish a guessed state.
Concurrent frames need queue ordering or an explicit GPU dependency; one global
`last_state` value cannot safely represent overlapping use.

## Fixed headless frame

0.2.3 keeps one renderer-owned coordinator rather than general draw-list
lowering. `FixedFrameRenderer::draw` accepts a ready `IndexedMeshSnapshot`, a
`Camera`, a `BasicMaterial`, and an extent. It serializes the camera and
material at start into one immutable 80-byte uniform: column-major
`projection * view` in bytes 0..64 and linear RGBA base color in bytes 64..80.
Inputs and their matrix product must be finite; every color component is in
`[0, 1]`. The operation owns a non-blocking uniform upload first, then submits
the Raster work only after that upload completes—there is no host wait between
the two phases.

The Raster graph imports tightly packed `f32x3` positions and `u32` indices
using each upload's reported `CopyDestination` state, and declares one
offscreen `Rgba8Unorm` pass. Its closed RHI recipe has exactly one static
group-0/binding-0, 80-byte uniform visible to vertex and fragment stages.
Callers cannot choose arbitrary shaders, layouts, bindings, or pipeline state.

The graph exports both immutable buffers back in `CopyDestination` after the
draw and exports the target in `CopySource`. Clones of one snapshot generation
share a single-in-flight reservation. Uniform-phase failure or Drop and a
pre-Raster-submit rejection release that reservation. Proven Raster completion
permits reuse; only an accepted Raster failure or Drop poisons the generation
because the snapshot's native state is then not known.
Polling never waits for the host or uses queue-idle as an ownership mechanism.
Only opaque image extent/format/ownership becomes public after completion.

The textured variant adds a separately uploaded immutable `Rgba8Unorm` image.
Only proven upload completion publishes an opaque snapshot. Its graph import
uses the reported `CopyDestination` state, declares a whole Sampled read, and
exports the texture back to `CopyDestination`. The shader derives planar UV
from model-space position, uses perspective-center interpolation, and performs
a fixed clamped mip-zero integer `textureLoad`. There is no sampler contract.
The renderer fail-closes non-finite, non-positive-w, or clipped textured input.
Mesh and texture reservations roll back together before acceptance and are
released or poisoned together after it.

0.2.4 keeps that path intact and adds a separate `TexturedGeometry` /
`TexturedIndexedMeshSnapshot` generation containing positions, indices, and one
finite `f32x2` UV per vertex. All three uploads must complete before publication.
`draw_textured_uv` binds positions and UVs in two closed vertex slots and uses
perspective-center interpolation before the same clamped integer mip-zero
`textureLoad`. The typed snapshot and RHI binding retain physical stream roles,
so equally sized generic buffers cannot be silently swapped. This still adds no
sampler, filtering, mip selection, sRGB, normals, lighting, or PBR contract.

This is evidence for one fixed textured snapshot-to-plan-to-native path, not a
claim that model transforms, `DrawList` lowering, samplers, UV attributes,
mips/LOD, sRGB, PBR, depth, blending, batching, general bindings, Surface, or
Present are implemented.

## Evolution gates

The next renderer changes follow evidence from the native vertical slices:

1. native resource creation establishes upload and allocation facts;
2. Copy/readback establishes geometry upload and completion ownership;
3. Compute/readback establishes shader compilation/reflection requirements;
4. Raster/readback establishes material, pipeline, vertex-layout, and draw
   lowering;
5. Surface/Present adds a visible frame boundary only after headless
   conformance is stable.

Sampler/UV/PBR, GLTF, batching, culling, animation, and platform runtime
unification remain later renderer features. Multi-queue scheduling, recording
caches, and transient aliasing require profiling evidence and are not implied
by this architecture.
