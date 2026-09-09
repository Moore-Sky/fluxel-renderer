# Fluxel Renderer design

**Status: 0.1.0 headless domain baseline**

`fluxel-renderer` is the application-facing layer that will translate scene
inputs into render-graph declarations and renderer/RHI-owned objects. The
current release establishes that ownership boundary with `Camera`,
`Geometry`, `Mesh`, `BasicMaterial`, and `DrawList`; it does not yet
perform the translation or execute GPU work.

## Responsibility split

```text
application / scene
        |
        v
fluxel-renderer       Camera, mesh/material inputs, draw-list policy
        |
        +--> fluxel-rendergraph   per-frame access and execution planning
        |
        +--> fluxel-rhi           native device/resources/commands (future)

fluxel-assets         persistent identity, loading, cache, lifetime
```

The renderer owns scene-to-frame policy: which objects are visible, how they
are ordered, which material/pipeline recipe they need, and which graph passes
are declared. RenderGraph owns the dependency facts after declaration. RHI owns
native objects and unsafe API calls. Assets remains a sibling project because
persistent identity and loading policy must not become frame-graph state.

## Why the first model is headless

Native Copy, Compute, and Raster conformance are still scheduled work. A
windowed façade built before those primitives would either hide unimplemented
behavior or force scene policy into the RHI. The 0.1.0 renderer therefore
models only inputs that can be tested without pretending to render pixels.

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

## Evolution gates

The next renderer changes follow evidence from the native vertical slices:

1. native resource creation establishes upload and allocation facts;
2. Copy/readback establishes geometry upload and completion ownership;
3. Compute/readback establishes shader compilation/reflection requirements;
4. Raster/readback establishes material, pipeline, vertex-layout, and draw
   lowering;
5. Surface/Present adds a visible frame boundary only after headless
   conformance is stable.

Texture/PBR, GLTF, batching, culling, animation, and platform runtime
unification remain later renderer features. Multi-queue scheduling, recording
caches, and transient aliasing require profiling evidence and are not implied
by this architecture.
