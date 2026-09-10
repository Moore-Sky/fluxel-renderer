# Fluxel Renderer design

## Purpose

`fluxel-renderer` is the application-facing, headless renderer layer in the
Fluxel workspace. It owns renderer policy: it accepts typed scene-domain data,
coordinates immutable GPU snapshots, selects one proven raster contract, and
declares the corresponding frame graph. It never exposes native handles to its
callers.

The crate provides portable domain objects (`Camera`, `Geometry`, `Mesh`,
`BasicMaterial`, and insertion-ordered `DrawList`); opt-in non-blocking uploads
that publish immutable mesh and texture snapshots after GPU completion; and six
closed indexed-draw contracts which yield opaque offscreen `FrameImage`
metadata after completion. The default crate has no graphics-backend dependency;
GPU coordination and fixed drawing require the `gpu-upload` feature.

This document is a description of the current system, not a release history.
The reasons for durable architectural choices are in the
[ADRs](adr/README.md).

## Layer boundaries

```text
application
    | Camera, Geometry, Mesh, BasicMaterial, upload inputs
    v
fluxel-renderer
    | snapshot lifetime, fixed draw policy, graph declaration
    v
fluxel-rendergraph
    | validate declarations, derive dependencies/transitions, immutable plan
    v
fluxel-rhi
    | resources, native recording/submission/completion, DX12/Vulkan
    v
native GPU API
```

The renderer chooses *what* one fixed frame means: it creates
renderer-owned declarations, imports ready snapshots, binds a closed RHI
artifact, and translates completion into renderer-visible status.

RenderGraph owns portable resource-access semantics. It validates declarations,
derives dependencies and state transitions, culls unused work, and creates an
immutable `ExecutionPlan`. It does not own assets, materials, native objects,
barriers, submission, or readback implementation.

RHI owns the native boundary: allocation, resource leases, native recording,
one-queue submission, completion observation, and all HAL/unsafe code.
Renderer code sees only safe opaque RHI types and the fixed
`RasterBackend`/`RasterKernel` contract.

Persistent asset identity, loading, cache eviction, and hot reload are outside
the graph. An asset system may resolve a ready GPU generation before frame
construction, but it is not a graph node or graph resource owner. See
[ADR-0001](adr/0001-assets-outside-rendergraph.md).

## Non-goals and current limits

The current renderer intentionally has no:

- `DrawList` lowering, frame packets, stable asset/resource handles, or cache;
- arbitrary shaders, reflection, pipeline layouts, bindings, vertex layouts,
  samplers, views, or material parameters;
- per-draw transforms, lights, depth, blending, culling, batching, instancing,
  PBR, animation, or scene-file loading;
- Surface acquisition, presentation, resize, or lost-surface recovery; or
- multi-queue scheduling, parallel recording, aliasing, or performance policy.

The fixed API is a correctness vertical slice, not an early general pipeline
API. See [ADR-0006](adr/0006-no-general-pipeline-yet.md) and
[ADR-0007](adr/0007-closed-fixed-renderer-recipes.md).

## Module map

```text
crates/renderer/src/
  lib.rs                 public domain model and feature-gated facade
  shader/                private stage-selection boundary
  frame_uniform.rs       fixed camera/material uniform ABI
  upload/
    indexed.rs           position/index snapshot upload and publication
    textured.rs          position/index/UV snapshot upload and publication
    normal.rs            position/index/normal snapshot upload and publication
    texture/             separate linear-UNORM and sRGB texture domains
    shared.rs            generation allocation and snapshot-use gate
  fixed_frame/
    recipe.rs            six closed renderer-to-RHI raster mappings
    renderer.rs          draw-start validation and executor setup
    graph.rs             graph declarations and resource-provider bindings
    submission.rs        two-phase submission/completion lifecycle
    ids.rs               private fixed pipeline and binding identifiers
    fixtures/            hardware CPU-oracle conformance fixtures
```

Each module has one responsibility. The `mod.rs` files are facades that retain
public paths rather than accumulating implementation logic. `shader` is a
private boundary, not a public shader system: it keeps stage selection out of
the public domain API until a supported renderer shader contract exists.

## Domain model and public API boundary

`Camera` stores column-major view and projection matrices. `BasicMaterial`
stores a linear RGBA base color. `Geometry` owns CPU `f32x3` positions and may
own validated `u32` indices; `Mesh` pairs one geometry with one basic material.
Fields are private, so constructors and accessors, not struct literals, define
the supported contract.

`DrawList<'a>` borrows a camera and meshes for one preparation scope and
preserves insertion order. It is currently typed submission input only: it does
not create a graph, allocate GPU memory, sort, cull, or draw. This avoids
presenting a borrowed transition model as a stable handle-based renderer ABI.

With `gpu-upload`, the crate also exposes opaque-ready resource families:

| Input domain | Upload operation | Ready snapshot | Fixed stream ABI |
| --- | --- | --- | --- |
| indexed geometry | `IndexedMeshUpload` | `IndexedMeshSnapshot` | position `f32x3`, index `u32` |
| indexed geometry plus UV | `TexturedIndexedMeshUpload` | `TexturedIndexedMeshSnapshot` | position `f32x3`, index `u32`, UV `f32x2` |
| indexed geometry plus normals | `NormalIndexedMeshUpload` | `NormalIndexedMeshSnapshot` | position `f32x3`, index `u32`, normal `f32x3` |
| linear RGBA8 image | `BaseColorTextureUpload` | `BaseColorTextureSnapshot` | `Rgba8Unorm` |
| sRGB RGBA8 image | `SrgbBaseColorTextureUpload` | `SrgbBaseColorTextureSnapshot` | `Rgba8UnormSrgb` |

`TexturedBasicMaterial` accepts only a linear texture snapshot, while
`SrgbTexturedBasicMaterial` accepts only an sRGB snapshot. That type split
prevents a draw from silently changing a texture's color-space meaning. Native
buffers, textures, leases, imported graph handles, and RHI bindings remain
crate-private.

Start and terminal errors are structured enums. `DrawStartError` and upload
start errors report rejection before work is accepted; `FixedFrameFailure` and
upload failure enums retain completion, observation, and later-start causes
rather than flattening them into strings.

## Immutable uploads and snapshot ownership

An upload begins native immutable uploads without a host wait. `poll()` observes
each accepted upload non-blockingly and returns `Pending`, `Ready`, or
structured `Failed`. A ready snapshot is cloneable and contains an opaque
generation, immutable RHI result(s), CPU metadata used by fixed validation, and
a shared use gate.

Publication is atomic at the renderer level:

- indexed mesh: both position and index uploads complete;
- textured mesh: position, index, and UV uploads complete;
- normal mesh: position, index, and normal uploads complete; and
- texture: its upload completes.

If a later upload cannot start after an earlier one is accepted, the operation
fails but continues retaining and observing accepted siblings. A partial
generation is never published and accepted RHI work is never dropped early.
Role-specific stream slots remain typed because position, index, UV, and normal
failures have distinct meanings.

Every snapshot generation has one `SnapshotUseGate`, shared by clones:

```text
Ready --reserve--> InFlight --proven complete--> Ready
                         \--uncertain/drop-----> Poisoned
```

Only one fixed draw can use a generation at once. A graph/native rejection
before acceptance releases its reservation; proven completion also releases it.
An accepted submission whose final outgoing state cannot be proven, or an
accepted submission dropped before proof, permanently poisons that generation.
This fail-closed rule prevents later work from assuming a native resource state
which may be unknown; see [ADR-0004](adr/0004-accepted-unknown-quarantine.md).

The generation counter is process-local opaque identity, not an asset handle or
persistence protocol. This crate does not implement hot reload, eviction, or
cross-frame asset selection.

## Fixed raster contracts

`FixedFrameRenderer` owns a clone of a safe `Device`, a capability snapshot,
and a `FrameExecutor<RasterBackend>`. It compiles graphs against capabilities
from that same executor/device. A draw start checks extent, device identity,
index count, finite camera/material data, and its recipe preconditions before
work is accepted.

Private `RasterRecipe` is the sole mapping point for six audited contracts. It
has private fields and exactly six associated constants. A recipe atomically
selects the RHI `RasterKernel`, fixed pipeline/binding IDs, vertex ABI, texture
interpretation, graph shape, snapshot domain, reservation topology, and export
topology. Callers cannot manufacture a new combination by mixing optional
resources, shaders, layouts, or bindings.

| Public start method | Vertex input | Texture operation | Shading result |
| --- | --- | --- | --- |
| `draw` | position, index | none | uniform base color |
| `draw_textured` | position, index | position-derived clamped mip-zero integer `textureLoad` | texture × base color |
| `draw_textured_uv` | position, index, UV | explicit-UV clamped mip-zero integer `textureLoad` | texture × base color |
| `draw_textured_uv_linear_clamp` | position, index, UV | linear UNORM, linear min/mag, nearest mip, clamp-to-edge, level zero | texture × base color |
| `draw_textured_uv_linear_clamp_srgb` | position, index, UV | sRGB view with the same fixed sampler policy | decoded/filter result × base color |
| `draw_lambert` | position, index, normal | none | base color RGB × fixed `+Z` Lambert factor |

All paths are indexed triangle draws. The fixed renderer does not expose a
sampler, LOD, wrap mode, filter choice, vertex-slot selection, shader source,
or pipeline-state selector. Provider/binding maps are private graph details and
are checked against their selected recipe.

## Frame declaration and execution flow

```text
ready snapshot(s) + Camera + material + extent
  -> validate and reserve every required generation
  -> serialize fixed uniform and build a graph declaration
  -> compile immutable ExecutionPlan against executor capabilities
  -> submit uniform upload (phase 1, non-blocking)
  -> when it completes, execute raster graph (phase 2, non-blocking)
  -> poll completion
  -> release reservations + FrameImage, or retain terminal failure
```

The graph imports immutable streams with their reported `CopyDestination`
outgoing state, declares vertex/index reads, imports a required texture as a
sampled resource, creates a linear offscreen `Rgba8Unorm` target, and exports
that target for copy-source readback. Imported snapshots are exported back in
their declared outgoing state. RenderGraph reports portable semantics; RHI
records transitions and fixed native commands from that plan.

Uniform upload and raster execution are separate accepted operations.
`FixedFrameSubmission::poll` does not block the CPU and starts raster only
after uniform completion succeeds. `FrameImage` appears only after raster
completion and reveals metadata, not a host mapping or native texture.
CPU-oracle readback is RHI test support, never a production renderer API.

Queue synchronization, encoders, barriers, leases, fences, and validation
capture remain RHI work. See [ADR-0002](adr/0002-rhi-unsafe-containment.md) and
[ADR-0003](adr/0003-serial-execution-lowering.md).

## Data semantics

### Uniform ABI

`FrameUniform` is exactly 80 bytes: `0..64` is the column-major
`projection * view` matrix; `64..80` is linear RGBA base color. Camera input,
the computed matrix, and each color component must be finite; color is in
`[0, 1]`. The payload is visible to the fixed vertex and fragment stages and is
not a general uniform-layout API.

### Geometry, UVs, and clipping

Positions are tightly packed `f32x3`; indices are tightly packed `u32`.
Textured geometry adds one finite `f32x2` UV per position. Normal geometry adds
one finite normal per position, canonicalized to unit `f32x3` at construction.
The textured and Lambert paths validate their fixed clip-space input before
acceptance, including non-finite values, invalid homogeneous `w`, and clipping
outside the closed contract. UV paths use perspective-correct center
interpolation.

### Color spaces and normals

`Rgba8Image` represents linear UNORM texels; `Srgba8Image` represents encoded
sRGB texels. Both upload original RGBA8 bytes, but the sRGB path chooses an sRGB
native view: RGB is decoded before linear filtering, alpha stays linear, and
the target stays linear `Rgba8Unorm`. UNORM and sRGB filterability are queried
independently from actual device capabilities.

`draw_lambert` perspective-interpolates canonical object-space normals, safely
normalizes a nonzero interpolation, applies a fixed object-space `+Z` Lambert
factor to RGB, and preserves alpha. It has no transforms, normal matrix,
caller-visible lights, or PBR semantics.

## Errors, lifetime, and concurrency

The API separates failures by ownership transition:

- invalid input, foreign device, unavailable capability, graph construction, or
  reservation conflict fail before raster acceptance;
- uniform failure means raster did not start and releases reservations;
- raster rejection before native acceptance releases reservations; and
- accepted work with failed/unobservable completion is terminal and poisons all
  reserved generations.

`FixedFrameSubmission` retains the snapshots, reservations, uniform operation,
graph, and RHI submission required by its state. Snapshot clones retain RHI
leases; device clones in the renderer/executor path keep the native device
alive until dependent work completes.

The mutex-protected gate safely coordinates snapshot clones, but is deliberately
conservative: it is not a promise of parallel rendering. Native queue
serialization and accepted-unknown handling are RHI concerns. Renderer code
never repairs a poisoned snapshot by guessing state.

## Validation and evidence

Portable unit tests cover domain validation, payload packing, publication,
reservations, recipe mapping, graph declaration, and failure paths. They do
not prove DX12/Vulkan correctness.

Windows hardware fixtures run the fixed contracts on both supported native
backends with required validation and compare readback to CPU oracles. Cases
distinguish perspective interpolation, integer versus linear sampling, clamp,
sRGB decode-before-filter, normal-stream binding, Lambert shading, and
pre-accept versus accepted-unknown faults. Unexpected validation diagnostics
are failures. RHI readback consumes the graph-exported outgoing state as its
actual incoming state, so it cannot silently repair a wrong export.

See [ADR-0005](adr/0005-gpu-conformance-evidence.md) and
[ADR-0008](adr/0008-native-platform-test-gates.md). Native conformance is
separate from compile/link and CPU-only graph evidence.

## Extension boundaries

New renderer work needs a narrow tested contract. A scene-submission path must
resolve scene data to renderer-owned packets before graph declaration; it must
not move asset lifetime or native commands into RenderGraph. A general material,
shader, or pipeline API must intentionally replace/extend the closed recipe
boundary with explicit layout, binding, capability, lifetime, and cross-backend
semantics.

Surface/present should sit at a renderer/RHI boundary above swapchain details,
while RenderGraph remains per-frame access planning. Scheduling, parallel
recording, transient aliasing, and caches are lowering optimizations only when
measurement proves they are needed; they do not alter declared frame semantics.

Until those contracts exist, the six-recipe system is the supported renderer
implementation and remains deliberately closed.
