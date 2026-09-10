# Fluxel ecosystem map

Fluxel is intended to grow as focused libraries rather than as one
renderer-shaped engine. This document records the intended ownership map and
development direction. It is not a release schedule, compatibility promise, or
commitment to create every named library.

## Architecture

The conceptual integration path is deliberately layered. Work moves toward
native execution; runtime and ABI layers compose and expose that work without
turning internal crates into FFI boundaries.

```text
Platform + Host Capability API
              |
              v
            Loader
              |
              v
            Assets
              |
              v
     Canvas / Renderer policy
              |
              v
          RenderGraph
              |
              v
              RHI
              |
              v
 Runtime composition and lifecycle
              |
              v
 Runtime ABI / JS-facing boundary
              |
              v
native DLL / dylib / so, or WASM imports and exports
```

This is an ownership and integration skeleton, not a literal Rust dependency
chain. `fluxel-runtime` composes the selected services in-process, while
`fluxel-runtime-abi` exposes a small stable external boundary to an embedder.
Internal crates continue to use direct Rust APIs; they never communicate through
a DLL, shared object, WASM boundary, or FFI solely because an external ABI
exists.

## Ownership

### Graphics foundation

The current workspace provides the graphics foundation.

- `fluxel-renderer` prepares frames, selects closed recipes, builds owned
  render packets, and lowers them into RenderGraph declarations.
- `fluxel-rendergraph` owns frame-local access declarations, validation,
  dependencies, transitions, and immutable execution plans.
- `fluxel-rhi` owns device-affine native resources, backend lowering,
  recording, submission, completion, diagnostics, and native lifetime
  containment.

Canvas-oriented drawing policy may share this foundation, but it must not force
2D policy into the renderer. Renderer and Canvas consume ready resource
snapshots; neither owns asset identity, host capabilities, runtime composition,
or JavaScript APIs.

### Resources and loading

`fluxel-assets` owns typed handles, generations, cross-frame lifetime,
loading/ready/failed state, residency, cache/reuse, and safe release. It is the
owner of a durable resource's identity, not of a particular frame's graph
binding or native command.

`fluxel-loader` orchestrates file or URL import into CPU meshes, textures,
shaders, JSON, and formats such as GLTF. It supplies assets without making file
formats, import policy, or decoding part of renderer code. Shader source
processing, compilation, reflection, variants, and cache artifacts remain a
separate shader-tooling concern whose outputs a renderer can consume.

### Platform and host capabilities

Platform owns application startup, windows, event-loop integration, timing and
frame control, foreground/background state, and surface lifecycle. Narrow Rust
capability implementations cover codec/encoding and hashing, filesystem,
networking, image processing, audio processing, and platform integration.

`fluxel-host` presents one coherent Host Capability API over those services:
handles, asynchronous results, files, network, images, audio, codecs, and
device/platform information. It defines the capability contract but does not
absorb each implementation. It is intentionally not called a HAL, because that
would blur host capabilities with the GPU RHI.

### Runtime, ABI, and JavaScript

`fluxel-runtime` composes platform, host, assets, renderer, and execution
services in-process. It does not reimplement their policies or require every
internal crate to become externally callable.

`fluxel-runtime-abi` is a deliberately small, stable export facade. It owns
external handles, lifecycle entry points, and cross-boundary result conventions,
not loading, rendering, media, or business logic. A native embedder may package
it as a DLL, dylib, or shared object; a web embedder may use the same boundary
to define WASM imports and exports.

JavaScript VM and bridge work owns VM integration plus JS objects/functions,
handles, promises, asynchronous conversion, and access to the stable
host/runtime boundary. It does not target private renderer, asset, or capability
implementation details.

## Development stages

### Near term — resource and frame foundation

- [ ] Finish the renderer submission foundation.
- [ ] Establish cross-frame asset identity, readiness, residency, reuse, and
  safe release.
- [ ] Add loader boundaries for CPU mesh and texture data.
- [ ] Connect assets and renderer through ready GPU snapshots.
- [ ] Extend toward multi-draw `RenderPacket` submission.
- [ ] Establish platform basics and a minimal runtime composition.

The first meaningful end-to-end closure is a window that loads a Mesh and
Texture, retains them through `Assets` across frames, lets the renderer submit
continuous frames, and safely releases resources once they are no longer in
use. This closure proves the resource and lifetime path before higher-level
scene or application features are introduced.

### Medium term — host and embedding services

After that closure has real contracts, the intended work includes the unified
Host Capability API and codec, filesystem, network, image, audio, and platform
implementations; Canvas2D-oriented drawing; shader tooling; the runtime ABI;
and JS VM/bridge integration. These remain separate concerns: shader tooling
produces artifacts for renderers, and Canvas2D shares execution foundations
without redefining 3D renderer policy.

### Long term — platforms and ecosystem

Long-term direction is semantic consistency across Web, Windows, Android, and
iOS; suitable VM choices such as V8, QuickJS, JavaScriptCore, or a web runtime;
a JS SDK; and possible declarative UI and ecosystem layers. These are directions
only, with no planned version, schedule, or compatibility commitment.

Animation, skeletal systems, navigation, physics, ECS, editors, and full
Three.js compatibility do not currently have planned libraries or versions.
They require separate ownership decisions after the resource, runtime, and
embedding foundations have proved useful.
