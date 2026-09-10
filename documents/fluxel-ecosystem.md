# Fluxel Ecosystem

Fluxel is intended to grow as focused libraries with explicit ownership
boundaries, not as one renderer-shaped engine. This document is an ecosystem
map and development direction, not a release schedule, compatibility promise,
or commitment to create every named library.

## Layers

### L0 — Foundation capabilities

`fluxel-base` provides small, stateless, cross-platform utilities such as
Base64, hashing such as MD5 where required, encoding, and byte/string helpers.

`fluxel-time` provides clocks, timers, timeouts, intervals, and frame timing.

`fluxel-fs` provides files, directories, paths, streams, and random-access
read/write operations. `fluxel-storage` separately provides application data
persistence such as key-value data, JSON, blobs, and local storage.

`fluxel-net` provides HTTP, WebSocket, download, and upload capabilities.
`fluxel-input` provides mouse, keyboard, touch, wheel, and gamepad input;
input events belong to this library rather than to a separate events layer.

`fluxel-image` provides image decoding, encoding, and pixel data.
`fluxel-audio` provides audio decoding, playback, volume, and output-device
integration.

`fluxel-platform` provides application startup, windows, screens, DPI,
resizing, the main loop, surfaces, and frame-rate control. Foreground and
background lifecycle policy is not planned yet.

### L1 — Resources

`fluxel-loader` obtains CPU resources from memory, files, URLs, and supported
formats. It loads data such as textures, meshes, shaders, JSON, and eventually
formats such as GLTF.

`fluxel-assets` owns resource identity, typed handles, generations,
cross-frame references and lifetime, cache/reuse, CPU/GPU residency, and safe
release.

The distinction is intentional: the loader decides how resource data is
obtained; assets decide how an obtained resource remains available and when it
can be released.

### L2 — GPU and rendering

`fluxel-rhi` owns GPU hardware abstraction and backend implementations,
including DX12, Vulkan, and future supported backends.

`fluxel-rendergraph` owns single-frame GPU passes, resource dependencies,
synchronization, and portable execution plans.

`fluxel-renderer` owns 3D render submission, render packets, ordering,
grouping, batching, instancing, and lowering into the RenderGraph. It consumes
prepared resources; it does not own durable asset identity or host policy.

`fluxel-canvas` will own Canvas-style 2D drawing, including images, sprites,
text, paths, transforms, clipping, blending, and offscreen work.

`fluxel-shader` may later own shader compilation, reflection, variants, and
caching once those concerns have real independent demand. Renderers consume
its outputs rather than owning its toolchain policy.

### L3 — Runtime

`fluxel-runtime` composes foundation capabilities, assets, 2D and 3D
rendering, and platform services into a running application runtime.

`fluxel-runtime-abi` is the stable outer ABI and packaging boundary. It can
package the complete runtime as a DLL, SO, dylib, or WASM module. Its
exports represent the assembled runtime—2D, 3D, resources, networking, files,
and audio—not merely a collection of miscellaneous native functions.

Internal libraries continue to use direct Rust APIs. The ABI boundary exists
for embedders and packaging, not as an internal replacement for Rust types.

### L4 — JavaScript

`fluxel-vm-js` abstracts JavaScript VM integration, including V8, QuickJS,
JavaScriptCore, and web or Edge environments where appropriate.

`fluxel-jsbridge` owns Rust/JavaScript objects, handles, functions, promises,
asynchronous conversion, and data conversion.

`fluxel-js` is the developer-facing JavaScript API. It may eventually take
inspiration from mini-game or Three.js-style APIs, but internal runtime and
renderer design must not be shaped by that public API.

### L5 — Higher-level UI

`fluxel-ui` is a future home for a mini-program or Vue-style declarative UI.
It is a direction only; its design is intentionally not specified yet.

## Development stages

### Near term

The immediate goal is to move the 3D renderer from single-frame experimental
capability to a durable runtime foundation.

- [ ] Finish the current renderer submission foundation: `DrawList`,
  `RenderPacket`, multiple draws, per-object transforms, and clear grouping or
  batching boundaries.
- [ ] Establish `fluxel-assets` with stable handles, generations,
  cross-frame references, CPU/GPU residency, cache/reuse, and safe release.
- [ ] Establish `fluxel-loader` for memory and file loading of meshes and
  textures; add formats such as GLTF only after the basic boundary is proven.
- [ ] Connect loader, assets, and renderer so resources load once, are reused
  by multiple objects and frames, and release safely when no longer needed.
- [ ] Establish minimal `fluxel-platform` support for a window, main loop,
  clock/frame timing, surface, resizing, and frame-rate control.
- [ ] Establish a minimal `fluxel-runtime` that runs a long-lived 3D
  application.

Completion means an application can start, load resources, render multiple
objects across many frames, reuse resources, and manage their lifetime
correctly.

### Medium term

The next goal is a practical lightweight game runtime rather than a renderer
demo.

- [ ] Add the required foundation systems: time, input, filesystem, storage,
  networking, image, audio, and base utilities.
- [ ] Add `fluxel-canvas` and establish the runtime ABI.
- [ ] Package complete native and WASM runtimes.
- [ ] Add JavaScript VM abstraction, the JS bridge, and a minimal JS API.

The intended result is a small application or game that can use input,
networking, storage, audio, 2D, and 3D without requiring Unity or Godot.

### Long term

Long-term work is direction only: consistent runtime semantics across Windows,
Web, Android, and iOS; stronger DX12, Vulkan, Metal, WebGPU, and WebGL2
backends; suitable VM integrations; a fuller JavaScript SDK; declarative UI;
and richer scene or game capabilities.

Multi-queue scheduling, parallel command recording, resource aliasing, and
other execution optimizations are introduced only when representative
benchmarks and profiling demonstrate a concrete need.

Possible directions include animation, skeletal animation, physics,
navigation, Blender integration, a Three.js-compatible facade, and a
Vue-compatible declarative UI. They may be reconsidered after the core runtime
closure is proven, but have no current library, version, compatibility, or
delivery commitment. ECS and editor work are likewise uncommitted.
