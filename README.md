# Fluxel Rendering

`fluxel-rendering` is Fluxel's host-agnostic Rust rendering kernel. Its typed
render graph, native RHI boundary, and renderer layer provide portable drawing
semantics and GPU execution without owning an application host. Cross-frame
asset identity, platform services, runtime packaging, and language SDKs belong
to adjacent Fluxel repositories described in the
[Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

The kernel accepts a host-supplied surface target, resize information, resource
bytes, scene or canvas updates, and explicit render calls. It does not create a
window, run an event loop, read files or URLs, collect input, provide storage,
audio, video, networking, or decide when the next frame runs. Browser and
mini-game adapters drive its WASM form; native hosts drive its library form.

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

## Released source dependency

The workspace releases its three crates together.  Git consumers must pin the
release tag rather than follow `main`:

```toml
fluxel-rendergraph = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.7.0" }
fluxel-rhi = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.7.0" }
fluxel-renderer = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.7.0" }
```

`v0.7.0` and each package's `0.7.0` version identify the same historical
workspace release. See [RELEASING.md](RELEASING.md) for the release gate.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](documents/design-rhi.md)
- [Renderer design](documents/design-renderer.md)
- [Architecture decisions](documents/adr/README.md)
- [Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
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

On a clean Windows `x86_64-pc-windows-msvc` checkout, the release-only GPU
gate is:

```powershell
./scripts/conformance.ps1
```

It runs every workspace ignored fixture, injects the checked-out commit through
`FLUXEL_TEST_COMMIT`, and preserves a manifest and complete log under
`target/conformance/<sha>/`, including failures. Successful release artifacts
are retained on the matching [GitHub Release](https://github.com/fluxel-project/fluxel-rendering/releases).

## Roadmap

This roadmap covers this workspace only. Near-term work is deliberately more
specific than distant direction; no item is a version or schedule commitment.
For ownership outside the renderer workspace, see the
[Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

### Completed — Stage 0 baseline

**Status:** Complete.

**Latest retained evidence:** [v0.7.0](https://github.com/fluxel-project/fluxel-rendering/releases/tag/v0.7.0),
tag commit `6bd3a25`, with 83/83 ignored real-GPU cases passing on AMD Radeon
780M Graphics across DX12 and Vulkan.

- [x] Use one release version and tag rule for workspace crates that ship
  together, and pin every documented Git dependency example to a tag or
  revision.
- [x] Preserve structured renderer errors through the public boundary.
- [x] Provide one repeatable CPU, CI, and real-GPU conformance entry point that
  records the commit, environment, diagnostics, and partial failure evidence.
- [x] Audit and reduce the RenderGraph public export surface to the current
  portable contract.
- [x] Freeze fixed-recipe growth; do not add another `draw_*` family member or
  public upload-state type.
- [x] Retain the complete baseline on the final fixed Stage 0 source commit.

`v0.7.0` is the fixed Stage 0 source and evidence commit. The commits after
that tag only realigned repository and ecosystem documentation; they did not
change Cargo manifests, Rust sources, examples, CI, the conformance gate, or
the released GPU behavior, so they do not create a second GPU certification
target.

### Current — Stage 1.1 DX12 first visible image

The kernel will gain only the narrow presentation boundary needed for a host
validation harness; it will not become a host-services runtime.

- [ ] Create a Windows window and the narrow DX12 surface or swapchain path
  needed by the demo harness.
- [ ] Clear an acquired image, draw a fixed triangle, and present it.
- [ ] Handle close, wait for required GPU work, and release resources in a
  valid order.
- [ ] Retain a screenshot, backend and adapter diagnostics, and focused tests
  for new platform-independent state.

### Unscheduled optimizations

- [ ] Batch compatible render packets while preserving declared ordering.
- [ ] Lower compatible repeated draws as instances when the selected recipe
  permits it.
- [ ] Reuse compatible renderer-side pipelines and bindings across a frame.
- [ ] Add multi-queue scheduling, parallel command recording, recording caches,
  transient resource aliasing, or more aggressive GPU scheduling and memory
  optimization only after representative benchmarks or profiling demonstrate a
  concrete need.

### Out of scope

Durable asset identity and caching, general file or URL loading, broad scene
graphs, animation, application host policy, and higher-level JS or UI APIs
belong to other Fluxel layers. This workspace owns backend bring-up and the
minimal presentation boundary needed to prove RHI and renderer behavior; it
does not own a general platform runtime or host services.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
