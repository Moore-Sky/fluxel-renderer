# Fluxel RHI Design

**Status: 0.5.0 RHI with 0.2.6 fixed sRGB-decode renderer support**

This document records the architecture and reasons behind `fluxel-rhi`. It is
not a promise of a complete RHI. In 0.1.4 the crate opens one headless DX12 or
Vulkan device on Windows, creates lease-backed owned resources, and implements
a fixed Raster + Compute + Copy portion of `ExecutionBackend`: semantic
transition lowering, recording, one submission, completion, retirement, and
crate-private exact test readback. It does not implement surfaces,
presentation, renderer lowering, or a general shader/pipeline API.

The 0.2.0 addition is a single immutable buffer-upload boundary for renderer
snapshots. It does not generalize mapping or command recording: an upload owns
its newly created DeviceOnly destination, private staging allocation, fixed
Copy command, completion, and leases. Only a proven Complete result publishes
an `UploadedBuffer` with `CopyDestination` state. Pending, timeout, failure, and
accepted-unknown paths cannot expose initialized contents or release referenced
storage early.

The 0.2.1 addition is one closed Raster artifact for renderer-owned snapshot
data: slot-zero tightly packed `float32x3` clip-space positions, `uint32`
indices, triangle-list topology, and a fixed opaque fragment color. 0.2.2 adds
a distinct closed variant with the same vertex/index recipe plus exactly one
80-byte static uniform at group 0/binding 0: column-major `projection * view`
and linear RGBA base color, visible to both vertex and fragment stages. Its
portable identity includes the binding/layout recipe version. It does not
expose configurable vertex layouts, index formats, shaders, pipeline
descriptors, or general binding APIs. Renderer lowering remains outside the
RHI; these artifacts only give that lowering safe native execution targets.

0.2.3 adds one production immutable texture upload and one closed textured
raster artifact. Upload accepts only whole tight D2 `Rgba8Unorm`, one
mip/layer/sample, DeviceOnly CopyDestination+Sampled storage; the private RHI
pads staging rows to 256 bytes and publishes only a proven-complete
`UploadedTexture` in `CopyDestination`. The raster artifact adds one
fragment-visible `texture_2d<f32>` at group 0/binding 1 beside the existing
80-byte uniform. It uses no sampler and records its complete texture ABI and
mapping version in the portable identity. Expanding the formerly exhaustive
closed artifact enums is reflected by the RHI crate's 0.2.0 semver bump.

0.3.0 adds one closed explicit-UV raster recipe. It fixes position at vertex
slot 0/location 0/stride 12 and texture coordinates at slot 1/location 1/stride
8. A per-draw opaque binding retains both expected physical buffer identities,
their leases, the uniform and the sampled texture. The encoder uses a UV-only
pipeline epoch and rejects swapped, repeated, missing, partial-range, wrong-
usage, or stale roles before HAL calls; the native boundary repeats pass,
pipeline, usage, identity, slot, and exact-range validation. Existing closed
artifact identities and state machines are unchanged.

0.4.0 adds a second explicit-UV kernel without changing the integer-load
kernel. It fixes a filterable `Rgba8Unorm` texture plus an internally created
linear-min/mag, nearest-mip, clamp-to-edge sampler and calls
`textureSampleLevel` at level zero. The selected adapter's `SAMPLED_LINEAR`
format fact is retained in `HardwareCapabilities`, propagated to RenderGraph's
format capabilities and checked again before pipeline and binding creation.
The opaque binding owns the sampler, view, bind group, pipeline and resource
leases until terminal completion; sampler state is neither configurable nor a
graph resource.

0.5.0 adds a separate sRGB-source variant without changing either earlier UV
kernel. `Rgba8UnormSrgb` upload preserves encoded bytes, its native sampled view
performs RGB decode before the fixed linear-clamp operation, and the render
target remains linear `Rgba8Unorm`. The adapter's sRGB `SAMPLED_LINEAR` fact is
queried, stored, fingerprinted, and validated independently from UNORM.

The workspace temporarily vendors `wgpu-hal` 30.0.1 with upstream
gfx-rs/wgpu#10221 backported. The published 30.0.1 DX12 lowering maps an absent
comparison function to `ALWAYS`, which D3D12 validation rejects for a standard
filter; the backport maps it to `D3D12_COMPARISON_FUNC_NONE`. This patch remains
local only until a Rust-1.87-compatible upstream release contains the fix.

The companion [render-graph design](design-rendergraph.md) defines the logical
graph and its execution SPI. This document defines the native boundary that
later SPI implementation slices will use.

`0.1.0` begins this workspace's independent release sequence. Its code was
imported from a pre-migration snapshot; prior version numbers and Git history
are not part of this repository's release line.

## Boundary and responsibility

The workspace separates portable graph semantics from native API ownership:

```text
fluxel-rendergraph                    fluxel-rhi
-------------------                   -------------------------------
graph declaration                     explicit DX12/Vulkan selection
compile / immutable plan              headless adapter enumeration
creation-usage requirements           native device and queue ownership
ExecutionBackend contract       -->   fixed Raster + Compute + Copy implementation
TestRhi deterministic execution       native GPU execution subset
```

`fluxel-rendergraph` owns dependency, version, culling, access-state, and
creation-usage semantics. `fluxel-rhi` must translate those already-compiled
facts to native calls; it must not duplicate graph compilation or invent a
second descriptor/pipeline framework.

This boundary keeps the core crate free of production GPU dependencies and
makes native failures local to a small crate. It also avoids exposing
`wgpu-hal` types as public API: applications receive Fluxel domain values
rather than borrowing an unstable internal HAL representation.

## Device and owned-resource lifecycle

`Device::open(backend, options)` is deliberately explicit. It chooses no
fallback backend, enumerates the requested API's adapters, selects the given
native index, verifies baseline limits and validation policy, creates a
headless device and queue, then snapshots driver facts into `HardwareInfo` and
`HardwareCapabilities`.

```text
Device::open
  -> private instance
  -> enumerate adapters
  -> select adapter_index
  -> check baseline limits and validation policy
  -> open device + queue
  -> public Device { private native ownership, copied facts }
```

The public `Device` deliberately has no handle getter. Internally it shares the
queue, device, adapter, and instance in declaration order. Rust drops fields in
declaration order, so the private representation orders them queue, device,
adapter, instance: dependents disappear before the factory/instance state they
use. Each opaque resource and its cloneable lease retain that native owner;
the final owner destroys the resource exactly once before the device can be
destroyed.

Creation reuses RenderGraph's typed descriptors and usage sets. Malformed
shapes, incompatible format/usage combinations, and `Present` on an ordinary
owned texture fail before creation. Reported allowed usage is the conservative
portable projection of the successful HAL creation descriptor and format
facts. Descriptor usage is the caller's requested minimum and the result obeys
`requested ⊆ allowed`. Native normalization may widen that set and must report
the widening; buffer `StorageWrite`, for example, becomes storage read/write
rather than pretending that a native write-only storage capability exists.

`CopyBackend` remains permanently Copy-only. `ComputeBackend` is a separate
backend profile that adds Compute while retaining Copy; this prevents a Copy
conformance client from acquiring Compute incidentally. Both use one logical
queue, one serial encoder, and one submission. The queue is protected by an
`OpenedDevice`-shared operation lock: every submission and any error-path
`wait_for_idle` are mutually exclusive across cloned `Device`s and separately
constructed backends. This is required by the HAL queue contract, not a claim
of multi-queue support.

Same-state write dependencies remain required ordering/visibility facts. Vulkan
may emit a native memory barrier; DX12 Copy ordering can satisfy the same fact
within the serial command list without inventing a state transition. Exported resources
carry the graph's outgoing state; the test readback helper consumes that exact
state as its incoming state before transitioning to `CopySource`, so it cannot
silently repair a wrong graph result.

Buffer-to-buffer Copy validates non-zero, in-range 4-byte-aligned source
offset, destination offset, and size both while graph recording and at the RHI
native boundary. This carry-forward rule keeps invalid safe-API input away from
the unsafe HAL even if a caller bypasses the normal recording path.

The Compute slice accepts only the closed `ComputeKernel` set and exactly one
read/write storage-buffer binding recipe. Embedded WGSL is parsed and validated
to Naga IR before the HAL lowers it to DX12 or Vulkan native code. A portable
artifact identity is the WGSL source hash, entry point, workgroup shape, and
recipe version—not a comparison of backend-specific native binaries. Pipelines
and bindings are opaque, device-affine `Arc` values with cloneable leases; a
binding retains its pipeline and buffer through completion. Raw hardware facts
bound dispatch dimensions, every workgroup dimension, and total workgroup
invocations. RenderGraph rejects zero or excessive dispatch dimensions before
backend recording, and the RHI rechecks the fixed shader's workgroup
requirements before native creation.

`RasterBackend` is a third, fixed profile. It retains Copy and the closed
Compute artifacts only because X01 needs one real Raster→Compute→Copy chain in
one encoder and one submission; it does not turn those APIs into a general
graphics interface. Its raster scope is a single `Rgba8Unorm`, D2,
single-mip/layer/sample color attachment, fixed vertex/index recipes, and the
R01 clear/triangle and R02 indexed viewport/scissor fixtures. The U02 renderer
fixture adds the distinct fixed `float32x3 + uint32` recipe. The U03 variant
adds only its closed 80-byte Camera/material uniform binding; neither widens an
older artifact into a general graphics interface. Raster uniform bindings are
opaque, device- and pipeline-affine lease-backed values: they retain the
matching pipeline and uniform buffer through completion and reject a wrong
device, pipeline, usage, range, or dynamic offset before native recording. X01 samples the
complete attachment through a closed binding recipe, packs row-major RGBA8
pixels into an RW storage buffer, then copies it to an exported buffer.
Raster artifacts, bindings, views, attachment resources, encoder, command
buffer, and submission retain device-affine leases until terminal completion.
Structured creation/recording/submission errors distinguish known rejection
from accepted-unknown completion failure; no error path may release native
objects early. `src/imp.rs` remains the sole unsafe boundary, whose rationale
covers device identity, states, ranges, alignment, thread serialization, and
lifetime for the fixed Raster path as well as earlier slices.

All HAL calls and all `unsafe` are contained in `src/imp.rs`. Each unsafe block
documents the validity and lifetime condition it relies on. The safe public
layer has no `unsafe`, native pointers, or HAL types. This is an intentional
audit boundary: later resource and command code may expand the private module,
but cannot silently make native lifetime rules part of the application API.

## Validation is fail-closed

`Validation::Required` means a native facility must be positively established, not merely requested:

```text
Required
  -> request HAL validation instance flag
  -> DX12: verify opened device exposes ID3D12InfoQueue
  -> Vulkan: verify VK_LAYER_KHRONOS_validation
             and VK_EXT_validation_features
  -> otherwise return OpenError::ValidationUnavailable
```

The Vulkan validation-features check matters because the HAL uses that facility
to request synchronization validation. Requiring it makes a `Required` result
meaningful for the A1 bootstrap rather than silently accepting a partial
installation. The project does not claim that native validation proves graph
correctness: command state and synchronization validation become relevant only
after this crate emits real commands.

`Validation::Disabled` remains the default to allow normal device probing on
machines without developer tooling. It does not hide the distinction: callers
that need diagnostic guarantees opt into `Required` and receive an error if the
prerequisites are missing.

## Hardware facts, not policy

`HardwareInfo` and `HardwareCapabilities` expose driver-reported identity and a
small selected set of limits without masking them into a common profile. That
is intentional. A renderer may need raw information to select a policy, log a
bug report, or decide that its own portability profile is unavailable.

The current facts are not a commitment to a final adapter-selection API or a
complete capabilities model. `HardwareInfo`, `HardwareCapabilities`, and
`DeviceOptions` should therefore still be treated as provisional public shapes.
Future versions may add device/driver IDs, format features, queues, memory
budgets, limits, and profile evaluation. The current `adapter_index` is a useful
bootstrap probe; a mature integration should instead enumerate facts, let
application/renderer policy choose an adapter, then open that choice.

## Features and platform policy

The package has `dx12` and `vulkan` feature flags, both enabled by default.
Feature flags control which Windows native backends compile. They are not a
runtime portability abstraction: requesting an omitted backend returns
`BackendDisabled`, while all current non-Windows targets return
`PlatformUnsupported`.

Windows is intentionally the A1 scope because it permits bringing up both
target native APIs under one ownership model. This is not evidence that Vulkan
is intrinsically Windows-only, nor a decision against future platform
backends. Platform expansion must preserve the same public boundary: explicit
selection, safe ownership, private native types, factual capability reporting,
and clear validation semantics.

## CI and test evidence

The normal test suite verifies descriptor rejection and lowering projections without opening a driver where
possible. Windows CI builds and tests each DX12/Vulkan feature combination, so
it is a native compile/link gate. Actual device-open and required-validation
tests are `#[ignore]`, because a hosted runner's installed driver and validation
layers are environmental facts, not reproducible source-level prerequisites.

Developers run ignored tests explicitly on provisioned hardware. The 0.1.2
fixtures execute C01 partial buffer copy, C02 texture copy with padded rows, and
C03 same-state overlapping WAW separately on both backends. Each compares
readback with a CPU oracle and requires programmatically collected diagnostics
to be empty. The 0.1.3 K01 fixture executes a fixed in-place wrapping add;
K02 executes ordered wrapping add then multiply on the same read/write storage
buffer, exercising a same-state RW-to-RW dependency. Each runs separately on
DX12 and Vulkan, compares with its CPU integer oracle, and requires collected
diagnostics to be empty. Those earlier fixtures establish only Copy and fixed
Compute correctness, not Raster or presentation conformance. The 0.1.4 release gate additionally
requires R01 clear/triangle, R02 indexed viewport/scissor, and X01
Raster→Compute→Copy to run as separate DX12 and Vulkan fixtures and compare
exact texture/buffer readback bytes with their CPU pixel/packing oracles.
Readback must use the export's actual outgoing state and strip texture-row
padding. Required-validation diagnostics are collected and empty on the
recorded exact-SHA DX12/Vulkan release fixtures.

## Evolution constraints

Further work remains vertical rather than speculative. The fixed native
`ExecutionBackend` honors the current core contract end to end:

1. Allocate transient resources from the compiled usage summary and report the
   actual allowed operation set for every transient and import, allowing core
   frame resolution to validate the required subset.
2. Map compiled state transitions to backend-correct barriers or ordering.
3. Record only declared fixed Raster, Compute, and Copy work and resolve only
   authorized pass-local resources.
4. Submit work, expose completion, retain leases until completion, and validate
   crate-private readback against native conformance oracles.
5. Add surface acquisition/presentation and multi-queue only with explicit
   ownership and synchronization semantics.

The current user model is: `fluxel-rhi` executes the fixed Raster, Compute,
and Copy artifact subset on native DX12/Vulkan; `TestRhi` remains the
deterministic CPU contract backend. The crate must not imply surface/present,
renderer, general texture-compute, or general shader/pipeline support in its
API, examples, benchmarks, or release notes.

## Portable lowering constraints

A logical pass, CPU recording job, native encoder/command buffer, and
submission batch are different units. An adapter may combine or split them
while preserving order, access, and lifetime contracts. GPU dependencies are
not a ready-made CPU recording DAG, and GPU completion is not CPU recording
completion.

Browser adapters remain fail-closed:

- WebGPU may use backend-managed native state, but it still validates compiled
  usage and ownership;
- WebGL2's immediate context must not be presented as a reusable native command
  buffer;
- when compute is unavailable, the renderer explicitly chooses a legal raster
  variant; RenderGraph does not silently rewrite compute semantics;
- queue labels and capabilities describe legal operations, not distinct
  hardware queues or promised parallelism.

These rules keep one ExecutionPlan vocabulary portable without pretending that
DX12, Vulkan, WebGPU, and WebGL2 share the same command model.
