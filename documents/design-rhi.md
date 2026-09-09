# Fluxel RHI Design

**Status: 0.1.1 owned-resource baseline**

This document records the architecture and reasons behind `fluxel-rhi`. It is
not a promise of a complete RHI. In 0.1.1 the crate safely opens one headless
DX12 or Vulkan device on Windows and creates lease-backed owned buffers and
textures with verified actual usage. It does not implement `ExecutionBackend`, command encoding,
submission, completion, readback, surfaces, or presentation.

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
ExecutionBackend contract       -->   future contract implementation
TestRhi deterministic execution       future native GPU execution
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
facts. It is an authorization upper bound, not an exhaustive claim about every
operation a physical API might technically permit (notably DX12 buffers).

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

Developers run ignored tests explicitly on provisioned hardware. The 0.1.1
fixture creates buffers and textures on both backends with required native
validation, verifies exact projected usage and rejection recovery, and exercises
lease teardown. It establishes resource creation only, not barrier, command, or
presentation conformance.

## Evolution constraints

Future work is vertical rather than speculative. A native `ExecutionBackend`
implementation begins only when it can honor the core contract end to end:

1. Allocate transient resources from the compiled usage summary and report the
   actual allowed operation set for every transient and import, allowing core
   frame resolution to validate the required subset.
2. Map compiled state transitions to native barriers.
3. Record declared copy/compute/raster work and resolve only authorized
   pass-local resources.
4. Submit work, expose completion, retain leases until completion, then add
   readback and native conformance tests.
5. Add surface acquisition/presentation and multi-queue lowering only with
   explicit ownership and synchronization semantics.

Until those slices land, the correct user model is: `fluxel-rhi` opens and
describes a native device; `TestRhi` executes the render-graph execution
contract deterministically on the CPU. The crate must not imply otherwise in
its API, examples, benchmarks, or release notes.

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
