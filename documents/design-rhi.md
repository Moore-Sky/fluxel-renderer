# Fluxel RHI Architecture

## Purpose

`fluxel-rhi` is Fluxel's headless native execution boundary. It lowers a
portable `fluxel-rendergraph::ExecutionPlan` into a deliberately small DX12 or
Vulkan command subset and retains native objects until GPU work is terminal. It
is not a renderer, surface layer, or general graphics API.

```text
Renderer       scene policy, snapshots, and closed render recipes
                    |
RenderGraph    declarations, dependencies, states, culling, immutable plan
                    |
RHI            native ownership, lowering, submission, completion, retirement
                    |
DX12 / Vulkan
```

RenderGraph owns portable meaning. RHI owns native object lifetime, barrier
encoding, recording, queue interaction, completion, and diagnostics. RHI can
reject a plan or binding which its closed native slice cannot safely represent,
but never recompiles graph dependencies, exposes native handles, or becomes a
second graph compiler. Renderer policy, assets, shader/material selection, and
frame scheduling remain above it. See [ADR-0001](adr/0001-assets-outside-rendergraph.md),
[ADR-0003](adr/0003-serial-execution-lowering.md), and
[ADR-0006](adr/0006-no-general-pipeline-yet.md).

## Public facade and module boundaries

`src/lib.rs` is the small safe facade. It exports explicit device opening,
driver facts, owned resources, fixed artifacts, execution backends, and
structured errors. It never exports `wgpu-hal`, native pointers, queues,
command allocators, image views, or host mapping.

```text
lib.rs
 |- test_support.rs  feature-gated conformance observation and fault injection
 |- resource/
 |   |- creation/ safe owned-resource, immutable-upload, compute and raster factories
 |   |- artifact/ portable fixed shader/pipeline identities
 |   `- lease/    completion-aware resource and accepted-work lifetime
 |- execution/
 |   |- provider/ RenderGraph object resolution
 |   `- raster/   raster capabilities, registry, operations and recording state
 `- imp/
     |- device_open.rs / lowering.rs   native bootstrap versus portable lowering
     |- command/ / pipeline/ / bindings/  closed native concerns split by pass kind
     |- upload/    production upload, staging mechanics and test observers
     `- submission.rs / resource.rs    queue completion and native ownership
```

`test_support.rs` is a doc-hidden, non-default fixture boundary; it exposes no
native handles or general readback API. `resource/` defines resource and artifact
lifetime. `execution/` implements the RenderGraph SPI using those opaque values.
`imp/` is the sole native
boundary: its Windows implementation contains HAL types and all `unsafe`, while
`imp/stub.rs` is the fail-closed counterpart for non-Windows targets and for
Windows builds with no native backend feature enabled. Splitting this tree
does not relax the boundary: no code outside `imp` makes an application obey
HAL lifetime rules. See [ADR-0002](adr/0002-rhi-unsafe-containment.md).

The public opening contract is explicit:

```rust
Device::open(Backend::Dx12 | Backend::Vulkan, DeviceOptions)
```

There is no fallback. `DeviceOptions::adapter_index` is a backend-native
bootstrap choice, not an adapter-policy API. `HardwareInfo` exposes driver
identity and `HardwareCapabilities` exposes the raw facts needed by current
fixed recipes, including filtering and compute limits. They are facts, not a
portability tier or a guarantee that an arbitrary graph can execute; renderer
policy evaluates them.

`OpenError` distinguishes unsupported platform, disabled feature, unavailable
adapter, unavailable required validation, baseline-limit failure, and
loader/driver failure. `ResourceCreateError`, upload errors,
`NativeExecutionError`, and `WaitError` retain the same principle: callers can
tell known pre-submit rejection from terminal failure after acceptance.

## Backend selection, capabilities, and validation

The `dx12` and `vulkan` Cargo features control which Windows backends compile;
both are default features. On Windows, requesting an omitted feature returns
`OpenError::BackendDisabled`. On current non-Windows targets, requesting either
backend returns `OpenError::PlatformUnsupported`. A disabled backend and an
unsupported platform are intentionally different, tested outcomes.

| Target | Backend feature | Result |
| --- | --- | --- |
| Windows | enabled | private DX12 or Vulkan path, subject to driver and validation facts |
| Windows | omitted | `BackendDisabled` |
| non-Windows | any | `PlatformUnsupported` through the stub |
| any target | `test-support` | doc-hidden conformance helpers, never a production native API |

Support is currently headless Windows DX12/Vulkan only. Headless means no
window, surface, swapchain, acquisition, present, resize, or loss-recovery
contract. This is not a claim that Vulkan is intrinsically Windows-only: a new
platform must preserve explicit selection, opaque ownership, factual
capabilities, and structured failure. Native platform paths must run on native
environments rather than be inferred from cross-compilation; see
[ADR-0008](adr/0008-native-platform-test-gates.md).

`Validation::Required` is fail-closed. The private opening path requests and
verifies native diagnostics before returning a device: DX12 needs a usable
information queue; Vulkan needs `VK_LAYER_KHRONOS_validation` and
`VK_EXT_validation_features`, used by the HAL for synchronization validation.
`Validation::Disabled` permits normal probing but is never evidence that
validation was active.

## Ownership, identity, and creation

`Device` holds an `Arc<OpenedDevice>` and an opaque RenderGraph
`DeviceIdentity`. Resources, pipelines, bindings, encoders, and command
buffers are device-affine; safe layers compare identity before an operation
reaches `imp`, preventing accidental cross-device mixing.

Owned `Buffer` and `Texture` values are `Arc`-backed opaque objects. Each
creates a cloneable `BufferLease` or `TextureLease`; its internal shared object
also retains the opened native device. Thus dropping a `Device` or public
resource cannot destroy native storage while a lease exists. The last resource
or lease destroys it once. `ResourceLease` unifies buffer and texture leases
for retirement. Pipelines and bindings follow the same model, retaining their
native recipe objects and referenced resource leases.

```text
Device -> Arc<OpenedDevice>
               ^ retained by shared resource/pipeline/binding state
Buffer / Texture / artifact value -- cloneable lease --> in-flight submission
```

`Device::create_buffer` and `Device::create_texture` use RenderGraph typed
descriptors and usage sets through `BufferDescriptor` and `TextureDescriptor`.
They validate shape, policy, format/usage compatibility, and reject `Present`
for ordinary owned textures before HAL. Requested usage is a minimum;
`allowed_usage()` reports conservative actual creation capability, so
`requested_usage ⊆ allowed_usage`. Native normalization may widen a capability
(for example, storage write becomes native storage read/write), and that
widening is reported rather than hidden. No raw handle can bypass this rule.

## Immutable uploads and publication

Uploads are narrow construction state machines, not a general mapping API.
Immutable buffer and texture upload create a DeviceOnly destination and private
staging allocation, record a fixed copy, submit it, and retain both through a
terminal completion.

```text
validate -> Pending*Upload -> poll/wait -> Uploaded* | Incomplete*Upload
                    |                       |
             owns staging + target      only Complete publishes
```

Published buffers and textures are immutable snapshots. Texture upload accepts
the closed whole tight RGBA8 image domain; RHI handles required staging-row
padding. A successful `UploadedBuffer` or `UploadedTexture` reports
`CopyDestination` as its known outgoing state. Pending, timeout, failure,
early-drop-after-acceptance, and accepted-unknown work never publish initialized
contents. The renderer can release a reservation before acceptance but poisons
the affected generation when its terminal state is unknown. See
[ADR-0004](adr/0004-accepted-unknown-quarantine.md).

## Closed artifacts and bindings

The native API supports evidence-backed *closed* compute and raster recipes.
`ComputeKernel` and `RasterKernel`, identities, pipeline values, and binding
constructors represent exactly the combinations implemented by this crate; they
are not pipeline builders.

Compute accepts a closed WGSL kernel set and fixed storage-binding recipes.
Embedded WGSL is parsed and validated into Naga IR, then lowered privately for
DX12 or Vulkan. Its portable identity is Fluxel data—source hash, entry point,
workgroup shape, and recipe version—not a comparison of DXIL with SPIR-V.
Dispatches must be nonzero and fit portable workgroup-count limits; creation
also checks each shader workgroup dimension and total invocation count against
the hardware facts.

Raster uses separate types for distinct ABI recipes rather than optional fields
that could combine incompatible choices. The supported family consists of the
current opaque triangle/indexed, camera/material uniform, texture-load,
explicit-UV, linear-clamp UNORM, linear-clamp sRGB, position-plus-normal
Lambert, and position-plus-vertex-color variants. Each fixes streams, index
format, texture/view domain, sampler behavior where used, binding slots,
uniform layout, attachment domain, and shader semantics. The sRGB recipe
retains encoded bytes and uses a sampled native view so decode occurs before
linear filtering; it is not inferred from UNORM support. Lambert has its own
position/normal ABI and fixed lighting rules.

The vertex-color artifact is likewise closed: it accepts `u32` indices,
tightly packed `f32x3` positions in vertex slot zero, and tightly packed linear
normalized `UNORM8x4` colors in vertex slot one. It binds exactly one 80-byte
camera/tint uniform visible to vertex and fragment stages, rasterizes into
`Rgba8Unorm`, perspective-interpolates the normalized color, and multiplies it
by the linear tint. Its portable identity records the two stream slots,
strides, shader locations, and vertex-color recipe version. The safe provider,
recording state machine, and native boundary all require the selected pipeline,
its one binding, both vertex slots, and the index stream before an indexed draw
can be emitted; no arbitrary second vertex stream can be supplied.

Pipelines, `ComputeBindings`, and raster binding types are opaque,
device-affine, and lease-backed. Binding creation rechecks pipeline/identity,
usage, ranges, and stream roles. Recording checks pass scope, pipeline epoch,
binding compatibility, fixed vertex-slot completeness, duplicate slots, index
format, viewport/scissor, and draw/dispatch preconditions. The native boundary
repeats native-relevant checks. RenderGraph validates portable declarations;
RHI independently protects its safe API before unsafe HAL calls.

General WGSL, reflection, descriptor layouts, arbitrary textures/samplers,
vertex formats, dynamic offsets, and draw state are intentionally absent. A
general layer waits for real renderer requirements that can specify safe
ownership and portability semantics; see [ADR-0006](adr/0006-no-general-pipeline-yet.md)
and [ADR-0007](adr/0007-closed-fixed-renderer-recipes.md).

## Execution-plan lowering

`CopyBackend`, `ComputeBackend`, and `RasterBackend` implement RenderGraph's
`ExecutionBackend` contract. They expose normalized portable capabilities and
allocate transients from a compiled descriptor/usage summary.

```text
CopyBackend       Copy only
ComputeBackend    Copy + fixed Compute
RasterBackend     Copy + fixed Compute + fixed Raster
```

Unsupported command families fail closed: a Copy backend cannot incidentally
record Compute or Raster. `ComputeObjectProvider` and `RasterObjectProvider`
map RenderGraph IDs to previously created fixed opaque objects. They resolve
only declared resources and reject missing, foreign, dynamic-offset, or
incompatible binding recipes; they do not own assets or infer renderer policy.

Each execution follows the plan's immutable order:

```text
allocate/bind resources
 -> apply planned transition or same-state memory dependency
 -> record declared closed commands in one encoder
 -> finish and submit one command buffer on logical queue 0
 -> expose completion and retain leases until retirement
```

`ResourceAccessState` and transitions originate in RenderGraph. RHI maps them
to API-specific state and memory/order operations. `before == after` is still a
memory dependency, never a discarded no-op: serial DX12 copy ordering may be
sufficient, while Vulkan may require a native memory barrier, but the portable
visibility/order result must match.

Copy validation is intentionally duplicated. RenderGraph validates recording;
the RHI boundary again rejects zero, overflowed, out-of-range, or unaligned
buffer-copy offsets and sizes (the source offset, destination offset, and size
must each be 4-byte aligned). Texture regions are likewise rechecked. This
keeps direct safe RHI callers from sending invalid data into unsafe HAL.

## Submission, completion, and quarantine

The current lowering has one logical queue, serial encoder recording, and one
submission per graph execution. It is a correctness baseline, not a claim about
hardware queue count or a premature performance strategy.

`OpenedDevice` owns a shared queue-operation mutex. Submit and every
error-path `wait_for_idle` hold it across cloned devices and independently
constructed backends, satisfying HAL external synchronization. It does not
provide multi-queue scheduling, parallel recording, command caching, or
aliasing.

Completion is `Pending`, `Complete`, or terminal `Failed(CompletionFailure)`.
A known pre-submit failure returns `SubmitRejected` and transfers no work. If a
native submission may have been accepted, RHI returns completion that can later
fail and quarantines the command allocator/buffer, temporary views, private
staging, bindings, resources, and leases. It never guesses outgoing state or
releases data early. Dropping a pending `FrameSubmission` transfers completion
and leases to non-blocking retirement; collection releases them only at a
terminal state. This preserves safety without blocking destructors and is the
rule in [ADR-0004](adr/0004-accepted-unknown-quarantine.md).

## Readback, diagnostics, and evidence

Readback is doc-hidden `test-support` for conformance fixtures, never a public
mapping API. It consumes an exported resource's reported outgoing state and
lease as its actual incoming contract, then transitions to `CopySource`, copies,
waits, and exposes CPU bytes. It must not assume or force a convenient state,
because that could mask an incorrect graph transition. Texture helpers strip
native row padding before comparing row-major CPU-oracle bytes.

The feature also exposes scoped validation diagnostic capture and fault/observation
helpers for workspace tests. They prove lifecycle and failure paths but expose
neither general native commands nor host mapping.

Ordinary tests cover descriptors, identity, state machines, providers, cfg
paths, and compile/link behavior; none proves driver correctness. Hardware
fixtures are ignored in ordinary test runs. On provisioned Windows hardware,
DX12 and Vulkan run the same compiled plan with `Validation::Required`, exact
readback, an independent CPU oracle, and programmatically empty diagnostics.
Release evidence records exact commit, backend, GPU/driver, inputs,
expected/actual result, completion, outgoing state, and diagnostics. See
[ADR-0005](adr/0005-gpu-conformance-evidence.md).

## Unsafe rationale and extension rules

Only `imp/` invokes HAL or contains `unsafe`. Its safety argument is supplied
by the surrounding contracts: identity, descriptor/usage, pass/transition,
range/alignment, workgroup limits, and binding-role checks occur before native
calls; queue operations are serialized; and native owners plus all in-flight
leases survive terminal completion. Private field order tears down dependent
queue/device/resource state before the instance/factory which created it. Each
unsafe operation documents its narrower validity and lifetime condition.

Future platform backends, surfaces, fixed recipes, or a justified generalized
artifact layer must preserve these rules:

1. RenderGraph first specifies portable declaration, dependency, state, and
   completion semantics.
2. RHI exposes the smallest opaque safe contract required to lower them.
3. Native objects and unsafe reasoning remain private to `imp`.
4. New behavior receives portable tests and native CPU-oracle evidence before
   it is described as cross-backend correctness.
5. Multi-queue, parallel recording, aliasing, and other performance work need
   profiling/benchmark evidence plus their own synchronization/lifetime design.

This keeps RHI a narrow enforceable boundary: portable plans stay portable,
native hazards stay private, and correctness claims rest on observable evidence
rather than successful compilation.
