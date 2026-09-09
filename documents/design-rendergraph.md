# Fluxel RenderGraph architecture

**Status: 0.1.3 Copy + fixed-Compute execution contract**

This workspace starts its own release sequence at `0.1.0`. Its code was
imported from a pre-migration snapshot; prior version numbers and Git history
are not part of this repository's release line.

## Scope

`fluxel-rendergraph` is the in-frame GPU-work planner. It turns a mutable,
typed declaration graph into an immutable compiled snapshot and, through a
narrow provisional SPI, a serial execution protocol. It owns logical resource
versions, access semantics, dependencies, culling, validation, transition
planning, and creation-time usage requirements.

It does not own asset loading, scenes, shaders, pipeline or descriptor policy,
GPU memory allocation policy, native device handles, surfaces, or presentation.
Those remain renderer/RHI responsibilities. The native device boundary and its
rationale live in [design-rhi.md](design-rhi.md).

## The central decision: declare data, derive order

A render graph is valuable only if resource declarations, rather than callback
order, are the source of truth:

```text
resource accesses -> dependencies -> live pass set -> transitions -> execution plan
```

Each pass declares `ResourceAccess` during setup. Names, captured values,
slot-graph edges, and explicit order edges cannot stand in for a missing
access. This lets the compiler consistently answer what may run, which version
is read or written, which work is live, and which state or memory ordering is
required. It also prevents a future backend from guessing native barriers from
opaque user command closures.

`depends_on` exists only for non-resource external protocol or diagnostic
ordering. It constrains the pass DAG but creates no resource lifetime,
transition, or synchronization fact.

## Versioned resources and range semantics

Logical `TextureVersion` and `BufferVersion` preserve a linear content story.
A reader borrows a version; a writer consumes it and produces its successor.
There is no implicit merge of two writer branches. Rust ownership shapes make
ordinary authoring natural, while compiler validation remains authoritative for
stale/foreign versions, branches, conflicts, and initialization.

Versions identify whole logical resources; access and validity are tracked by
texture subresource range or buffer range. A partial successor inherits
untouched content and validity. `WriteCoverage::Full` proves the declared range
initialized; `Unknown` adds no fact. Attachment load/clear/store/discard
semantics are likewise compile-visible. This separates intended GPU access from
the stronger claim that contents are known initialized.

The declared range is part of access identity and cannot be widened or changed
while recording. `WriteCoverage::Full` means the complete declared range, not
the whole resource. `WriteCoverage::Unknown` neither proves new valid contents
nor destroys validity inherited from an earlier version. A `read_write_*`
access has no coverage escape hatch: its full read range must already contain
defined data, and its successor inherits that validity.

A pass cannot consume a successor version produced earlier in its own setup.
That would imply ordering stages and a barrier inside an opaque execute
callback, which the graph cannot inspect. A single `read_write_*` remains valid
because it consumes a version produced before the pass. The compiler also
rejects two writers that consume the same version, even when their present
ranges do not overlap; whole-resource versions deliberately have no implicit
merge operation.

Pass-local typed read/write handles are the bridge to recording. Execute
callbacks can resolve only their own declared handles, never arbitrary versions
or raw native resources. This preserves the access declaration as the sole
authority while leaving pipelines and descriptor layouts renderer-owned.

### Raster attachment contents

Raster attachments make beginning- and end-of-pass content behavior explicit:

- `LoadOp::Load` reads and preserves previously defined contents.
- `LoadOp::Clear(value)` defines the selected attachment range.
- `LoadOp::DontCare` neither preserves old contents nor proves new contents.
- `StoreOp::Store` preserves results that are already defined; it does not
  initialize them by itself.
- `StoreOp::Discard` makes the selected contents undefined after the pass.

These operations, the attachment range, and write coverage jointly drive
read-before-initialization validation. A clear command recorded later inside a
callback is a different operation and cannot retroactively change the declared
load/store contract. This is why attachment contents belong in graph setup
rather than being inferred from command recording.

## Retained recipes, not recorded work

Pass setup occurs while the graph is authored and returns downstream versions
plus retained pass data. Execute callbacks are repeatable `Fn` recipes invoked
only for an instantiated frame. A compile never invokes them. Retained recipes
are `Send + Sync + 'static`; local frame inputs may be thread-local so a
renderer can vary per-frame data without recompiling topology.

The setup closure returns `(O, D)`: `O` exposes successor versions to later
graph authoring, while `D` is retained for that pass's execute callback.
`instantiate_local` relaxes only the owned frame-data type `F`; retained data
and callbacks remain `Send + Sync + 'static`. This narrow rule keeps compiled
graphs shareable without pretending that arbitrary thread-bound renderer state
can be captured safely. A real backend can carry thread-bound objects in its
execution context instead.

This split allows one `CompiledGraph` snapshot to be reused, keeps acquired
images and encoders out of compiler state, and leaves future recording
parallelism as an implementation decision rather than a public promise. A
logical pass is not synonymous with one encoder, command buffer, queue,
recording job, or submission.

Dynamic frame inputs and a compatible replacement for an import binding do not
change graph topology and therefore do not require recompilation. Accesses,
roots, descriptors, or the capability model do. Reusing a `CompiledGraph`
does not skip per-frame binding validation, recording, submission, or
completion; a future recording/content cache needs an explicit compatibility
key and measured benefit.

## Imports, exports, and ownership boundary

Graph-created resources are transient logical declarations. External resources
are stable import slots with static descriptor, incoming state, ownership, and
initial-content contracts; concrete objects are supplied per frame. Exports are
roots with outgoing-state contracts. Imported leases survive until full-frame
completion, not merely submission. Each exported value reports that outgoing
state alongside its physical object and lease, so a graph-external consumer
such as readback must use the graph result as its actual incoming state.

Incoming state and incoming contents are intentionally separate. A resource in
a readable native state is not necessarily initialized, so ordinary imports
must state `InitialContents::Defined` or `Undefined`. During frame resolution,
the provider verifies the selected device, descriptor, initial state, and
stable physical identity. Distinct same-kind logical resources cannot resolve
to the same identity while the plan has no physical-alias model.

The compiled plan knows every live resource's required operation set, while
each `BoundTexture` or `BoundBuffer` reports the physical resource's actual
allowed operation set. Providers and backends derive that set from the resource
creation contract and known native descriptor, flags, format, view, and
allocation constraints; they must not echo the compiled requirement.

Frame resolution validates `required operations <= actual allowed operations`
for both imports and backend-created transients before opening an encoder. This
catches an insufficient imported resource and a faulty transient allocator at
the same domain boundary instead of deferring either failure to native command
validation.

The distinction prevents frame-local native state leaking into compiled graphs
and makes resource reuse/retirement explicit. Surface imports have a narrower
portable descriptor and ownership model. Compilation can validate surface
semantics, but acquisition and present execution are deliberately outside the
current executor.

The portable surface shape is a single-sampled, single-mip, single-layer 2D
texture. Only capability-approved color-attachment or copy-destination use,
followed by presentation, is currently legal. Sampling, storage, copy-source,
and depth/stencil surface use fail closed until the surface contract grows the
corresponding facts. Surface ownership and acquisition state stay with a future
surface adapter rather than ordinary graph imports.

## Roots, side effects, and explicit order

Exports and presentation targets are resource roots. `mark_side_effect` adds a
root only for a declared non-resource observable effect. `depends_on` requires
a diagnostic reason and is reserved for external protocol ordering that no
resource can express. It constrains the pass DAG but supplies no access,
transition, lifetime, or initialization fact.

This separation prevents an ordering edge from hiding a missing GPU access.
Root-driven liveness also means a dead reader does not become live merely
because a retained successor writer would require a write-after-read edge if
both passes were retained.

## Culling and immutable compilation

Exports, presentation targets, and declared side effects form roots. The
compiler retains their producer closure and culls unrelated passes. Culling is
observable through `CompileReport`, because it is a planning decision useful to
tools and users, not an exceptional condition.

`CompileReport` also exposes inferred resource dependencies, retained explicit
orders, retained side-effect reasons, and safe capability fallbacks. The
compiler never silently converts an unsupported semantic into a different one.
Structured `CompileError` categories cover stale or foreign versions,
read-before-initialization, writer branches, conflicting accesses, invalid
ranges, cycles, incomplete imports, unsupported requirements, and invalid
export/present roots. Panics and raw native errors are not substitutes for this
domain-level diagnostic contract.

Compilation produces an immutable `CompiledGraph`: retained recipes,
capability fingerprint, dependencies, roots, semantic transitions, and an
`ExecutionPlan`. Later edits to `RenderGraph` cannot modify the snapshot.
`FrameInputs` instantiates that topology with owned data and import bindings.

The plan also summarizes creation-time texture/buffer usage from retained
accesses plus import/export boundaries. It is calculated after culling, so dead
work cannot enlarge allocation requirements. Backends map this typed summary to
native allocation flags; they must preserve same-state barriers following an
overlapping write, which encode memory ordering rather than a state change.

Tracking the previous access mode as well as the state is necessary because
two overlapping writes can require a memory barrier even when their named state
is identical. Consecutive reads do not require that barrier, and disjoint ranges
are tracked independently. Overlapping same-pass reads that demand incompatible
states fail closed until a proven combined native read state exists.

## Submission, completion, and retirement

Presentation readiness and frame retirement are different events. A future
presentation dependency means a surface image is ready for presentation;
`FrameCompletion` means every terminal submission for the frame is complete and
executor-held imports and transients may retire. The current serial plan has
one submission, but the all-of completion meaning is preserved for later
multi-submit lowering.

Dropping a pending `FrameSubmission` does not block and does not release its
leases. It transfers them into an executor inbox; the next executor operation
hands them to backend retirement until completion is terminal. Caller-held
export leases remain caller-owned. Re-entrant executor operations report
`ExecutorBusy` instead of waiting while a backend lock is held. These rules
make safe native lifetime behavior part of the protocol without putting native
handles into graph state.

`CompletionStatus` distinguishes pending, complete, and an accepted
submission's structured terminal failure (`DeviceLost` or
`ExecutionFailed`). A backend may return submit `Err` only when it knows no GPU
work was accepted. Accepted-unknown native errors must instead produce a
completion and retain or quarantine every referenced object; they cannot be
treated as an unsubmitted command buffer.

## Capabilities: facts, not a feature wishlist

`DeviceCapabilities` contains observed facts that compilation needs: logical
queues, recording and synchronization model, formats, limits, buffer support,
transient-resource facts, and optional surface facts. It is demand-driven and
non-exhaustive: new vocabulary is added only when a graph semantic or validated
fixture requires it. Missing facts fail closed.

Format entries describe sampling/filtering, storage, attachment/sample-count,
and copy support. Buffer entries describe storage and indirect support; limits
cover only values consumed by current validation. In 0.1.3 that includes the
per-dimension compute-dispatch maximum: zero and over-limit dimensions are
rejected before a backend records a dispatch. Fixed shader workgroup shape and
invocation limits remain RHI hardware facts because the graph has no general
shader declaration surface. Surface facts are optional so headless compilation
never fabricates a presentation target. A surface copy fact does not imply
arbitrary conversion, scaling, or format reinterpretation.

The graph selects one logical queue for the current execution baseline. Queue
labels do not claim hardware parallelism; multi-queue scheduling is a future
optimization. This keeps the contract portable across immediate contexts,
deferred command buffers, and future explicit APIs without fabricating support.

## Execution baseline and evolution

For each frame, `FrameExecutor` checks the capability fingerprint, resolves live
imports, allocates transients from the compiled usage summary, emits semantic
transitions, invokes retained callbacks in deterministic plan order, finishes
one encoder, and makes one submission through `ExecutionBackend`.
`ExecutionError` keeps frame-binding, recording, capability,
unsupported-feature, and backend failures separate.

`TestRhi` is the deterministic CPU reference implementation. It records a
structured trace, supports injected failures and manually advanced completion,
and validates protocol order and retirement. It does not execute shaders,
emulate GPU memory, or prove native validation or GPU performance.

The current boundary is intentionally narrow and provisional. A native backend
must truthfully report actual allowed usage for imports and allocations, then
turn planned transitions and commands into DX12/Vulkan behavior.
`fluxel-rhi` currently proves that path for Copy and a closed fixed-Compute
subset: `CopyBackend` remains Copy-only, while `ComputeBackend` adds only fixed
kernel dispatch with opaque RHI-provided pipeline and binding objects. The
graph neither accepts WGSL nor owns Naga/native shader lowering, descriptor
layouts, or pipeline policy. Those objects are registered by a provider and
must still correspond to explicit pass resource accesses.

Native Copy conformance covers partial buffers, padded texture rows, and
overlapping same-state WAW; buffer-copy offsets and sizes are rejected unless
4-byte aligned before reaching the native boundary. Fixed Compute conformance
covers K01's in-place wrapping add and K02's ordered add-then-multiply on one
RW storage buffer, each read back against a CPU oracle on DX12 and Vulkan.
Readback consumes the export's reported outgoing state as its actual incoming
state; it is not allowed to repair a wrong graph result.

Surface presentation, multi-queue lowering, aliasing, and performance claims
follow only after real conformance evidence exists.

## Asset and renderer boundary

Assets own content identity, loading, decoding, streaming, hot reload, cache,
and cross-frame lifetime. The graph owns in-frame accesses, versions,
dependencies, transients, and execution planning. Renderer/frame-coordinator
code resolves an asset into a GPU-ready snapshot and binds it to an import slot.

Accordingly, the graph does not accept an asset handle or learn cache/source
state, and the asset system does not choose barriers, queues, or pass order.
Pipelines, samplers, layouts, descriptor allocation, and dynamic binding policy
also remain renderer/RHI-owned. `BindingSetId` and `RenderObjectProvider` form
an opaque validation bridge instead of duplicating a descriptor framework in
the graph.

Opacity does not hide resource facts. Every texture or buffer referenced by a
material, pipeline, or binding recipe must also appear in that pass's explicit
`ResourceAccess` manifest with its range and semantic use. A binding ID cannot
grant undeclared access or substitute for a graph dependency.

## Stable and provisional surfaces

The stable semantic center is declaration: versions, access/range semantics,
attachment contents, roots, capability validation, and structured compile
diagnostics. The execution SPI—providers, `ExecutionBackend`, executor, and
submission/export handoff—remains provisional while native execution exposes
integration requirements. Neither surface leaks native backend handle types.

The present evidence is compiler/validation coverage, TestRhi protocol
coverage, and real headless DX12/Vulkan Copy plus fixed-Compute readback.
Raster readback is the next evidence level; surface acquire/resize/recreate/
present follows that.
Native multi-queue, aliasing, recording caches, and performance claims remain
deferred until representative fixtures and measurements exist.

## Related documents

- [RHI architecture](design-rhi.md)
- [RenderGraph crate user guide](../crates/rendergraph/README.md)
- [Project overview](../README.md)
