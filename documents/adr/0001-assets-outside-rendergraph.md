# ADR-0001: Keep assets and renderer policy outside RenderGraph

**Status:** Accepted

## Context

Frame planning needs resource access, dependency, version, and transition
semantics. Asset loading, cache identity, readiness, scene policy, and native
object ownership evolve on different timescales.

## Decision

RenderGraph owns portable in-frame declarations and immutable execution plans.
Renderer owns scene-to-frame policy and GPU-ready snapshot selection. Assets
own persistent identity, loading, caching, and hot reload. RHI owns native
resources and execution.

## Alternatives

- Put asset handles, cache state, or native handles in graph declarations.
- Make RHI a second renderer that selects materials or scene policy.

## Consequences

Imports bind renderer-selected physical snapshots with explicit incoming state;
the graph never discovers assets or guesses native state. DrawList lowering and
asset handles remain separate future work.

## Evidence

0.1.0 established the crate boundary. 0.2.0–0.2.4 published opaque immutable
renderer snapshots and imported them privately into graphs without exposing
native handles or asset policy.

See [RenderGraph architecture](../design-rendergraph.md) and
[Renderer design](../design-renderer.md) for the current boundary.
