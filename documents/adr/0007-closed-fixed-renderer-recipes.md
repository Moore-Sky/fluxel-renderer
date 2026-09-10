# ADR-0007: Keep fixed renderer recipes closed

**Status:** Accepted

## Context

The fixed renderer slices revealed real variation in vertex streams, texture
domain, sampling, bindings, reservations, exports, and RHI kernels. Repeated
`draw_*` branching risks letting those facts drift apart.

## Decision

Use a private, closed `RasterRecipe` to represent exactly the seven validated
renderer combinations. A recipe atomically selects graph declaration, provider
registration, snapshot/texture domain, reservation topology, exports, and the
existing closed RHI kernel. It has no public constructor or arbitrary
composition API.

## Alternatives

- Continue duplicating each `draw_*` implementation.
- Introduce public general material, pipeline, or render-packet abstractions.

## Consequences

Internal implementation can share a stable decision point while public APIs
and GPU semantics remain unchanged. Public admission and terminal failures keep
their graph, pipeline, provider, upload, and execution causes as structured
errors rather than stringifying them. DrawList lowering, handles, transforms,
PBR, and general material policy remain future, separately justified work.

## Evidence

The legacy fixed-color suite establishes the baseline contract; camera/material,
texture-load, UV texture-load, linear-clamp, sRGB, Lambert, and vertex-color
suites establish the seven `RasterRecipe` combinations. v0.7.0 records their
mapping and rollback invariants without promoting them to general material or
pipeline APIs.

See [Renderer design](../design-renderer.md) for the current fixed-renderer
shape.
