# ADR-0006: Do not expose a general pipeline abstraction yet

**Status:** Accepted

## Context

The renderer has only evidence-backed fixed shader, layout, texture, sampler,
and binding combinations. A generic pipeline API would guess at unvalidated
ownership, reflection, layout, and portability requirements.

## Decision

Represent each proven native combination as a closed RHI artifact and binding
recipe. Keep shaders, descriptors, samplers, and native objects opaque; add a
new closed recipe only for a separately planned vertical slice.

## Alternatives

- Expose arbitrary WGSL, descriptors, layouts, and pipeline builders now.
- Encode every new combination as unchecked optional fields in one recipe.

## Consequences

Public surface area stays minimal and semver changes are explicit when an
exhaustive closed enum grows. General shader/material/pipeline policy waits for
real renderer requirements.

## Evidence

0.1.3 fixed compute and 0.1.4 fixed raster established closed artifacts.
0.2.1–0.2.7 added discrete indexed, uniform, texture-load, UV, sampler, sRGB,
and Lambert recipes without broadening them into a general API.

See [RHI design](../design-rhi.md) and [Renderer design](../design-renderer.md)
for the currently supported recipes.
