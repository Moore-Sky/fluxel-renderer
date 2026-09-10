# ADR-0006: Do not expose a general pipeline abstraction yet

**Status:** Accepted

## Context

The renderer has only evidence-backed fixed shader, layout, texture, sampler,
and binding combinations. A generic pipeline API would guess at unvalidated
ownership, reflection, layout, and portability requirements.

## Decision

Represent each proven native combination as a closed RHI artifact and binding
recipe. Renderer-shaped Raster artifacts are exposed only under
`fluxel_rhi::experimental::fixed_artifacts`; keep shaders, descriptors,
samplers, and native objects opaque. Add a new closed recipe only for a
separately planned vertical slice.

## Alternatives

- Expose arbitrary WGSL, descriptors, layouts, and pipeline builders now.
- Encode every new combination as unchecked optional fields in one recipe.

## Consequences

The stable RHI surface does not imply that fixed renderer recipes are a lasting
general contract. General shader/material/pipeline policy waits for real
renderer requirements; experimental recipe changes may evolve with the current
vertical slice.

## Evidence

0.1.3 fixed compute and 0.1.4 fixed raster established closed artifacts.
The v0.7.0 workspace release contains discrete indexed, uniform, texture-load,
UV, sampler, sRGB, Lambert, and vertex-color recipes without broadening them
into a general API.

See [RHI design](../design-rhi.md) and [Renderer design](../design-renderer.md)
for the currently supported recipes.
