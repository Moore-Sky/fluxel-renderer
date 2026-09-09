# Fluxel Renderer

`fluxel-renderer` defines the user-facing scene inputs `Camera`, `Geometry`,
`Mesh`, `BasicMaterial`, and insertion-ordered `DrawList`. Its optional
`gpu-upload` feature also turns validated indexed geometry into an opaque,
immutable GPU-ready snapshot without blocking the frame-building thread.
The same feature can consume a ready snapshot, a `Camera`, and a
`BasicMaterial` in one deliberately closed headless `f32x3/u32` indexed draw
and return opaque offscreen image metadata. It can also upload one immutable
tight `Rgba8Unorm` image and use its opaque snapshot in a closed textured draw.

It is intentionally not a complete GPU renderer: it does not lower draw lists,
compile general material shaders, provide samplers/PBR or a general
pipeline/bind-group API, or present pixels.

## Optional GPU upload

```toml
[dependencies.fluxel-renderer]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.2.4"
features = ["gpu-upload"]
```

`IndexedMeshUpload::begin` serializes positions as tightly packed
little-endian `f32x3` and indices as little-endian `u32`. Polling is
non-blocking. A cloneable `IndexedMeshSnapshot` appears only after both native
submissions complete; failure and partial acceptance never expose a half-ready
generation. Buffers, resource states, mapping, and native handles remain
renderer-private.

`FixedFrameRenderer::draw` accepts a ready snapshot, `Camera`, `BasicMaterial`,
and a nonzero extent. It serializes `projection * view` and the linear
`base_color` into one immutable 80-byte uniform, starts its upload without
blocking, then submits one fixed triangle-list Raster recipe only after that
upload completes. Poll the returned `FixedFrameSubmission`: `Pending` covers
either uniform upload or Raster completion, while retryable `Busy` retains the
operation. `Complete` yields an opaque `FrameImage`; terminal or otherwise
unproven accepted Raster work poisons that snapshot generation. Snapshot clones
share the same single-in-flight state; no native texture or buffer handle is
exposed.

`Rgba8Image` validates a nonzero two-dimensional tight RGBA8 payload.
`BaseColorTextureUpload::begin` borrows it and starts a non-blocking immutable
upload; only proven completion publishes `BaseColorTextureSnapshot`.
`TexturedBasicMaterial` couples that snapshot to `BasicMaterial`, and
`FixedFrameRenderer::draw_textured` uses a fixed mip-zero `textureLoad` mapping
derived from model-space `position.xy`. `TexturedGeometry` and
`TexturedIndexedMeshUpload` add an isolated three-stream position/index/UV
generation; `draw_textured_uv` consumes its explicit finite `f32x2` coordinates
through a closed second vertex slot. Both paths deliberately expose no sampler,
filter/wrap/LOD, sRGB, or PBR contract. The textured paths
accepts only finite, positive-w triangles wholly inside the clip volume, and
mesh/texture generations share one atomic draw reservation outcome.

## Use today

Build a mesh and schedule it for a camera:

```rust
use fluxel_renderer::{BasicMaterial, Camera, DrawList, Geometry, Mesh};

let camera = Camera::default();
let geometry = Geometry::from_positions(vec![
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.0, 0.5, 0.0],
])
.with_indices(vec![0, 1, 2])?;
let mesh = Mesh::new(geometry, BasicMaterial::new([0.2, 0.7, 1.0, 1.0]));

let mut draws = DrawList::new(&camera);
draws.push(&mesh);
assert_eq!(draws.len(), 1);
# Ok::<(), fluxel_renderer::GeometryError>(())
```

`Geometry::with_indices` checks indices when the geometry is constructed.
`DrawList` preserves insertion order so a future renderer has an explicit,
inspectable submission input without adding sorting policy prematurely.

## Performance and compatibility

With default features the crate has no dependencies, no `unsafe`, and no native
API calls, and builds wherever Rust 1.87 supports the workspace. `gpu-upload`
adds the safe `fluxel-rhi` boundary; native execution is currently implemented
on Windows DX12/Vulkan. The 0.2.2 U03 fixtures compare the Camera/material
uniform fixed offscreen draw byte-for-byte with CPU pixel oracles on both
backends under Required native validation. The renderer crate itself contains
no `unsafe`.

The 0.2.3 U04 fixtures compare this textured draw byte-for-byte with an
independent CPU oracle on DX12 and Vulkan under Required validation.
The 0.2.4 U05 fixtures separately prove explicit UV perspective interpolation
against position-derived and linear CPU counter-oracles on both backends.

The internal `shader` module is an ownership boundary for material shader
modules. Shader compilation and reflection are deliberately future backend
work.

The [renderer design](https://github.com/Moore-Sky/fluxel-renderer/blob/main/documents/design-renderer.md)
explains the ownership split and the evidence gates for future GPU integration.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
