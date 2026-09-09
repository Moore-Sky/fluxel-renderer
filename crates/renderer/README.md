# Fluxel Renderer

`fluxel-renderer` defines the user-facing scene inputs `Camera`, `Geometry`,
`Mesh`, `BasicMaterial`, and insertion-ordered `DrawList`. Its optional
`gpu-upload` feature also turns validated indexed geometry into an opaque,
immutable GPU-ready snapshot without blocking the frame-building thread.
The same feature can consume a ready snapshot in one deliberately closed,
headless `f32x3/u32` indexed draw and return opaque offscreen image metadata.

It is intentionally not a complete GPU renderer: it does not lower draw lists,
compile general material shaders, consume camera/material data in the fixed
draw, or present pixels.

## Optional GPU upload

```toml
[dependencies.fluxel-renderer]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.2.1"
features = ["gpu-upload"]
```

`IndexedMeshUpload::begin` serializes positions as tightly packed
little-endian `f32x3` and indices as little-endian `u32`. Polling is
non-blocking. A cloneable `IndexedMeshSnapshot` appears only after both native
submissions complete; failure and partial acceptance never expose a half-ready
generation. Buffers, resource states, mapping, and native handles remain
renderer-private.

`FixedFrameRenderer::draw` accepts only a ready snapshot and a nonzero extent.
It submits one fixed opaque-color triangle-list recipe without blocking. Poll
the returned `FixedFrameSubmission`: `Pending` and retryable `Busy` retain the
operation, `Complete` yields an opaque `FrameImage`, and terminal or otherwise
unproven accepted work poisons that snapshot generation. Snapshot clones share
the same single-in-flight state; no native texture or buffer handle is exposed.

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
on Windows DX12/Vulkan. The 0.2.1 U02 fixture compares the fixed offscreen draw
byte-for-byte with a CPU pixel oracle on both backends under Required native
validation. The renderer crate itself contains no `unsafe`.

The internal `shader` module is an ownership boundary for material shader
modules. Shader compilation and reflection are deliberately future backend
work.

The [renderer design](https://github.com/Moore-Sky/fluxel-renderer/blob/main/documents/design-renderer.md)
explains the ownership split and the evidence gates for future GPU integration.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
