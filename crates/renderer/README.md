# Fluxel Renderer

`fluxel-renderer` is the headless renderer domain layer of the Fluxel
workspace. It currently defines the user-facing scene inputs that a future
renderer will lower into a render graph: `Camera`, `Geometry`, `Mesh`,
`BasicMaterial`, and an insertion-ordered `DrawList`.

It is intentionally not a GPU renderer yet: it does not allocate native
resources, compile or reflect shader source, record commands, submit work, or
present pixels.

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

The crate has no dependencies, no `unsafe`, and no native API calls. Its work
is ordinary CPU-side allocation for caller-provided geometry and draw-list
items. It therefore builds wherever Rust 1.87 supports the workspace; it makes
no GPU-performance or platform-rendering claim.

The internal `shader` module is an ownership boundary for material shader
modules. Shader compilation and reflection are deliberately future backend
work.

The [renderer design](https://github.com/Moore-Sky/fluxel-renderer/blob/main/documents/design-renderer.md)
explains the ownership split and the evidence gates for future GPU integration.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
