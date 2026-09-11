# Windows DX12 presentation harness

This is a deliberately narrow Stage 1 proof harness. `fluxel-host` owns the
fixed-size Win32 `Window` primitive and its message pump; this example joins it
to the renderer-owned DX12 surface API. It introduces no input, clock, general
host runtime, or renderer → host dependency: both crates meet only at the
standard raw window/display handle traits.

Each frame follows exactly one path: acquire a surface image, call the existing
fixed-frame renderer's `draw_to_surface`, poll its submission to completion,
then acquire the next image. The harness never assembles a graph or pipeline,
and it has no headless fallback: headless success is not presentation evidence.

```powershell
cargo run --manifest-path examples/windows-dx12/Cargo.toml -- --frames 120 --repeat 2
```

`--frames K` makes a finite evidence run; omitting it runs until `WM_CLOSE`.
`--repeat R` recreates the complete window/device/surface lifecycle R times.
Shutdown is intentionally ordered: stop frames, complete the live submission,
drop renderer-owned resources, unconfigure the surface (which drains accepted
work), drop the surface's retained window reference, then explicitly close and
drop the host window. The manifest enables RHI's conformance-only
`test-support` feature solely to capture DX12 debug-layer diagnostics; any
diagnostic makes a run fail.
