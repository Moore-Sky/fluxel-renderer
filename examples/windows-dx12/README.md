# Windows DX12 presentation harness

This is a deliberately narrow Stage 1 proof harness. `fluxel-host` owns the
Win32 `Window` primitive and its ordered lifecycle events; this example joins it
to the renderer-owned DX12 surface API. It introduces no input, clock, general
host runtime, or renderer → host dependency: both crates meet only at the
standard raw window/display handle traits.

The harness privately owns a three-slot bounded ring. It reserves a slot before
acquiring a surface image, advances the new renderer transaction until raster
submission is accepted and the image enters the native presentation path, then
admits another frame. A slot remains unavailable until its own completion is
known; when all three are live, the harness records back pressure and polls
until at least one exact slot retires. The count, slots, fence values, and image
indices are not renderer API. The harness never assembles a graph or pipeline,
and it has no headless fallback: headless success is not presentation evidence.

Resize, minimize, zero-size, and restore events drive the RHI-owned surface
state. Each non-zero configuration has an opaque generation; the old
generation stops acquisition and reaches known completion before its native
resources are released. Suspended intervals never acquire or submit a frame.

```powershell
cargo run --manifest-path examples/windows-dx12/Cargo.toml -- --frames 120 --repeat 2 --induce-back-pressure
```

`--frames K` makes a finite evidence run; omitting it runs until `WM_CLOSE`.
`--repeat R` recreates the complete window/device/surface lifecycle R times.
`--induce-back-pressure` attaches an independent conformance-only latch to each
of the first three successful presentation completions. After all three slots
are live, the fourth start is rejected before native acquire/record; the harness
then releases one exact latch, proves only that slot retires, and proves the next
frame reuses that same slot. The other two completions remain held during that
observation, so driver timing cannot accidentally satisfy the oracle.
`--verify-accepted-unknown` is a separate fault-process oracle: after a real
successful submit/present it injects unknown completion, then requires slot
reuse, resize, and shutdown to remain refused and verifies Surface drop retains
the window owner. This mode intentionally exits the process with quarantined
ownership and must not be combined with normal clean-shutdown evidence.
Shutdown is intentionally ordered: stop admission, drain every live submission,
drop renderer-owned resources, unconfigure the surface (which drains accepted
work), drop the surface's retained window reference, then explicitly close and
drop the host window. The manifest enables RHI's conformance-only
`test-support` feature solely to capture DX12 debug-layer diagnostics; any
diagnostic makes a run fail.
