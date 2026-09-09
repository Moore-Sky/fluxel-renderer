# fluxel-rhi

`fluxel-rhi` is Fluxel's native-device bootstrap crate. It opens one headless
Direct3D 12 or Vulkan device on Windows, owns it safely, and reports adapter
facts for a later renderer.

It is deliberately a narrow A1 milestone, not a general graphics API. It does
**not** allocate resources, record commands, submit work, acquire or present a
surface, compile shaders, or implement
`fluxel_rendergraph::ExecutionBackend`. The render-graph crate's CPU-only
`TestRhi` remains the executable backend used by the numbered graph examples.

## Installation

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.1.0"
```

The crate is not published on crates.io yet, so the tagged Git dependency is
the current installation path. `v0.1.0` begins this workspace's independent
release sequence from a pre-migration source snapshot. The default feature set
enables both Windows backends.
To select one explicitly:

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.1.0"
default-features = false
features = ["dx12"]
```

| Feature | Effect on Windows |
| --- | --- |
| `dx12` | Compile Direct3D 12 device bootstrap. |
| `vulkan` | Compile Vulkan device bootstrap. |

Both features are enabled by default. A build with a requested backend omitted
returns `OpenError::BackendDisabled`; a non-Windows build returns
`OpenError::PlatformUnsupported`. Adding the dependency opens no device. Rust
1.87 and edition 2024 are required.

## Open a device

Select a backend explicitly, open adapter zero, then inspect the native facts:

```rust,no_run
use fluxel_rhi::{Backend, Device, DeviceOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::open(Backend::Dx12, DeviceOptions::default())?;
    println!("{:#?}", device.hardware());
    println!("{:#?}", device.capabilities());
    Ok(())
}
```

On a Windows machine with a suitable driver, run the complete selectable
example from the workspace root:

```powershell
cargo run -p fluxel-rhi --example 01_open_device -- dx12
cargo run -p fluxel-rhi --example 01_open_device -- vulkan
```

`Device` retains the queue, device, adapter, and instance for its lifetime.
Dropping it closes the headless native device. There are intentionally no raw
native handles to extract and no submission methods to call.

## Core concepts

`Backend` is an explicit choice between `Dx12` and `Vulkan`; Fluxel never
silently falls back. `DeviceOptions::adapter_index` selects the backend-native
enumeration index, with zero as the default. It is a bootstrap/probing
mechanism, not yet a polished cross-platform adapter-selection API.

`Device::hardware()` returns unmodified driver identity: backend, name,
vendor/device IDs, device kind, PCI bus ID where available, and driver strings.
`Device::capabilities()` returns raw adapter limits currently needed by the
bootstrap contract. These facts are not a portability profile, feature tier,
or promise that a particular Fluxel graph can execute; a future renderer makes
those policy decisions.

`OpenError` distinguishes an omitted feature, unsupported platform, absent
adapter, unavailable validation facility, insufficient baseline limits, and
native loader/driver failure. Include its display text in diagnostics, but do
not parse it as a stable native error code.

## Native validation

The default `Validation::Disabled` does not request native validation. During
development, request fail-closed validation:

```rust,no_run
use fluxel_rhi::{Backend, Device, DeviceOptions, Validation};

let device = Device::open(
    Backend::Vulkan,
    DeviceOptions {
        validation: Validation::Required,
        ..DeviceOptions::default()
    },
)?;
# let _: fluxel_rhi::Device = device;
# Ok::<(), fluxel_rhi::OpenError>(())
```

For DX12, `Required` needs the D3D12 debug layer and verifies that the opened
device exposes its information queue. For Vulkan, it needs
`VK_LAYER_KHRONOS_validation` and the layer's
`VK_EXT_validation_features`; the internal HAL request enables synchronization
validation when that facility is available. If Fluxel cannot positively
establish the requested facility, opening fails with
`OpenError::ValidationUnavailable` rather than continuing with a weaker
configuration.

Validation only covers the device-open path today. It cannot validate resource
states or command synchronization until those operations exist in this crate.

## Platform compatibility

| Platform | DX12 | Vulkan |
| --- | --- | --- |
| Windows | Supported when its feature, loader, driver, and selected adapter are available | Supported when its feature, loader, driver, and selected adapter are available |
| macOS, Linux, other targets | Returns `PlatformUnsupported` | Returns `PlatformUnsupported` |

Headless means no window or surface is created. It does not imply that a
surface, swapchain, or presentation path is supported.

## Performance and scope

Opening a native device is setup work, not a rendering benchmark. This crate
makes no frame-time, throughput, allocation, synchronization, or
GPU-performance claim. It keeps native objects in a private backend module to
establish a safe ownership boundary for later vertical slices.

The current scope ends after successful device bootstrap and fact reporting.
Native GPU conformance needs real resource allocation with creation usage,
transition/barrier emission, command recording, submission/completion, and
readback first.

## Testing and development

Ordinary tests verify API behavior and native-backend compile/link coverage.
Tests that actually open hardware are ignored because they depend on the local
driver and validation installation. Run them deliberately on a configured
Windows development machine:

```powershell
cargo +1.87.0 test -p fluxel-rhi --all-features -- --ignored --nocapture
```

The DX12 required-validation case needs the D3D12 debug layer. The Vulkan
required-validation case needs LunarG's Khronos validation layer and its
validation-features extension. CI compiling the crate is not evidence that a
GitHub-hosted runner opened a real GPU device.

See the
[RHI architecture design](https://github.com/Moore-Sky/fluxel-renderer/blob/main/documents/design-rhi.md)
for ownership, validation, and evolution decisions. See the
[`fluxel-rendergraph` user guide](https://github.com/Moore-Sky/fluxel-renderer/tree/main/crates/rendergraph)
for graph authoring and its numbered examples.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
