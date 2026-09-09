# fluxel-rhi

`fluxel-rhi` opens one headless Direct3D 12 or Vulkan device on Windows and
creates opaque owned buffers and 2D textures through a safe portable contract.
Its Copy and fixed-artifact Compute backends execute the same compiled
RenderGraph plan on both APIs, including semantic transition lowering, one
queue submission, completion, and retirement.

It is deliberately a narrow 0.1.3 milestone, not a general graphics API.
`CopyBackend` implements only Copy commands; the distinct `ComputeBackend`
adds only closed `ComputeKernel` artifacts and their single RW storage-buffer
binding recipe. It does **not** expose general mapping/readback, acquire or
present a surface, a general shader API, raster drawing, multiple queues, or
transient aliasing.

## Installation

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.1.3"
```

The crate is not published on crates.io yet, so the tagged Git dependency is
the current installation path. `v0.1.0` begins this workspace's independent
release sequence from a pre-migration source snapshot. The default feature set
enables both Windows backends.
To select one explicitly:

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/Moore-Sky/fluxel-renderer"
tag = "v0.1.3"
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

`Device::create_buffer` and `Device::create_texture` validate RenderGraph's
typed descriptors and usage sets before entering the native boundary. Returned
resources expose only descriptor, actual allowed usage, opaque identity and a
cloneable lease. The final resource or lease destroys the native object once;
it also retains the owning device. Requested usage is a minimum: native
normalization may report a wider allowed set, such as buffer `StorageWrite`
becoming storage read/write. There are no raw handles.

`CopyBackend` implements the execution SPI with one logical queue and one
command buffer per graph execution. It permanently rejects compute and raster
families. `ComputeBackend` separately enables Copy plus the fixed Compute
subset. Embedded WGSL is parsed and validated as Naga IR, then lowered by the
HAL for DX12 or Vulkan; Fluxel's portable identity is source hash, entry point,
workgroup shape, and binding-recipe version, never a native-binary hash.
Pipelines and bindings are opaque device-affine values with cloneable leases;
accepted submissions retain every referenced lease until retirement is safe.

## Core concepts

`Backend` is an explicit choice between `Dx12` and `Vulkan`; Fluxel never
silently falls back. `DeviceOptions::adapter_index` selects the backend-native
enumeration index, with zero as the default. It is a bootstrap/probing
mechanism, not yet a polished cross-platform adapter-selection API.

`Device::hardware()` returns unmodified driver identity: backend, name,
vendor/device IDs, device kind, PCI bus ID where available, and driver strings.
`Device::capabilities()` returns raw adapter limits currently needed by the
bootstrap contract, including fixed-Compute dispatch, workgroup-dimension, and
total-invocation bounds. These facts are not a portability profile, feature
tier, or promise that a particular Fluxel graph can execute; a future renderer
makes those policy decisions. Zero or excessive dispatch dimensions fail before
recording; the fixed shader workgroup is checked again before native creation.

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

The conformance fixtures collect DX12 information-queue and Vulkan validation
diagnostics programmatically while checking command states and synchronization.
C01/C02/C03 cover Copy and K01/K02 cover the fixed Compute artifacts, each
against a CPU oracle on both native backends.

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

The current scope ends after Copy plus fixed-Compute transition/order lowering,
command recording, one submission, completion, lease retirement, and
crate-private test readback. A readback consumes the export's reported outgoing
state as its real incoming state; it cannot silently repair a graph-state bug.
Buffer Copies require 4-byte-aligned source offset, destination offset, and
size, checked in both graph recording and RHI lowering. Queue operations are
serialized per opened device, including error-path idle waits. Raster remains a
later vertical slice.

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
