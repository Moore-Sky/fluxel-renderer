//! Native GPU ownership plus Copy and fixed-artifact Compute execution for Fluxel.
//!
//! This crate exposes no `wgpu-hal` types. It opens one headless DX12 or Vulkan
//! device, owns buffers and 2D textures, and executes Copy plus a deliberately
//! fixed Compute subset of a portable RenderGraph plan through barriers, one
//! submission, completion, and lease-backed retirement. Raster, surfaces, and
//! presentation remain outside this milestone.

#![deny(missing_docs)]

use core::fmt;
use std::sync::Arc;

mod execution;
mod resource;

pub use execution::*;
pub use resource::*;

/// A native graphics API supported by Fluxel's Windows RHI.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Microsoft Direct3D 12.
    Dx12,
    /// Khronos Vulkan.
    Vulkan,
}

/// The required validation behavior when opening a device.
///
/// `Required` is fail-closed. Because the HAL has no portable proof
/// of validation, private backend-specific probes verify the facility before a
/// device is returned.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Validation {
    /// Do not request native validation.
    #[default]
    Disabled,
    /// Require a verifiably enabled native validation facility. Vulkan also
    /// requires the validation-features extension used for synchronization
    /// validation.
    Required,
}

/// Choices that affect opening a headless device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct DeviceOptions {
    /// Backend-native index of the adapter to open; default is zero.
    pub adapter_index: usize,
    /// Native validation policy.
    pub validation: Validation,
}

/// The broad kind of physical device reported by the native driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// The driver did not classify the device more precisely.
    Other,
    /// An integrated GPU.
    Integrated,
    /// A discrete GPU.
    Discrete,
    /// A virtual GPU.
    Virtual,
    /// A CPU or software implementation.
    Cpu,
}

/// Hardware identity reported by the selected native backend.
///
/// These are driver facts, not a conformance profile. Fluxel performs any
/// profile lowering separately and never masks this value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct HardwareInfo {
    /// The selected backend.
    pub backend: Backend,
    /// Driver-reported adapter name.
    pub name: String,
    /// Backend-specific vendor identifier.
    pub vendor_id: u32,
    /// Backend-specific device identifier.
    pub device_id: u32,
    /// Broad device classification.
    pub kind: DeviceKind,
    /// Backend-specific PCI bus identifier, if reported.
    pub pci_bus_id: String,
    /// Driver name reported by the backend.
    pub driver: String,
    /// Additional driver information reported by the backend.
    pub driver_info: String,
}

/// Raw capability facts from the selected native adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct HardwareCapabilities {
    /// Maximum two-dimensional texture extent reported by the backend.
    pub max_texture_dimension_2d: u32,
    /// Maximum bind groups reported by the backend.
    pub max_bind_groups: u32,
    /// Minimum uniform-buffer dynamic-offset alignment.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Minimum storage-buffer dynamic-offset alignment.
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum byte size of a single storage-buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum workgroup count for each dimension of a compute dispatch.
    ///
    /// A zero component means compute dispatches are unavailable and is
    /// intentionally fail-closed by [`ComputeBackend`].
    pub max_compute_workgroups_per_dimension: [u32; 3],
    /// Maximum local workgroup size for each dimension of a compute shader.
    ///
    /// A zero component means the fixed compute artifacts are unavailable and
    /// is rejected before the private native shader boundary is entered.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum total invocations in one compute shader workgroup.
    ///
    /// Zero is treated as unavailable and rejected before native shader
    /// creation.
    pub max_compute_invocations_per_workgroup: u32,
}

/// A headless native device that owns resources and copy-queue execution.
pub struct Device {
    #[allow(
        dead_code,
        reason = "native ownership and the queue are retained for safe shutdown and later slices"
    )]
    inner: Arc<imp::OpenedDevice>,
    identity: fluxel_rendergraph::DeviceIdentity,
    hardware: HardwareInfo,
    capabilities: HardwareCapabilities,
}

impl Clone for Device {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            identity: self.identity,
            hardware: self.hardware.clone(),
            capabilities: self.capabilities,
        }
    }
}

impl Device {
    /// Opens one explicitly selected backend and adapter for headless use.
    ///
    /// This method enters the private HAL boundary, creates no surface, and
    /// exposes no native handles. Copy submission and fixed-artifact compute
    /// execution are available through [`CopyBackend`] and [`ComputeBackend`].
    pub fn open(backend: Backend, options: DeviceOptions) -> Result<Self, OpenError> {
        let opened = imp::open(backend, options)?;
        Ok(Self {
            hardware: opened.hardware.clone(),
            capabilities: opened.capabilities,
            inner: Arc::new(opened),
            identity: fluxel_rendergraph::DeviceIdentity::new(next_identity()),
        })
    }

    /// Returns unmodified identity facts for the selected adapter.
    pub fn hardware(&self) -> &HardwareInfo {
        &self.hardware
    }

    /// Returns unmodified capability facts for the selected adapter.
    pub fn capabilities(&self) -> HardwareCapabilities {
        self.capabilities
    }

    /// Returns the identity used to reject resources from another device.
    pub fn identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.identity
    }
}

fn next_identity() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl fmt::Debug for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Device")
            .field("hardware", &self.hardware)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// Why opening a requested native device was not possible.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OpenError {
    /// This crate currently implements native opening only on Windows.
    PlatformUnsupported {
        /// The requested backend.
        backend: Backend,
    },
    /// The selected backend was not compiled into this build.
    BackendDisabled {
        /// The requested backend.
        backend: Backend,
    },
    /// The requested adapter index was not returned by the native driver.
    AdapterUnavailable {
        /// The requested backend.
        backend: Backend,
        /// The requested index.
        adapter_index: usize,
        /// Number of adapters found.
        available_adapters: usize,
    },
    /// The adapter cannot meet the baseline limits used to open a device.
    RequiredLimitsUnavailable {
        /// The requested backend.
        backend: Backend,
    },
    /// The requested validation mode cannot be positively verified.
    ValidationUnavailable {
        /// The requested backend.
        backend: Backend,
    },
    /// The native loader, driver, or device open operation failed.
    NativeUnavailable {
        /// The requested backend.
        backend: Backend,
        /// A diagnostic captured at the FFI boundary.
        reason: String,
    },
}

impl fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformUnsupported { backend } => {
                write!(formatter, "{backend:?} is only supported on Windows")
            }
            Self::BackendDisabled { backend } => write!(formatter, "{backend:?} is disabled"),
            Self::AdapterUnavailable {
                backend,
                adapter_index,
                available_adapters,
            } => write!(
                formatter,
                "{backend:?} adapter {adapter_index} is unavailable ({available_adapters} adapters found)"
            ),
            Self::RequiredLimitsUnavailable { backend } => {
                write!(
                    formatter,
                    "{backend:?} cannot meet the baseline device limits"
                )
            }
            Self::ValidationUnavailable { backend } => write!(
                formatter,
                "{backend:?} validation cannot be positively verified during bootstrap"
            ),
            Self::NativeUnavailable { backend, reason } => {
                write!(formatter, "{backend:?} is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for OpenError {}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::{Backend, Device, DeviceOptions, Validation};

    #[cfg(not(windows))]
    #[test]
    fn native_backends_are_rejected_off_windows() {
        use super::{Backend, Device, DeviceOptions, OpenError};

        assert!(matches!(
            Device::open(Backend::Vulkan, DeviceOptions::default()),
            Err(OpenError::PlatformUnsupported {
                backend: Backend::Vulkan
            })
        ));
    }

    #[cfg(all(windows, feature = "dx12", feature = "vulkan"))]
    #[test]
    #[ignore = "requires a locally installed native GPU driver"]
    fn opens_both_real_headless_backends() {
        for backend in [Backend::Dx12, Backend::Vulkan] {
            let device = Device::open(backend, DeviceOptions::default())
                .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
            assert_eq!(device.hardware().backend, backend);
            assert_ne!(device.hardware().name, "");
            assert_ne!(device.capabilities().max_texture_dimension_2d, 0);
            eprintln!("{device:?}");
        }
    }

    #[cfg(all(windows, feature = "dx12", feature = "vulkan"))]
    #[test]
    #[ignore = "requires native validation facilities and a real GPU driver"]
    fn creates_owned_resources_on_both_real_backends() {
        use crate::{
            BufferDescriptor, InvalidResourceReason, MemoryPolicy, ResourceCreateError,
            ResourceKind, TextureDescriptor,
        };
        use fluxel_rendergraph::{
            BufferDesc, BufferUsage, BufferUsageKind, Extent3d, TextureDesc, TextureDimension,
            TextureFormat, TextureUsage, TextureUsageKind,
        };

        for backend in [Backend::Dx12, Backend::Vulkan] {
            let before = crate::imp::destruction_counts();
            let device = Device::open(
                backend,
                DeviceOptions {
                    validation: Validation::Required,
                    ..DeviceOptions::default()
                },
            )
            .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
            let buffer_usage = BufferUsage::from_kinds([
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
                BufferUsageKind::StorageRead,
                BufferUsageKind::StorageWrite,
            ]);
            let buffer = device
                .create_buffer(BufferDescriptor {
                    buffer: BufferDesc { size: 4096 },
                    usage: buffer_usage,
                    memory: MemoryPolicy::DeviceOnly,
                })
                .unwrap_or_else(|error| panic!("{backend:?} A01/A03 failed: {error}"));
            assert_eq!(buffer.allowed_usage(), buffer_usage);
            assert_eq!(buffer.device_identity(), device.identity());
            let lease = buffer.lease();
            let lease_clone = lease.clone();
            drop(buffer);
            assert_eq!(crate::imp::destruction_counts(), before);
            drop(lease);
            assert_eq!(crate::imp::destruction_counts(), before);
            drop(device);
            assert_eq!(crate::imp::destruction_counts(), before);
            drop(lease_clone);
            assert_eq!(crate::imp::destruction_counts().0, before.0 + 1);

            let device = Device::open(
                backend,
                DeviceOptions {
                    validation: Validation::Required,
                    ..DeviceOptions::default()
                },
            )
            .unwrap();
            let texture_usage = TextureUsage::from_kinds([
                TextureUsageKind::CopySource,
                TextureUsageKind::CopyDestination,
                TextureUsageKind::ColorAttachment,
            ]);
            let texture_desc = TextureDescriptor {
                texture: TextureDesc {
                    dimension: TextureDimension::D2,
                    extent: Extent3d {
                        width: 64,
                        height: 32,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    sample_count: 1,
                    format: TextureFormat::Rgba8Unorm,
                },
                usage: texture_usage,
                memory: MemoryPolicy::DeviceOnly,
            };
            let texture = device
                .create_texture(texture_desc)
                .unwrap_or_else(|error| panic!("{backend:?} A02 failed: {error}"));
            assert_eq!(texture.descriptor(), texture_desc);
            assert_eq!(texture.allowed_usage(), texture_usage);

            let invalid = TextureDescriptor {
                usage: TextureUsage::empty().with(TextureUsageKind::Present),
                ..texture_desc
            };
            assert_eq!(
                device.create_texture(invalid).unwrap_err(),
                ResourceCreateError::InvalidDescriptor {
                    resource: ResourceKind::Texture,
                    reason: InvalidResourceReason::PresentRequiresSurface,
                }
            );
            assert_eq!(crate::imp::destruction_counts().1, before.1);
            let incompatible = TextureDescriptor {
                texture: TextureDesc {
                    format: TextureFormat::Depth32Float,
                    ..texture_desc.texture
                },
                usage: TextureUsage::empty().with(TextureUsageKind::ColorAttachment),
                ..texture_desc
            };
            assert_eq!(
                device.create_texture(incompatible).unwrap_err(),
                ResourceCreateError::InvalidDescriptor {
                    resource: ResourceKind::Texture,
                    reason: InvalidResourceReason::IncompatibleUsage,
                }
            );
            let subsequent = device
                .create_texture(texture_desc)
                .unwrap_or_else(|error| panic!("{backend:?} A04 recovery failed: {error}"));
            assert_ne!(texture.identity(), subsequent.identity());
            drop(texture);
            drop(subsequent);
            assert_eq!(crate::imp::destruction_counts().1, before.1 + 2);
            eprintln!(
                "A01-A06 {backend:?}: hardware={:?}; buffer={:?}; texture={:?}; structured rejection recovered; destroyed buffer=1 texture=2; validation diagnostics=none observed",
                device.hardware(),
                buffer_usage,
                texture_desc
            );
        }
    }

    #[cfg(all(windows, feature = "dx12"))]
    #[test]
    #[ignore = "requires native validation facilities"]
    fn opens_dx12_with_required_validation() {
        open_with_required_validation(Backend::Dx12);
    }

    #[cfg(all(windows, feature = "vulkan"))]
    #[test]
    #[ignore = "requires native validation facilities"]
    fn opens_vulkan_with_required_validation() {
        open_with_required_validation(Backend::Vulkan);
    }

    #[cfg(windows)]
    fn open_with_required_validation(backend: Backend) {
        let device = Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("failed to open {backend:?}: {error}"));
        assert_eq!(device.hardware().backend, backend);
        eprintln!("{device:?}");
    }
}

#[cfg(windows)]
mod imp;

#[cfg(not(windows))]
mod imp {
    use super::{
        Backend, BufferDescriptor, DeviceOptions, HardwareCapabilities, HardwareInfo, OpenError,
        ResourceCreateError, ResourceLease, TextureDescriptor,
    };
    use fluxel_rendergraph::{
        BufferCopyRegion, BufferUsage, CompletionFailure, CompletionStatus, ResourceAccessState,
        TextureCopyRegion, TextureDesc, TextureRange, TextureUsage,
    };
    use std::sync::Arc;
    pub(super) struct OpenedDevice {
        pub(super) hardware: HardwareInfo,
        pub(super) capabilities: HardwareCapabilities,
    }
    pub(super) struct OwnedBuffer;
    pub(super) struct OwnedTexture;
    pub(super) struct CopyEncoder;
    pub(super) struct CopyCommandBuffer;
    pub(super) struct NativeComputePipeline;
    pub(super) struct NativeComputeBindings;
    #[derive(Clone)]
    pub(super) struct NativeCompletion;
    pub(super) fn open(backend: Backend, _: DeviceOptions) -> Result<OpenedDevice, OpenError> {
        Err(OpenError::PlatformUnsupported { backend })
    }
    pub(super) fn create_buffer(
        owner: &Arc<OpenedDevice>,
        _: BufferDescriptor,
    ) -> Result<(OwnedBuffer, BufferUsage), ResourceCreateError> {
        Err(ResourceCreateError::NativeFailure {
            backend: owner.hardware.backend,
            reason: "native resources are only supported on Windows".into(),
        })
    }
    pub(super) fn create_texture(
        owner: &Arc<OpenedDevice>,
        _: TextureDescriptor,
    ) -> Result<(OwnedTexture, TextureUsage), ResourceCreateError> {
        Err(ResourceCreateError::NativeFailure {
            backend: owner.hardware.backend,
            reason: "native resources are only supported on Windows".into(),
        })
    }
    pub(super) fn begin_copy_encoder(_: &Arc<OpenedDevice>) -> Result<CopyEncoder, String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn create_compute_pipeline(
        _: &Arc<OpenedDevice>,
        _: &str,
        _: &str,
    ) -> Result<NativeComputePipeline, String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn create_compute_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeComputePipeline,
        _: &OwnedBuffer,
        _: u64,
        _: u64,
    ) -> Result<NativeComputeBindings, String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn transition_texture(
        _: &mut CopyEncoder,
        _: &OwnedTexture,
        _: TextureDesc,
        _: TextureRange,
        _: ResourceAccessState,
        _: ResourceAccessState,
    ) -> Result<(), String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn transition_buffer(
        _: &mut CopyEncoder,
        _: &OwnedBuffer,
        _: ResourceAccessState,
        _: ResourceAccessState,
    ) -> Result<(), String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn begin_compute(_: &mut CopyEncoder, _: &str) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn end_compute(_: &mut CopyEncoder) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn set_compute_pipeline(
        _: &mut CopyEncoder,
        _: &NativeComputePipeline,
    ) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn set_compute_bindings(
        _: &mut CopyEncoder,
        _: &NativeComputeBindings,
    ) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn dispatch(_: &mut CopyEncoder, _: [u32; 3]) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn copy_texture(
        _: &mut CopyEncoder,
        _: &OwnedTexture,
        _: &OwnedTexture,
        _: TextureDesc,
        _: TextureCopyRegion,
    ) -> Result<(), String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn copy_buffer(
        _: &mut CopyEncoder,
        _: &OwnedBuffer,
        _: &OwnedBuffer,
        _: BufferCopyRegion,
    ) -> Result<(), String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn finish_copy_encoder(_: CopyEncoder) -> Result<CopyCommandBuffer, String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn submit_copy(
        _: CopyCommandBuffer,
        _: Vec<ResourceLease>,
    ) -> Result<NativeCompletion, String> {
        Err("native execution is only supported on Windows".into())
    }
    pub(super) fn completion_status(_: &NativeCompletion) -> Result<CompletionStatus, String> {
        Ok(CompletionStatus::Failed(CompletionFailure::DeviceLost))
    }
    pub(super) fn wait_completion(
        _: &NativeCompletion,
        _: core::time::Duration,
    ) -> Result<CompletionStatus, String> {
        Ok(CompletionStatus::Failed(CompletionFailure::DeviceLost))
    }
}
