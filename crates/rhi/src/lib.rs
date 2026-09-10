//! Native GPU ownership plus fixed-artifact Copy, Compute, and Raster execution for Fluxel.
//!
//! This crate exposes no `wgpu-hal` types. It opens one headless DX12 or Vulkan
//! device, owns buffers and 2D textures, and executes deliberately fixed Copy,
//! Compute, and Raster subsets of a portable RenderGraph plan through barriers,
//! one submission, completion, and lease-backed retirement. Surfaces and
//! presentation remain outside this milestone. Narrow immutable buffer and
//! whole RGBA8 texture uploads retain staging and destination storage through
//! completion.

#![deny(missing_docs)]

use core::fmt;
use std::sync::Arc;

mod execution;
mod resource;

pub use execution::*;
/// Opaque identity of one physical RenderGraph resource generation.
pub use fluxel_rendergraph::PhysicalResourceIdentity;
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
    /// Whether `Rgba8Unorm` supports filtering linear texture samples.
    ///
    /// This is an unmodified adapter fact. The fixed linear-clamp raster
    /// artifact rejects creation when it is false rather than assuming that
    /// a sampled `Rgba8Unorm` texture is filterable.
    pub rgba8_unorm_filterable: bool,
    /// Whether `Rgba8UnormSrgb` supports filtering linear texture samples.
    ///
    /// This is a separate, unmodified adapter fact. In particular, Fluxel
    /// never infers sRGB filterability from the corresponding UNORM fact.
    pub rgba8_unorm_srgb_filterable: bool,
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

/// A headless native device that owns resources and serial graphics-queue execution.
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
        #[cfg(all(windows, feature = "test-support"))]
        imp::initialize_validation_capture();
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
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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

    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
mod imp;

/// Doc-hidden conformance observation helpers for workspace hardware fixtures.
///
/// This module is non-default and deliberately exposes neither host mapping,
/// command recording, nor native handles. Production callers must not use it
/// as a general readback API.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support {
    use crate::{
        BufferUploadError, BufferUploadStage, Device, TextureUploadError, TextureUploadStage,
        UploadedBuffer, UploadedTexture,
    };

    /// Clears diagnostics collected after `Device::open` enabled capture.
    pub fn clear_validation_diagnostics(device: &Device) {
        #[cfg(windows)]
        crate::imp::clear_validation_diagnostics(&device.inner);
        #[cfg(not(windows))]
        let _ = device;
    }

    /// Returns validation warnings and errors collected for `device`.
    #[must_use]
    pub fn validation_diagnostics(device: &Device) -> Vec<String> {
        #[cfg(windows)]
        {
            crate::imp::validation_diagnostics(&device.inner)
        }
        #[cfg(not(windows))]
        {
            let _ = device;
            Vec::new()
        }
    }

    /// Makes the next native submission fail before queue acceptance.
    ///
    /// This conformance-only hook is one-shot and has no production build
    /// surface. It exercises the path which must release reservations because
    /// native work was never accepted.
    pub fn inject_submit_rejected_once() {
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        crate::imp::inject_submit_rejected_once();
    }

    /// Makes the submission after exactly `successful_submits` accepted
    /// submissions fail before queue acceptance.
    ///
    /// Passing zero is identical to [`inject_submit_rejected_once`]. This is
    /// conformance-only and fixtures must serialize configuration with their
    /// existing guard.
    pub fn inject_submit_rejected_after(successful_submits: usize) {
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        crate::imp::inject_submit_rejected_after(successful_submits);
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        let _ = successful_submits;
    }

    /// Makes the next native submission report accepted-but-unknown failure.
    ///
    /// This conformance-only hook is one-shot and exercises quarantine of all
    /// submitted leases: callers must not infer a final resource state.
    pub fn inject_submit_accepted_unknown_once() {
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        crate::imp::inject_submit_accepted_unknown_once();
    }

    /// Makes the submission after exactly `successful_submits` accepted
    /// submissions report accepted-but-unknown failure.
    ///
    /// Passing zero is identical to [`inject_submit_accepted_unknown_once`].
    /// This is conformance-only and fixtures must serialize configuration with
    /// their existing guard.
    pub fn inject_submit_accepted_unknown_after(successful_submits: usize) {
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        crate::imp::inject_submit_accepted_unknown_after(successful_submits);
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        let _ = successful_submits;
    }

    /// Makes the next nonblocking native completion observation report `Pending`.
    ///
    /// This conformance-only hook is one-shot and is consumed by the same
    /// completion-status path used by renderer polling. It never alters a
    /// later blocking readback wait.
    pub fn inject_completion_pending_once() {
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        crate::imp::inject_completion_pending_once();
    }

    /// Reads a finalized immutable upload using its exact published state.
    ///
    /// The helper is only for CPU-oracle hardware fixtures. It consumes the
    /// upload's `CopyDestination` state as the true incoming state before the
    /// private readback transition, so it cannot repair a wrong upload state.
    pub fn readback_uploaded_buffer(
        device: &Device,
        uploaded: &UploadedBuffer,
    ) -> Result<Vec<u8>, BufferUploadError> {
        if uploaded.buffer().device_identity() != device.identity() {
            return Err(BufferUploadError::ForeignDevice);
        }
        if !uploaded
            .buffer()
            .allowed_usage()
            .contains(fluxel_rendergraph::BufferUsageKind::CopySource)
        {
            return Err(BufferUploadError::InvalidRequest(
                crate::InvalidBufferUploadReason::CopySourceUsageRequired,
            ));
        }
        #[cfg(windows)]
        {
            crate::imp::readback_buffer_for_test(
                &device.inner,
                uploaded.buffer().native(),
                uploaded.lease().into(),
                uploaded.outgoing_state(),
                uploaded.buffer().descriptor().buffer.size,
            )
            .map_err(|reason| BufferUploadError::Native {
                backend: device.hardware().backend,
                stage: BufferUploadStage::Completion,
                reason,
            })
        }
        #[cfg(not(windows))]
        {
            let _ = uploaded;
            Err(BufferUploadError::Native {
                backend: device.hardware().backend,
                stage: BufferUploadStage::Completion,
                reason: "native readback is only supported on Windows".into(),
            })
        }
    }

    /// Reads a finalized immutable texture using its exact exported incoming
    /// state. The returned tuple is `(tight_rgba8, padded_native_rows,
    /// bytes_per_row)` for hardware CPU-oracle fixtures only.
    ///
    /// This helper consumes [`UploadedTexture::outgoing_state`] as its actual
    /// incoming state and restores that state afterward; it never assumes or
    /// silently repairs an incorrect graph export.
    pub fn readback_uploaded_texture(
        device: &Device,
        uploaded: &UploadedTexture,
    ) -> Result<(Vec<u8>, Vec<u8>, u32), TextureUploadError> {
        if uploaded.texture().device_identity() != device.identity() {
            return Err(TextureUploadError::ForeignDevice);
        }
        if !uploaded
            .texture()
            .allowed_usage()
            .contains(fluxel_rendergraph::TextureUsageKind::CopySource)
        {
            return Err(TextureUploadError::InvalidRequest(
                crate::InvalidTextureUploadReason::UnexpectedUsage,
            ));
        }
        #[cfg(windows)]
        {
            let readback = crate::imp::readback_texture_for_test(
                &device.inner,
                uploaded.texture().native(),
                uploaded.lease().into(),
                uploaded.texture().descriptor().texture,
                uploaded.outgoing_state(),
            )
            .map_err(|reason| TextureUploadError::Native {
                backend: device.hardware().backend,
                stage: TextureUploadStage::Completion,
                reason,
            })?;
            Ok((readback.tight, readback.padded, readback.bytes_per_row))
        }
        #[cfg(not(windows))]
        {
            let _ = uploaded;
            Err(TextureUploadError::Native {
                backend: device.hardware().backend,
                stage: TextureUploadStage::Completion,
                reason: "native readback is only supported on Windows".into(),
            })
        }
    }
}

#[cfg(any(
    not(windows),
    all(windows, not(any(feature = "dx12", feature = "vulkan")))
))]
mod imp {
    use super::{
        Backend, BufferDescriptor, BufferUploadStage, DeviceOptions, HardwareCapabilities,
        HardwareInfo, OpenError, ResourceCreateError, ResourceLease, TextureDescriptor,
        TextureUploadStage,
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
    pub(super) struct NativeTexturePackBindings;
    pub(super) struct NativeRasterPipeline;
    pub(super) struct NativeRasterUniformBindings;
    pub(super) struct NativeRasterTextureBindings;
    #[derive(Clone)]
    pub(super) struct NativeCompletion;
    pub(super) fn open(backend: Backend, _: DeviceOptions) -> Result<OpenedDevice, OpenError> {
        #[cfg(windows)]
        {
            Err(OpenError::BackendDisabled { backend })
        }
        #[cfg(not(windows))]
        {
            Err(OpenError::PlatformUnsupported { backend })
        }
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
    pub(super) fn upload_immutable_buffer(
        _: &Arc<OpenedDevice>,
        _: &OwnedBuffer,
        _: ResourceLease,
        _: &[u8],
    ) -> Result<NativeCompletion, (BufferUploadStage, String)> {
        Err((
            BufferUploadStage::Staging,
            "native uploads are unavailable without a Windows backend".into(),
        ))
    }
    pub(super) fn upload_immutable_texture(
        _: &Arc<OpenedDevice>,
        _: &OwnedTexture,
        _: ResourceLease,
        _: TextureDesc,
        _: &[u8],
    ) -> Result<NativeCompletion, (TextureUploadStage, String)> {
        Err((
            TextureUploadStage::Staging,
            "native uploads are unavailable without a Windows backend".into(),
        ))
    }
    #[cfg(feature = "test-support")]
    pub(super) fn initialize_validation_capture() {}
    #[cfg(feature = "test-support")]
    pub(super) fn clear_validation_diagnostics(_: &Arc<OpenedDevice>) {}
    #[cfg(feature = "test-support")]
    pub(super) fn validation_diagnostics(_: &Arc<OpenedDevice>) -> Vec<String> {
        Vec::new()
    }
    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn readback_buffer_for_test(
        _: &Arc<OpenedDevice>,
        _: &OwnedBuffer,
        _: ResourceLease,
        _: ResourceAccessState,
        _: u64,
    ) -> Result<Vec<u8>, String> {
        Err("native readback is unavailable without a Windows backend".into())
    }
    #[cfg(any(test, feature = "test-support"))]
    pub(super) struct TextureReadback {
        pub(super) tight: Vec<u8>,
        pub(super) padded: Vec<u8>,
        pub(super) bytes_per_row: u32,
    }
    #[cfg(any(test, feature = "test-support"))]
    pub(super) fn readback_texture_for_test(
        _: &Arc<OpenedDevice>,
        _: &OwnedTexture,
        _: ResourceLease,
        _: TextureDesc,
        _: ResourceAccessState,
    ) -> Result<TextureReadback, String> {
        Err("native texture readback is unavailable without a Windows backend".into())
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
    pub(super) fn create_texture_pack_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeComputePipeline,
        _: &OwnedTexture,
        _: &OwnedBuffer,
        _: u64,
        _: u64,
    ) -> Result<NativeTexturePackBindings, String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn create_raster_pipeline(
        _: &Arc<OpenedDevice>,
        _: &str,
        _: &str,
        _: &str,
        _: crate::RasterKernel,
    ) -> Result<NativeRasterPipeline, String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn create_raster_uniform_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
    ) -> Result<NativeRasterUniformBindings, String> {
        Err("native raster is only supported on Windows".into())
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed native normal recipe passes all independently validated role facts"
    )]
    pub(super) fn create_raster_normal_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
    ) -> Result<NativeRasterUniformBindings, String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn create_raster_texture_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
        _: &OwnedTexture,
    ) -> Result<NativeRasterTextureBindings, String> {
        Err("native raster is only supported on Windows".into())
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed native UV recipe passes all independently validated role facts"
    )]
    pub(super) fn create_raster_uv_texture_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
        _: &OwnedTexture,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
    ) -> Result<NativeRasterTextureBindings, String> {
        Err("native raster is only supported on Windows".into())
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed native linear-clamp UV recipe passes all independently validated role facts"
    )]
    pub(super) fn create_raster_uv_linear_clamp_texture_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
        _: &OwnedTexture,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
    ) -> Result<NativeRasterTextureBindings, String> {
        Err("native raster is only supported on Windows".into())
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed native sRGB linear-clamp UV recipe passes all independently validated role facts"
    )]
    pub(super) fn create_raster_uv_linear_clamp_srgb_texture_bindings(
        _: &Arc<OpenedDevice>,
        _: &NativeRasterPipeline,
        _: &OwnedBuffer,
        _: &OwnedTexture,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: u64,
    ) -> Result<NativeRasterTextureBindings, String> {
        Err("native raster is only supported on Windows".into())
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
    pub(super) fn begin_raster(
        _: &mut CopyEncoder,
        _: &OwnedTexture,
        _: TextureDesc,
        _: Option<[f32; 4]>,
        _: bool,
        _: bool,
        _: &str,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn end_raster(_: &mut CopyEncoder) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
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
    pub(super) fn set_texture_pack_bindings(
        _: &mut CopyEncoder,
        _: &NativeTexturePackBindings,
    ) -> Result<(), String> {
        Err("native compute is only supported on Windows".into())
    }
    pub(super) fn set_raster_pipeline(
        _: &mut CopyEncoder,
        _: &NativeRasterPipeline,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_raster_uniform_bindings(
        _: &mut CopyEncoder,
        _: &NativeRasterUniformBindings,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_raster_texture_bindings(
        _: &mut CopyEncoder,
        _: &NativeRasterTextureBindings,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_raster_uv_texture_bindings(
        _: &mut CopyEncoder,
        _: &NativeRasterTextureBindings,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_raster_uv_linear_clamp_texture_bindings(
        _: &mut CopyEncoder,
        _: &NativeRasterTextureBindings,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_raster_uv_linear_clamp_srgb_texture_bindings(
        _: &mut CopyEncoder,
        _: &NativeRasterTextureBindings,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed native UV recipe passes all independently validated role facts"
    )]
    pub(super) fn set_vertex_buffer(
        _: &mut CopyEncoder,
        _: &OwnedBuffer,
        _: u64,
        _: u64,
        _: crate::RasterKernel,
        _: u32,
        _: fluxel_rendergraph::PhysicalResourceIdentity,
        _: Option<&NativeRasterTextureBindings>,
        _: Option<&NativeRasterUniformBindings>,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_index_buffer(
        _: &mut CopyEncoder,
        _: &OwnedBuffer,
        _: u64,
        _: u64,
        _: fluxel_rendergraph::IndexFormat,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_viewport(
        _: &mut CopyEncoder,
        _: f32,
        _: f32,
        _: f32,
        _: f32,
        _: f32,
        _: f32,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn set_scissor(
        _: &mut CopyEncoder,
        _: u32,
        _: u32,
        _: u32,
        _: u32,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn draw(_: &mut CopyEncoder, _: u32, _: u32, _: u32, _: u32) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
    }
    pub(super) fn draw_indexed(
        _: &mut CopyEncoder,
        _: u32,
        _: u32,
        _: i32,
        _: u32,
        _: u32,
    ) -> Result<(), String> {
        Err("native raster is only supported on Windows".into())
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
