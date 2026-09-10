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
    // This counter supplies process-local uniqueness only. Relaxed ordering is
    // sufficient because identity does not publish native objects or synchronize
    // their lifetime; ownership and queue locks provide those guarantees.
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
mod tests;

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

mod imp;
