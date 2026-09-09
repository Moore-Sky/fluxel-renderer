//! Native GPU bootstrap for Fluxel execution.
//!
//! This crate deliberately exposes no `wgpu-hal` types. At this stage it can
//! open one headless DX12 or Vulkan device and report the unmodified hardware
//! facts used to do so; command recording is introduced by later milestones.

#![deny(missing_docs)]

use core::fmt;

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
/// `Required` is fail-closed during A1. Because the HAL has no portable proof
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
}

/// A headless native device that has not yet recorded or submitted work.
pub struct Device {
    #[allow(
        dead_code,
        reason = "A1 retains native ownership solely for safe shutdown"
    )]
    inner: imp::OpenedDevice,
    hardware: HardwareInfo,
    capabilities: HardwareCapabilities,
}

impl Device {
    /// Opens one explicitly selected backend and adapter for headless use.
    ///
    /// This method contains all HAL `unsafe` calls, creates no surface, and
    /// exposes no native handles. No command submission occurs during A1.
    pub fn open(backend: Backend, options: DeviceOptions) -> Result<Self, OpenError> {
        let opened = imp::open(backend, options)?;
        Ok(Self {
            hardware: opened.hardware.clone(),
            capabilities: opened.capabilities,
            inner: opened,
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
    /// This crate only implements native opening on Windows in A1.
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
    /// The requested validation mode cannot be positively verified in A1.
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
    use super::{Backend, DeviceOptions, HardwareCapabilities, HardwareInfo, OpenError};
    pub(super) struct OpenedDevice {
        pub(super) hardware: HardwareInfo,
        pub(super) capabilities: HardwareCapabilities,
    }
    pub(super) fn open(backend: Backend, _: DeviceOptions) -> Result<OpenedDevice, OpenError> {
        Err(OpenError::PlatformUnsupported { backend })
    }
}
