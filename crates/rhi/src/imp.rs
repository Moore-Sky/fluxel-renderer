//! Windows-only containment of the `wgpu-hal` bootstrap API.

use crate::{
    Backend, BufferDescriptor, DeviceKind, DeviceOptions, HardwareCapabilities, HardwareInfo,
    OpenError, ResourceCreateError, TextureDescriptor, Validation,
};
use fluxel_rendergraph::{
    BufferUsage, BufferUsageKind, TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};
use std::sync::Arc;
use wgpu_hal::{Adapter as _, Device as _, Instance as _};
use wgpu_types as wgt;

pub(super) struct OpenedDevice {
    // Drop order: queue, device, adapter, then instance. This retains DX12
    // adapter-owned driver state until all dependent values are gone.
    #[allow(dead_code)]
    native: NativeDevice,
    pub(super) hardware: HardwareInfo,
    pub(super) capabilities: HardwareCapabilities,
}

#[allow(
    dead_code,
    clippy::large_enum_variant,
    reason = "native objects are retained for safe shutdown and remain inline because device creation is not a hot path"
)]
enum NativeDevice {
    #[cfg(feature = "dx12")]
    Dx12 {
        queue: wgpu_hal::dx12::Queue,
        device: wgpu_hal::dx12::Device,
        adapter: wgpu_hal::dx12::Adapter,
        instance: wgpu_hal::dx12::Instance,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        queue: wgpu_hal::vulkan::Queue,
        device: wgpu_hal::vulkan::Device,
        adapter: wgpu_hal::vulkan::Adapter,
        instance: wgpu_hal::vulkan::Instance,
    },
}

enum NativeBuffer {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Buffer),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::Buffer),
}

enum NativeTexture {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Texture),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::Texture),
}

pub(super) struct OwnedBuffer {
    native: Option<NativeBuffer>,
    owner: Arc<OpenedDevice>,
}

pub(super) struct OwnedTexture {
    native: Option<NativeTexture>,
    owner: Arc<OpenedDevice>,
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        let native = self.native.take().expect("owned buffer destroyed once");
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(buffer)) => {
                // SAFETY: this buffer was created by this live device, has never
                // been submitted in 0.1.1, and this unique owner destroys it once.
                unsafe { device.destroy_buffer(buffer) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(buffer)) => {
                // SAFETY: same-device ownership is retained by `owner`; 0.1.1
                // has no mapping or submission and the Option enforces one drop.
                unsafe { device.destroy_buffer(buffer) };
            }
            _ => unreachable!("resource and device backend always match"),
        }
        #[cfg(test)]
        DESTROYED_BUFFERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for OwnedTexture {
    fn drop(&mut self) {
        let native = self.native.take().expect("owned texture destroyed once");
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeTexture::Dx12(texture)) => {
                // SAFETY: this texture belongs to the retained live device and
                // no commands can reference it in the 0.1.1 API.
                unsafe { device.destroy_texture(texture) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeTexture::Vulkan(texture)) => {
                // SAFETY: same-device ownership and unique final destruction are
                // enforced structurally; 0.1.1 exposes no submission.
                unsafe { device.destroy_texture(texture) };
            }
            _ => unreachable!("resource and device backend always match"),
        }
        #[cfg(test)]
        DESTROYED_TEXTURES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
static DESTROYED_BUFFERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static DESTROYED_TEXTURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(super) fn destruction_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering;
    (
        DESTROYED_BUFFERS.load(Ordering::SeqCst),
        DESTROYED_TEXTURES.load(Ordering::SeqCst),
    )
}

pub(super) fn create_buffer(
    owner: &Arc<OpenedDevice>,
    descriptor: BufferDescriptor,
) -> Result<(OwnedBuffer, BufferUsage), ResourceCreateError> {
    let native_usage = lower_buffer_usage(descriptor.usage);
    let hal_desc = wgpu_hal::BufferDescriptor {
        label: Some("fluxel owned buffer"),
        size: descriptor.buffer.size,
        usage: native_usage,
        memory_flags: wgpu_hal::MemoryFlags::empty(),
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => {
            // SAFETY: the safe layer validated non-zero size and domain usage;
            // lowering uses only HAL-defined flags and no mapping is requested.
            NativeBuffer::Dx12(
                unsafe { device.create_buffer(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Dx12, error))?,
            )
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => {
            // SAFETY: identical portable validation precedes backend-specific
            // creation and MemoryFlags is the validated DeviceOnly policy.
            NativeBuffer::Vulkan(
                unsafe { device.create_buffer(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Vulkan, error))?,
            )
        }
    };
    Ok((
        OwnedBuffer {
            native: Some(native),
            owner: Arc::clone(owner),
        },
        buffer_usage_from_native(native_usage),
    ))
}

pub(super) fn create_texture(
    owner: &Arc<OpenedDevice>,
    descriptor: TextureDescriptor,
) -> Result<(OwnedTexture, TextureUsage), ResourceCreateError> {
    let native_usage = lower_texture_usage(descriptor.usage);
    let texture = descriptor.texture;
    let hal_desc = wgpu_hal::TextureDescriptor {
        label: Some("fluxel owned texture"),
        size: wgt::Extent3d {
            width: texture.extent.width,
            height: texture.extent.height,
            depth_or_array_layers: match texture.dimension {
                TextureDimension::D2 => texture.array_layers,
                TextureDimension::D3 => texture.extent.depth,
                _ => 1,
            },
        },
        mip_level_count: texture.mip_levels,
        sample_count: texture.sample_count,
        dimension: lower_dimension(texture.dimension),
        format: lower_format(texture.format),
        usage: native_usage,
        memory_flags: wgpu_hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 {
            device, adapter, ..
        } => {
            // SAFETY: the format is a valid wgpu format and the retained
            // adapter is the exact parent used to open this device.
            let caps = unsafe { adapter.texture_format_capabilities(hal_desc.format) };
            validate_texture_capabilities(caps, descriptor)?;
            // SAFETY: extent, dimension, mip, sample, format and usage
            // compatibility were validated before this private HAL boundary.
            NativeTexture::Dx12(
                unsafe { device.create_texture(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Dx12, error))?,
            )
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan {
            device, adapter, ..
        } => {
            // SAFETY: the retained live adapter accepts this portable format;
            // the query performs no resource operation.
            let caps = unsafe { adapter.texture_format_capabilities(hal_desc.format) };
            validate_texture_capabilities(caps, descriptor)?;
            // SAFETY: all HAL descriptor preconditions consumed by this slice
            // are validated, and the returned texture remains device-owned.
            NativeTexture::Vulkan(
                unsafe { device.create_texture(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Vulkan, error))?,
            )
        }
    };
    Ok((
        OwnedTexture {
            native: Some(native),
            owner: Arc::clone(owner),
        },
        texture_usage_from_native(native_usage, texture.format),
    ))
}

fn resource_error(backend: Backend, error: impl core::fmt::Display) -> ResourceCreateError {
    ResourceCreateError::NativeFailure {
        backend,
        reason: error.to_string(),
    }
}

fn validate_texture_capabilities(
    caps: wgpu_hal::TextureFormatCapabilities,
    descriptor: TextureDescriptor,
) -> Result<(), ResourceCreateError> {
    let usage = descriptor.usage;
    let supported = (!usage.contains(TextureUsageKind::Sampled)
        || caps.contains(wgpu_hal::TextureFormatCapabilities::SAMPLED))
        && (!usage.contains(TextureUsageKind::StorageRead)
            || caps.intersects(
                wgpu_hal::TextureFormatCapabilities::STORAGE_READ_ONLY
                    | wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE,
            ))
        && (!usage.contains(TextureUsageKind::StorageWrite)
            || caps.intersects(
                wgpu_hal::TextureFormatCapabilities::STORAGE_WRITE_ONLY
                    | wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE,
            ))
        && (!(usage.contains(TextureUsageKind::StorageRead)
            && usage.contains(TextureUsageKind::StorageWrite))
            || caps.contains(wgpu_hal::TextureFormatCapabilities::STORAGE_READ_WRITE))
        && (!usage.contains(TextureUsageKind::ColorAttachment)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COLOR_ATTACHMENT))
        && (!usage.contains(TextureUsageKind::DepthStencilAttachment)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::DEPTH_STENCIL_ATTACHMENT))
        && (!usage.contains(TextureUsageKind::CopySource)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COPY_SRC))
        && (!usage.contains(TextureUsageKind::CopyDestination)
            || caps.contains(wgpu_hal::TextureFormatCapabilities::COPY_DST));
    let sample_supported = match descriptor.texture.sample_count {
        1 => true,
        2 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X2),
        4 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X4),
        8 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X8),
        16 => caps.contains(wgpu_hal::TextureFormatCapabilities::MULTISAMPLE_X16),
        _ => false,
    };
    if supported && sample_supported {
        Ok(())
    } else {
        Err(ResourceCreateError::InvalidDescriptor {
            resource: crate::ResourceKind::Texture,
            reason: crate::InvalidResourceReason::IncompatibleUsage,
        })
    }
}

fn lower_buffer_usage(usage: BufferUsage) -> wgt::BufferUses {
    let mut native = wgt::BufferUses::empty();
    for (kind, flag) in [
        (BufferUsageKind::Uniform, wgt::BufferUses::UNIFORM),
        (
            BufferUsageKind::StorageRead,
            wgt::BufferUses::STORAGE_READ_ONLY,
        ),
        (
            BufferUsageKind::StorageWrite,
            wgt::BufferUses::STORAGE_READ_WRITE,
        ),
        (BufferUsageKind::Vertex, wgt::BufferUses::VERTEX),
        (BufferUsageKind::Index, wgt::BufferUses::INDEX),
        (BufferUsageKind::Indirect, wgt::BufferUses::INDIRECT),
        (BufferUsageKind::CopySource, wgt::BufferUses::COPY_SRC),
        (BufferUsageKind::CopyDestination, wgt::BufferUses::COPY_DST),
    ] {
        if usage.contains(kind) {
            native |= flag;
        }
    }
    native
}

fn buffer_usage_from_native(native: wgt::BufferUses) -> BufferUsage {
    let mut kinds = Vec::new();
    if native.contains(wgt::BufferUses::UNIFORM) {
        kinds.push(BufferUsageKind::Uniform);
    }
    if native.intersects(wgt::BufferUses::STORAGE_READ_ONLY | wgt::BufferUses::STORAGE_READ_WRITE) {
        kinds.push(BufferUsageKind::StorageRead);
    }
    if native.contains(wgt::BufferUses::STORAGE_READ_WRITE) {
        kinds.push(BufferUsageKind::StorageWrite);
    }
    if native.contains(wgt::BufferUses::VERTEX) {
        kinds.push(BufferUsageKind::Vertex);
    }
    if native.contains(wgt::BufferUses::INDEX) {
        kinds.push(BufferUsageKind::Index);
    }
    if native.contains(wgt::BufferUses::INDIRECT) {
        kinds.push(BufferUsageKind::Indirect);
    }
    if native.contains(wgt::BufferUses::COPY_SRC) {
        kinds.push(BufferUsageKind::CopySource);
    }
    if native.contains(wgt::BufferUses::COPY_DST) {
        kinds.push(BufferUsageKind::CopyDestination);
    }
    BufferUsage::from_kinds(kinds)
}

fn lower_texture_usage(usage: TextureUsage) -> wgt::TextureUses {
    let mut native = wgt::TextureUses::empty();
    if usage.contains(TextureUsageKind::Sampled) {
        native |= wgt::TextureUses::RESOURCE;
    }
    match (
        usage.contains(TextureUsageKind::StorageRead),
        usage.contains(TextureUsageKind::StorageWrite),
    ) {
        (true, true) => native |= wgt::TextureUses::STORAGE_READ_WRITE,
        (true, false) => native |= wgt::TextureUses::STORAGE_READ_ONLY,
        (false, true) => native |= wgt::TextureUses::STORAGE_WRITE_ONLY,
        _ => {}
    }
    if usage.contains(TextureUsageKind::ColorAttachment) {
        native |= wgt::TextureUses::COLOR_TARGET;
    }
    if usage.contains(TextureUsageKind::DepthStencilAttachment) {
        native |= wgt::TextureUses::DEPTH_STENCIL_WRITE;
    }
    if usage.contains(TextureUsageKind::CopySource) {
        native |= wgt::TextureUses::COPY_SRC;
    }
    if usage.contains(TextureUsageKind::CopyDestination) {
        native |= wgt::TextureUses::COPY_DST;
    }
    native
}

fn texture_usage_from_native(native: wgt::TextureUses, format: TextureFormat) -> TextureUsage {
    let mut kinds = Vec::new();
    if native.contains(wgt::TextureUses::RESOURCE) {
        kinds.push(TextureUsageKind::Sampled);
    }
    if native.intersects(wgt::TextureUses::STORAGE_READ_ONLY | wgt::TextureUses::STORAGE_READ_WRITE)
    {
        kinds.push(TextureUsageKind::StorageRead);
    }
    if native
        .intersects(wgt::TextureUses::STORAGE_WRITE_ONLY | wgt::TextureUses::STORAGE_READ_WRITE)
    {
        kinds.push(TextureUsageKind::StorageWrite);
    }
    if native.contains(wgt::TextureUses::COLOR_TARGET) && format != TextureFormat::Depth32Float {
        kinds.push(TextureUsageKind::ColorAttachment);
    }
    if native.contains(wgt::TextureUses::DEPTH_STENCIL_WRITE)
        && format == TextureFormat::Depth32Float
    {
        kinds.push(TextureUsageKind::DepthStencilAttachment);
    }
    if native.contains(wgt::TextureUses::COPY_SRC) {
        kinds.push(TextureUsageKind::CopySource);
    }
    if native.contains(wgt::TextureUses::COPY_DST) {
        kinds.push(TextureUsageKind::CopyDestination);
    }
    TextureUsage::from_kinds(kinds)
}

fn lower_dimension(dimension: TextureDimension) -> wgt::TextureDimension {
    match dimension {
        TextureDimension::D1 => wgt::TextureDimension::D1,
        TextureDimension::D2 => wgt::TextureDimension::D2,
        TextureDimension::D3 => wgt::TextureDimension::D3,
        _ => unreachable!("validated dimension"),
    }
}

fn lower_format(format: TextureFormat) -> wgt::TextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => wgt::TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm => wgt::TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float => wgt::TextureFormat::Rgba16Float,
        TextureFormat::Depth32Float => wgt::TextureFormat::Depth32Float,
        _ => unreachable!("known portable format"),
    }
}

pub(super) fn open(backend: Backend, options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    match backend {
        Backend::Dx12 => open_dx12(options),
        Backend::Vulkan => open_vulkan(options),
    }
}

#[cfg(feature = "dx12")]
fn open_dx12(options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    let descriptor = instance_descriptor(options.validation);
    // SAFETY: the descriptor owns no display handle, outlives initialization,
    // and requests only HAL-defined instance flags.
    let instance = unsafe { wgpu_hal::dx12::Instance::init(&descriptor) }
        .map_err(|e| native_error(Backend::Dx12, e))?;
    // SAFETY: the live instance owns the factory used for enumeration; no
    // surface constraint is supplied for this headless device.
    let adapters = unsafe { instance.enumerate_adapters(None) };
    let available_adapters = adapters.len();
    let exposed =
        adapters
            .into_iter()
            .nth(options.adapter_index)
            .ok_or(OpenError::AdapterUnavailable {
                backend: Backend::Dx12,
                adapter_index: options.adapter_index,
                available_adapters,
            })?;
    let hardware = hardware(Backend::Dx12, &exposed.info);
    let capabilities = capabilities(exposed.features, &exposed.capabilities);
    let requested_limits = required_limits(Backend::Dx12, &exposed.capabilities)?;
    let adapter = exposed.adapter;
    // SAFETY: features are empty and default limits are validated by the
    // exposed adapter before HAL creates a device and its matching queue.
    let wgpu_hal::OpenDevice { device, queue } = unsafe {
        adapter.open(
            wgt::Features::empty(),
            &requested_limits,
            &wgt::MemoryHints::default(),
        )
    }
    .map_err(|e| native_error(Backend::Dx12, e))?;
    if options.validation == Validation::Required && !dx12_validation_is_enabled(&device) {
        return Err(OpenError::ValidationUnavailable {
            backend: Backend::Dx12,
        });
    }
    Ok(OpenedDevice {
        native: NativeDevice::Dx12 {
            queue,
            device,
            adapter,
            instance,
        },
        hardware,
        capabilities,
    })
}
#[cfg(not(feature = "dx12"))]
fn open_dx12(_: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    Err(OpenError::BackendDisabled {
        backend: Backend::Dx12,
    })
}

#[cfg(feature = "vulkan")]
fn open_vulkan(options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    if options.validation == Validation::Required && !vulkan_validation_is_available() {
        return Err(OpenError::ValidationUnavailable {
            backend: Backend::Vulkan,
        });
    }
    let descriptor = instance_descriptor(options.validation);
    // SAFETY: the descriptor owns no display handle, outlives initialization,
    // and requests only HAL-defined instance flags.
    let instance = unsafe { wgpu_hal::vulkan::Instance::init(&descriptor) }
        .map_err(|e| native_error(Backend::Vulkan, e))?;
    // SAFETY: the live instance owns all Vulkan entry points used during
    // enumeration; no surface constraint is supplied for headless execution.
    let adapters = unsafe { instance.enumerate_adapters(None) };
    let available_adapters = adapters.len();
    let exposed =
        adapters
            .into_iter()
            .nth(options.adapter_index)
            .ok_or(OpenError::AdapterUnavailable {
                backend: Backend::Vulkan,
                adapter_index: options.adapter_index,
                available_adapters,
            })?;
    let hardware = hardware(Backend::Vulkan, &exposed.info);
    let capabilities = capabilities(exposed.features, &exposed.capabilities);
    let requested_limits = required_limits(Backend::Vulkan, &exposed.capabilities)?;
    let adapter = exposed.adapter;
    // SAFETY: features are empty and default limits are validated by the
    // exposed adapter before HAL creates a device and its matching queue.
    let wgpu_hal::OpenDevice { device, queue } = unsafe {
        adapter.open(
            wgt::Features::empty(),
            &requested_limits,
            &wgt::MemoryHints::default(),
        )
    }
    .map_err(|e| native_error(Backend::Vulkan, e))?;
    Ok(OpenedDevice {
        native: NativeDevice::Vulkan {
            queue,
            device,
            adapter,
            instance,
        },
        hardware,
        capabilities,
    })
}
#[cfg(not(feature = "vulkan"))]
fn open_vulkan(_: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    Err(OpenError::BackendDisabled {
        backend: Backend::Vulkan,
    })
}

fn instance_descriptor(validation: Validation) -> wgpu_hal::InstanceDescriptor<'static> {
    wgpu_hal::InstanceDescriptor {
        name: "fluxel-rhi",
        flags: match validation {
            Validation::Disabled => wgt::InstanceFlags::empty(),
            Validation::Required => wgt::InstanceFlags::VALIDATION,
        },
        memory_budget_thresholds: wgt::MemoryBudgetThresholds::default(),
        backend_options: wgt::BackendOptions::default(),
        telemetry: None,
        display: None,
    }
}

#[cfg(feature = "dx12")]
fn dx12_validation_is_enabled(device: &wgpu_hal::dx12::Device) -> bool {
    use windows::{Win32::Graphics::Direct3D12::ID3D12InfoQueue, core::Interface as _};

    device.raw_device().cast::<ID3D12InfoQueue>().is_ok()
}

#[cfg(feature = "vulkan")]
fn vulkan_validation_is_available() -> bool {
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    let validation_features = c"VK_EXT_validation_features";
    // SAFETY: loading the process Vulkan loader performs no device operation;
    // failure is converted into an unavailable validation facility.
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return false;
    };
    // SAFETY: `entry` owns valid loader function pointers for the duration of
    // this enumeration call.
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    let has_layer = layers.iter().any(|layer| {
        layer
            .layer_name_as_c_str()
            .is_ok_and(|name| name == validation_layer)
    });
    if !has_layer {
        return false;
    }
    // SAFETY: `entry` owns valid loader function pointers and the layer name
    // remains alive for the duration of the enumeration call.
    let Ok(extensions) =
        (unsafe { entry.enumerate_instance_extension_properties(Some(validation_layer)) })
    else {
        return false;
    };
    extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|name| name == validation_features)
    })
}
fn native_error(backend: Backend, error: impl core::fmt::Display) -> OpenError {
    OpenError::NativeUnavailable {
        backend,
        reason: error.to_string(),
    }
}
fn required_limits(
    backend: Backend,
    capabilities: &wgpu_hal::Capabilities,
) -> Result<wgt::Limits, OpenError> {
    let requested = wgt::Limits::default();
    if requested.check_limits(&capabilities.limits) {
        Ok(requested)
    } else {
        Err(OpenError::RequiredLimitsUnavailable { backend })
    }
}
fn hardware(backend: Backend, info: &wgt::AdapterInfo) -> HardwareInfo {
    HardwareInfo {
        backend,
        name: info.name.clone(),
        vendor_id: info.vendor,
        device_id: info.device,
        kind: match info.device_type {
            wgt::DeviceType::Other => DeviceKind::Other,
            wgt::DeviceType::IntegratedGpu => DeviceKind::Integrated,
            wgt::DeviceType::DiscreteGpu => DeviceKind::Discrete,
            wgt::DeviceType::VirtualGpu => DeviceKind::Virtual,
            wgt::DeviceType::Cpu => DeviceKind::Cpu,
        },
        pci_bus_id: info.device_pci_bus_id.clone(),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
    }
}
fn capabilities(_: wgt::Features, capabilities: &wgpu_hal::Capabilities) -> HardwareCapabilities {
    HardwareCapabilities {
        max_texture_dimension_2d: capabilities.limits.max_texture_dimension_2d,
        max_bind_groups: capabilities.limits.max_bind_groups,
        min_uniform_buffer_offset_alignment: capabilities
            .limits
            .min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: capabilities
            .limits
            .min_storage_buffer_offset_alignment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_projection_exposes_storage_widening_exactly() {
        let read = BufferUsage::empty().with(BufferUsageKind::StorageRead);
        assert_eq!(buffer_usage_from_native(lower_buffer_usage(read)), read);

        let write = BufferUsage::empty().with(BufferUsageKind::StorageWrite);
        assert_eq!(
            buffer_usage_from_native(lower_buffer_usage(write)),
            BufferUsage::from_kinds([BufferUsageKind::StorageRead, BufferUsageKind::StorageWrite])
        );
    }

    #[test]
    fn texture_storage_projection_preserves_access_mode() {
        for usage in [
            TextureUsage::empty().with(TextureUsageKind::StorageRead),
            TextureUsage::empty().with(TextureUsageKind::StorageWrite),
            TextureUsage::from_kinds([
                TextureUsageKind::StorageRead,
                TextureUsageKind::StorageWrite,
            ]),
        ] {
            assert_eq!(
                texture_usage_from_native(lower_texture_usage(usage), TextureFormat::Rgba8Unorm),
                usage
            );
        }
    }
}
