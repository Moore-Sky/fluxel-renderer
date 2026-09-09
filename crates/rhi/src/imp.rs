//! Windows-only containment of the `wgpu-hal` bootstrap API.

#![cfg_attr(
    any(
        all(feature = "dx12", not(feature = "vulkan")),
        all(feature = "vulkan", not(feature = "dx12"))
    ),
    allow(
        dead_code,
        irrefutable_let_patterns,
        unreachable_patterns,
        unused_mut,
        unused_variables,
        reason = "single-backend builds intentionally collapse exhaustive native enums and omit the dual-backend hardware fixture"
    )
)]

use crate::{
    Backend, BufferDescriptor, DeviceKind, DeviceOptions, HardwareCapabilities, HardwareInfo,
    OpenError, ResourceCreateError, ResourceLease, TextureDescriptor, Validation,
};
use fluxel_rendergraph::{
    BufferCopyRegion, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus,
    ResourceAccessState, TextureAspect, TextureCopyRegion, TextureDesc, TextureDimension,
    TextureFormat, TextureRange, TextureUsage, TextureUsageKind,
};
use std::sync::Arc;
use wgpu_hal::{Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _};
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

pub(super) struct CopyEncoder {
    native: Option<NativeEncoder>,
    owner: Arc<OpenedDevice>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "native encoders are setup objects retained once per in-flight submission, not a dense collection"
)]
enum NativeEncoder {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::CommandEncoder),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::CommandEncoder),
}

pub(super) struct CopyCommandBuffer {
    native: Option<NativeFinished>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "the finished command buffer uniquely owns its native encoder until submission"
)]
enum NativeFinished {
    #[cfg(feature = "dx12")]
    Dx12 {
        owner: Arc<OpenedDevice>,
        encoder: wgpu_hal::dx12::CommandEncoder,
        command_buffer: wgpu_hal::dx12::CommandBuffer,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        owner: Arc<OpenedDevice>,
        encoder: wgpu_hal::vulkan::CommandEncoder,
        command_buffer: wgpu_hal::vulkan::CommandBuffer,
    },
}

#[derive(Clone)]
pub(super) struct NativeCompletion(Arc<std::sync::Mutex<NativeSubmission>>);

#[allow(
    clippy::large_enum_variant,
    reason = "one native submission bundle must retain the concrete encoder and command buffer together"
)]
enum NativeSubmission {
    #[cfg(feature = "dx12")]
    Dx12 {
        owner: Arc<OpenedDevice>,
        encoder: Option<wgpu_hal::dx12::CommandEncoder>,
        command_buffer: Option<wgpu_hal::dx12::CommandBuffer>,
        fence: Option<wgpu_hal::dx12::Fence>,
        leases: Vec<ResourceLease>,
        failure: Option<CompletionFailure>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        owner: Arc<OpenedDevice>,
        encoder: Option<wgpu_hal::vulkan::CommandEncoder>,
        command_buffer: Option<wgpu_hal::vulkan::CommandBuffer>,
        fence: Option<wgpu_hal::vulkan::Fence>,
        leases: Vec<ResourceLease>,
        failure: Option<CompletionFailure>,
    },
    TerminalFailure(CompletionFailure),
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
                // SAFETY: this buffer was created by this live device. Submission
                // bundles retain a strong resource lease through completion, so
                // the unique final owner cannot run while commands reference it.
                unsafe { device.destroy_buffer(buffer) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(buffer)) => {
                // SAFETY: same-device ownership and in-flight leases are retained;
                // the Option enforces one native destruction.
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
                // in-flight command bundles keep its resource lease alive.
                unsafe { device.destroy_texture(texture) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeTexture::Vulkan(texture)) => {
                // SAFETY: same-device ownership, completion retention, and unique
                // final destruction are enforced structurally.
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

impl Drop for CopyEncoder {
    fn drop(&mut self) {
        if let Some(mut native) = self.native.take() {
            // SAFETY: a live CopyEncoder is always recording; an unconsumed
            // encoder must discard exactly once and is never submitted.
            unsafe {
                match &mut native {
                    #[cfg(feature = "dx12")]
                    NativeEncoder::Dx12(encoder) => encoder.discard_encoding(),
                    #[cfg(feature = "vulkan")]
                    NativeEncoder::Vulkan(encoder) => encoder.discard_encoding(),
                }
            }
        }
    }
}

impl Drop for CopyCommandBuffer {
    fn drop(&mut self) {
        if let Some(finished) = self.native.take() {
            reset_finished(finished);
        }
    }
}

fn reset_finished(finished: NativeFinished) {
    match finished {
        #[cfg(feature = "dx12")]
        NativeFinished::Dx12 {
            mut encoder,
            command_buffer,
            ..
        } => {
            // SAFETY: encoder is closed, the command buffer was never accepted
            // or has completed, and it is the only live buffer from this encoder.
            unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
        }
        #[cfg(feature = "vulkan")]
        NativeFinished::Vulkan {
            mut encoder,
            command_buffer,
            ..
        } => {
            // SAFETY: same closed-encoder and complete/unsubmitted ownership
            // proof as the DX12 branch.
            unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
        }
    }
}

impl Drop for NativeSubmission {
    fn drop(&mut self) {
        match self {
            #[cfg(feature = "dx12")]
            Self::Dx12 {
                owner,
                encoder,
                command_buffer,
                fence,
                leases,
                failure,
            } => {
                let (
                    NativeDevice::Dx12 { device, .. },
                    Some(mut encoder),
                    Some(command_buffer),
                    Some(fence),
                ) = (
                    &owner.native,
                    encoder.take(),
                    command_buffer.take(),
                    fence.take(),
                )
                else {
                    return;
                };
                // A submit/query failure can mean ExecuteCommandLists was accepted
                // before fence signaling failed. In that state no fence value can
                // prove completion. Retain the complete native/resource bundle for
                // process lifetime rather than resetting storage still in use.
                if failure.is_some() {
                    let retained = std::mem::take(leases);
                    std::mem::forget((owner.clone(), encoder, command_buffer, fence, retained));
                } else if unsafe { device.wait(&fence, 1, None) }.is_ok() {
                    // SAFETY: the sole command buffer completed at fence value 1.
                    unsafe {
                        encoder.reset_all(core::iter::once(command_buffer));
                        device.destroy_fence(fence);
                    }
                } else {
                    let retained = std::mem::take(leases);
                    std::mem::forget((owner.clone(), encoder, command_buffer, fence, retained));
                }
            }
            #[cfg(feature = "vulkan")]
            Self::Vulkan {
                owner,
                encoder,
                command_buffer,
                fence,
                leases,
                failure,
            } => {
                let (
                    NativeDevice::Vulkan { device, .. },
                    Some(mut encoder),
                    Some(command_buffer),
                    Some(fence),
                ) = (
                    &owner.native,
                    encoder.take(),
                    command_buffer.take(),
                    fence.take(),
                )
                else {
                    return;
                };
                if failure.is_some() {
                    let retained = std::mem::take(leases);
                    std::mem::forget((owner.clone(), encoder, command_buffer, fence, retained));
                } else if unsafe { device.wait(&fence, 1, None) }.is_ok() {
                    // SAFETY: the sole command buffer completed at fence value 1.
                    unsafe {
                        encoder.reset_all(core::iter::once(command_buffer));
                        device.destroy_fence(fence);
                    }
                } else {
                    let retained = std::mem::take(leases);
                    std::mem::forget((owner.clone(), encoder, command_buffer, fence, retained));
                }
            }
            Self::TerminalFailure(_) => {}
        }
    }
}

pub(super) fn begin_copy_encoder(owner: &Arc<OpenedDevice>) -> Result<CopyEncoder, String> {
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, queue, .. } => {
            let desc = wgpu_hal::CommandEncoderDescriptor {
                label: Some("fluxel copy graph"),
                queue,
            };
            // SAFETY: queue and device were opened together and remain retained.
            let mut encoder =
                unsafe { device.create_command_encoder(&desc) }.map_err(|e| e.to_string())?;
            // SAFETY: a freshly created encoder is closed.
            unsafe { encoder.begin_encoding(Some("fluxel copy graph")) }
                .map_err(|e| e.to_string())?;
            NativeEncoder::Dx12(encoder)
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, queue, .. } => {
            let desc = wgpu_hal::CommandEncoderDescriptor {
                label: Some("fluxel copy graph"),
                queue,
            };
            // SAFETY: queue and device share the retained adapter/device lineage.
            let mut encoder =
                unsafe { device.create_command_encoder(&desc) }.map_err(|e| e.to_string())?;
            // SAFETY: a freshly created encoder is closed.
            unsafe { encoder.begin_encoding(Some("fluxel copy graph")) }
                .map_err(|e| e.to_string())?;
            NativeEncoder::Vulkan(encoder)
        }
    };
    Ok(CopyEncoder {
        native: Some(native),
        owner: Arc::clone(owner),
    })
}

pub(super) fn transition_buffer(
    encoder: &mut CopyEncoder,
    buffer: &OwnedBuffer,
    before: ResourceAccessState,
    after: ResourceAccessState,
) -> Result<(), String> {
    let from = buffer_state(before)?;
    let to = buffer_state(after)?;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        buffer.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeBuffer::Dx12(buffer)) => unsafe {
            encoder.transition_buffers(core::iter::once(wgpu_hal::BufferBarrier {
                buffer,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeBuffer::Vulkan(buffer)) => unsafe {
            encoder.transition_buffers(core::iter::once(wgpu_hal::BufferBarrier {
                buffer,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        _ => return Err("buffer and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(super) fn transition_texture(
    encoder: &mut CopyEncoder,
    texture: &OwnedTexture,
    descriptor: TextureDesc,
    range: TextureRange,
    before: ResourceAccessState,
    after: ResourceAccessState,
) -> Result<(), String> {
    let from = texture_state(before)?;
    let to = texture_state(after)?;
    let range = texture_range(descriptor, range)?;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        texture.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeTexture::Dx12(texture)) => unsafe {
            encoder.transition_textures(core::iter::once(wgpu_hal::TextureBarrier {
                texture,
                range,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeTexture::Vulkan(texture)) => unsafe {
            encoder.transition_textures(core::iter::once(wgpu_hal::TextureBarrier {
                texture,
                range,
                usage: wgpu_hal::StateTransition { from, to },
            }));
        },
        _ => return Err("texture and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(super) fn copy_buffer(
    encoder: &mut CopyEncoder,
    source: &OwnedBuffer,
    destination: &OwnedBuffer,
    region: BufferCopyRegion,
) -> Result<(), String> {
    let size =
        wgt::BufferSize::new(region.size).ok_or_else(|| "copy size must be non-zero".to_owned())?;
    let copy = wgpu_hal::BufferCopy {
        src_offset: region.source_offset,
        dst_offset: region.destination_offset,
        size,
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live buffer"),
        destination.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeBuffer::Dx12(source),
            NativeBuffer::Dx12(destination),
        ) => unsafe {
            encoder.copy_buffer_to_buffer(source, destination, core::iter::once(copy));
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeBuffer::Vulkan(source),
            NativeBuffer::Vulkan(destination),
        ) => unsafe {
            encoder.copy_buffer_to_buffer(source, destination, core::iter::once(copy));
        },
        _ => return Err("copy buffers and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(super) fn copy_texture(
    encoder: &mut CopyEncoder,
    source: &OwnedTexture,
    destination: &OwnedTexture,
    descriptor: TextureDesc,
    region: TextureCopyRegion,
) -> Result<(), String> {
    let copy = wgpu_hal::TextureCopy {
        src_base: copy_base(region.source_mip_level, region.source_origin),
        dst_base: copy_base(region.destination_mip_level, region.destination_origin),
        size: wgpu_hal::CopyExtent {
            width: region.extent[0],
            height: region.extent[1],
            depth: region.extent[2],
        },
    };
    let source_usage = wgt::TextureUses::COPY_SRC;
    let _ = descriptor;
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live texture"),
        destination.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeTexture::Dx12(source),
            NativeTexture::Dx12(destination),
        ) => unsafe {
            encoder.copy_texture_to_texture(
                source,
                source_usage,
                destination,
                core::iter::once(copy),
            );
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeTexture::Vulkan(source),
            NativeTexture::Vulkan(destination),
        ) => unsafe {
            encoder.copy_texture_to_texture(
                source,
                source_usage,
                destination,
                core::iter::once(copy),
            );
        },
        _ => return Err("copy textures and encoder backend mismatch".into()),
    }
    Ok(())
}

pub(super) fn finish_copy_encoder(mut encoder: CopyEncoder) -> Result<CopyCommandBuffer, String> {
    let native = encoder.native.take().expect("recording encoder");
    let finished = match native {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(mut native) => {
            // SAFETY: this encoder is recording and no prior error was retained.
            let command_buffer = unsafe { native.end_encoding() }.map_err(|e| e.to_string())?;
            NativeFinished::Dx12 {
                owner: Arc::clone(&encoder.owner),
                encoder: native,
                command_buffer,
            }
        }
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(mut native) => {
            // SAFETY: this encoder is recording and no prior error was retained.
            let command_buffer = unsafe { native.end_encoding() }.map_err(|e| e.to_string())?;
            NativeFinished::Vulkan {
                owner: Arc::clone(&encoder.owner),
                encoder: native,
                command_buffer,
            }
        }
    };
    Ok(CopyCommandBuffer {
        native: Some(finished),
    })
}

pub(super) fn submit_copy(
    mut buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
) -> Result<NativeCompletion, String> {
    let finished = buffer.native.take().expect("finished command buffer");
    let submission = match finished {
        #[cfg(feature = "dx12")]
        NativeFinished::Dx12 {
            owner,
            mut encoder,
            command_buffer,
        } => {
            let NativeDevice::Dx12 { device, queue, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: device is live and owns the new unsignaled fence.
            let fence = unsafe { device.create_fence() }.map_err(|e| e.to_string())?;
            // SAFETY: command buffer and encoder belong to this device/queue and
            // remain owned by the returned completion until fence value 1.
            match unsafe { queue.submit(&[&command_buffer], &[], (&fence, 1)) } {
                Ok(()) => NativeSubmission::Dx12 {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(fence),
                    leases,
                    failure: None,
                },
                Err(error) => {
                    let failure = completion_failure(&error);
                    // DX12 may execute command lists before a later fence signal
                    // reports this error. Only a successful queue-idle wait makes
                    // immediate command allocator reuse legal.
                    if unsafe { queue.wait_for_idle() }.is_ok() {
                        unsafe {
                            encoder.reset_all(core::iter::once(command_buffer));
                            device.destroy_fence(fence);
                        }
                        NativeSubmission::TerminalFailure(failure)
                    } else {
                        NativeSubmission::Dx12 {
                            owner,
                            encoder: Some(encoder),
                            command_buffer: Some(command_buffer),
                            fence: Some(fence),
                            leases,
                            failure: Some(CompletionFailure::DeviceLost),
                        }
                    }
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeFinished::Vulkan {
            owner,
            mut encoder,
            command_buffer,
        } => {
            let NativeDevice::Vulkan { device, queue, .. } = &owner.native else {
                unreachable!()
            };
            // SAFETY: device is live and owns the new unsignaled fence.
            let fence = unsafe { device.create_fence() }.map_err(|e| e.to_string())?;
            // SAFETY: all submitted objects remain retained to completion.
            match unsafe { queue.submit(&[&command_buffer], &[], (&fence, 1)) } {
                Ok(()) => NativeSubmission::Vulkan {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(fence),
                    leases,
                    failure: None,
                },
                Err(error) => {
                    let failure = completion_failure(&error);
                    if unsafe { queue.wait_for_idle() }.is_ok() {
                        unsafe {
                            encoder.reset_all(core::iter::once(command_buffer));
                            device.destroy_fence(fence);
                        }
                        NativeSubmission::TerminalFailure(failure)
                    } else {
                        NativeSubmission::Vulkan {
                            owner,
                            encoder: Some(encoder),
                            command_buffer: Some(command_buffer),
                            fence: Some(fence),
                            leases,
                            failure: Some(CompletionFailure::DeviceLost),
                        }
                    }
                }
            }
        }
    };
    Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
        submission,
    ))))
}

fn completion_failure(error: &wgpu_hal::DeviceError) -> CompletionFailure {
    if *error == wgpu_hal::DeviceError::Lost {
        CompletionFailure::DeviceLost
    } else {
        CompletionFailure::ExecutionFailed
    }
}

pub(super) fn completion_status(completion: &NativeCompletion) -> Result<CompletionStatus, String> {
    let mut submission = completion
        .0
        .lock()
        .map_err(|_| "completion lock poisoned".to_owned())?;
    let value = match &mut *submission {
        #[cfg(feature = "dx12")]
        NativeSubmission::Dx12 {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Dx12 { device, .. } = &owner.native else {
                unreachable!()
            };
            match unsafe { device.get_fence_value(fence) } {
                Ok(value) => value,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeSubmission::Vulkan {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Vulkan { device, .. } = &owner.native else {
                unreachable!()
            };
            match unsafe { device.get_fence_value(fence) } {
                Ok(value) => value,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        NativeSubmission::TerminalFailure(failure) => {
            return Ok(CompletionStatus::Failed(*failure));
        }
        _ => {
            return Ok(CompletionStatus::Failed(
                fluxel_rendergraph::CompletionFailure::ExecutionFailed,
            ));
        }
    };
    Ok(if value >= 1 {
        CompletionStatus::Complete
    } else {
        CompletionStatus::Pending
    })
}

pub(super) fn wait_completion(
    completion: &NativeCompletion,
    timeout: core::time::Duration,
) -> Result<CompletionStatus, String> {
    let mut submission = completion
        .0
        .lock()
        .map_err(|_| "completion lock poisoned".to_owned())?;
    let complete = match &mut *submission {
        #[cfg(feature = "dx12")]
        NativeSubmission::Dx12 {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Dx12 { device, .. } = &owner.native else {
                unreachable!()
            };
            match unsafe { device.wait(fence, 1, Some(timeout)) } {
                Ok(complete) => complete,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        #[cfg(feature = "vulkan")]
        NativeSubmission::Vulkan {
            owner,
            fence: Some(fence),
            failure,
            ..
        } => {
            if let Some(failure) = *failure {
                return Ok(CompletionStatus::Failed(failure));
            }
            let NativeDevice::Vulkan { device, .. } = &owner.native else {
                unreachable!()
            };
            match unsafe { device.wait(fence, 1, Some(timeout)) } {
                Ok(complete) => complete,
                Err(error) => {
                    let reason = completion_failure(&error);
                    *failure = Some(reason);
                    return Ok(CompletionStatus::Failed(reason));
                }
            }
        }
        NativeSubmission::TerminalFailure(failure) => {
            return Ok(CompletionStatus::Failed(*failure));
        }
        _ => {
            return Ok(CompletionStatus::Failed(
                fluxel_rendergraph::CompletionFailure::ExecutionFailed,
            ));
        }
    };
    Ok(if complete {
        CompletionStatus::Complete
    } else {
        CompletionStatus::Pending
    })
}

#[cfg(test)]
struct ValidationLogger;

#[cfg(test)]
static VALIDATION_LOGGER: ValidationLogger = ValidationLogger;
#[cfg(test)]
static VALIDATION_MESSAGES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
#[cfg(test)]
static VALIDATION_LOGGER_INIT: std::sync::Once = std::sync::Once::new();

#[cfg(test)]
impl log::Log for ValidationLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            VALIDATION_MESSAGES
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(format!("{}: {}", record.target(), record.args()));
        }
    }
    fn flush(&self) {}
}

#[cfg(test)]
pub(super) fn initialize_validation_capture() {
    VALIDATION_LOGGER_INIT.call_once(|| {
        let _ = log::set_logger(&VALIDATION_LOGGER);
        // wgpu-hal chooses Vulkan debug-utils severities from this global
        // filter while creating the instance, so Warn must be enabled before
        // Device::open to capture validation warnings as well as errors.
        log::set_max_level(log::LevelFilter::Warn);
    });
}

#[cfg(test)]
pub(super) fn clear_validation_diagnostics(owner: &Arc<OpenedDevice>) {
    VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
    #[cfg(feature = "dx12")]
    if let NativeDevice::Dx12 { device, .. } = &owner.native {
        use windows::{Win32::Graphics::Direct3D12::ID3D12InfoQueue, core::Interface as _};
        if let Ok(queue) = device.raw_device().cast::<ID3D12InfoQueue>() {
            // SAFETY: the information queue belongs to the retained live device.
            unsafe { queue.ClearStoredMessages() };
        }
    }
}

#[cfg(test)]
pub(super) fn validation_diagnostics(owner: &Arc<OpenedDevice>) -> Vec<String> {
    let mut messages = VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    #[cfg(feature = "dx12")]
    if let NativeDevice::Dx12 { device, .. } = &owner.native {
        use windows::{Win32::Graphics::Direct3D12::ID3D12InfoQueue, core::Interface as _};
        if let Ok(queue) = device.raw_device().cast::<ID3D12InfoQueue>() {
            // SAFETY: read-only count query on the retained device queue.
            let count = unsafe { queue.GetNumStoredMessagesAllowedByRetrievalFilter() };
            if count != 0 {
                messages.push(format!("DX12 info queue stored {count} message(s)"));
            }
        }
    }
    messages
}

#[cfg(test)]
pub(super) fn upload_buffer_for_test(
    owner: &Arc<OpenedDevice>,
    target: &OwnedBuffer,
    target_lease: ResourceLease,
    bytes: &[u8],
) -> Result<ResourceAccessState, String> {
    let staging = create_staging_buffer(
        owner,
        bytes.len() as u64,
        wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
    )?;
    with_mapped_write(&staging, bytes)?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopySource,
    )?;
    transition_buffer(
        &mut encoder,
        target,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    copy_buffer(
        &mut encoder,
        &staging,
        target,
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: bytes.len() as u64,
        },
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![target_lease])?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        if matches!(status, CompletionStatus::Failed(_)) {
            std::mem::forget(staging);
        }
        return Err(format!("upload did not complete: {status:?}"));
    }
    drop(completion);
    drop(staging);
    Ok(ResourceAccessState::CopyDestination)
}

#[cfg(test)]
pub(super) fn readback_buffer_for_test(
    owner: &Arc<OpenedDevice>,
    source: &OwnedBuffer,
    source_lease: ResourceLease,
    incoming_state: ResourceAccessState,
    size: u64,
) -> Result<Vec<u8>, String> {
    let staging = create_staging_buffer(
        owner,
        size,
        wgt::BufferUses::MAP_READ | wgt::BufferUses::COPY_DST,
    )?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_buffer(
        &mut encoder,
        source,
        incoming_state,
        ResourceAccessState::CopySource,
    )?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    copy_buffer(
        &mut encoder,
        source,
        &staging,
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size,
        },
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![source_lease])?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        if matches!(status, CompletionStatus::Failed(_)) {
            std::mem::forget(staging);
        }
        return Err(format!("readback did not complete: {status:?}"));
    }
    drop(completion);
    let bytes = with_mapped_read(&staging, size)?;
    drop(staging);
    Ok(bytes)
}

#[cfg(test)]
pub(super) struct TextureReadback {
    pub(super) tight: Vec<u8>,
    pub(super) padded: Vec<u8>,
    pub(super) bytes_per_row: u32,
}

#[cfg(test)]
pub(super) fn upload_texture_for_test(
    owner: &Arc<OpenedDevice>,
    target: &OwnedTexture,
    target_lease: ResourceLease,
    descriptor: TextureDesc,
    tight: &[u8],
) -> Result<ResourceAccessState, String> {
    let row_bytes = descriptor
        .extent
        .width
        .checked_mul(4)
        .ok_or("row size overflow")?;
    let pitch = row_bytes.next_multiple_of(256);
    let staging_size = u64::from(pitch) * u64::from(descriptor.extent.height);
    if tight.len() as u64 != u64::from(row_bytes) * u64::from(descriptor.extent.height) {
        return Err("tight texture upload length mismatch".into());
    }
    let mut padded = vec![0xEE; staging_size as usize];
    for row in 0..descriptor.extent.height as usize {
        padded[row * pitch as usize..row * pitch as usize + row_bytes as usize]
            .copy_from_slice(&tight[row * row_bytes as usize..(row + 1) * row_bytes as usize]);
    }
    let staging = create_staging_buffer(
        owner,
        staging_size,
        wgt::BufferUses::MAP_WRITE | wgt::BufferUses::COPY_SRC,
    )?;
    with_mapped_write(&staging, &padded)?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopySource,
    )?;
    transition_texture(
        &mut encoder,
        target,
        descriptor,
        TextureRange::Whole,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    copy_buffer_to_texture(
        &mut encoder,
        &staging,
        target,
        pitch,
        descriptor.extent.width,
        descriptor.extent.height,
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![target_lease])?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        if matches!(status, CompletionStatus::Failed(_)) {
            std::mem::forget(staging);
        }
        return Err(format!("texture upload did not complete: {status:?}"));
    }
    drop(completion);
    drop(staging);
    Ok(ResourceAccessState::CopyDestination)
}

#[cfg(test)]
pub(super) fn readback_texture_for_test(
    owner: &Arc<OpenedDevice>,
    source: &OwnedTexture,
    source_lease: ResourceLease,
    descriptor: TextureDesc,
    incoming_state: ResourceAccessState,
) -> Result<TextureReadback, String> {
    let row_bytes = descriptor
        .extent
        .width
        .checked_mul(4)
        .ok_or("row size overflow")?;
    let pitch = row_bytes.next_multiple_of(256);
    let staging_size = u64::from(pitch) * u64::from(descriptor.extent.height);
    let staging = create_staging_buffer(
        owner,
        staging_size,
        wgt::BufferUses::MAP_READ | wgt::BufferUses::COPY_DST,
    )?;
    let mut encoder = begin_copy_encoder(owner)?;
    transition_texture(
        &mut encoder,
        source,
        descriptor,
        TextureRange::Whole,
        incoming_state,
        ResourceAccessState::CopySource,
    )?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::Undefined,
        ResourceAccessState::CopyDestination,
    )?;
    clear_buffer(&mut encoder, &staging, staging_size)?;
    transition_buffer(
        &mut encoder,
        &staging,
        ResourceAccessState::CopyDestination,
        ResourceAccessState::CopyDestination,
    )?;
    copy_texture_to_buffer(
        &mut encoder,
        source,
        &staging,
        pitch,
        descriptor.extent.width,
        descriptor.extent.height,
    )?;
    let completion = submit_copy(finish_copy_encoder(encoder)?, vec![source_lease])?;
    let status = wait_completion(&completion, core::time::Duration::from_secs(10))?;
    if status != CompletionStatus::Complete {
        if matches!(status, CompletionStatus::Failed(_)) {
            std::mem::forget(staging);
        }
        return Err(format!("texture readback did not complete: {status:?}"));
    }
    drop(completion);
    let padded = with_mapped_read(&staging, staging_size)?;
    let mut tight =
        Vec::with_capacity((u64::from(row_bytes) * u64::from(descriptor.extent.height)) as usize);
    for row in 0..descriptor.extent.height as usize {
        tight.extend_from_slice(
            &padded[row * pitch as usize..row * pitch as usize + row_bytes as usize],
        );
    }
    drop(staging);
    Ok(TextureReadback {
        tight,
        padded,
        bytes_per_row: pitch,
    })
}

#[cfg(test)]
fn copy_buffer_to_texture(
    encoder: &mut CopyEncoder,
    source: &OwnedBuffer,
    destination: &OwnedTexture,
    pitch: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let copy = wgpu_hal::BufferTextureCopy {
        buffer_layout: wgt::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pitch),
            rows_per_image: Some(height),
        },
        texture_base: copy_base(0, [0, 0, 0]),
        size: wgpu_hal::CopyExtent {
            width,
            height,
            depth: 1,
        },
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live buffer"),
        destination.native.as_ref().expect("live texture"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeBuffer::Dx12(source),
            NativeTexture::Dx12(destination),
        ) => unsafe {
            encoder.copy_buffer_to_texture(source, destination, core::iter::once(copy));
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeBuffer::Vulkan(source),
            NativeTexture::Vulkan(destination),
        ) => unsafe {
            encoder.copy_buffer_to_texture(source, destination, core::iter::once(copy));
        },
        _ => return Err("buffer-to-texture backend mismatch".into()),
    }
    Ok(())
}

#[cfg(test)]
fn copy_texture_to_buffer(
    encoder: &mut CopyEncoder,
    source: &OwnedTexture,
    destination: &OwnedBuffer,
    pitch: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let copy = wgpu_hal::BufferTextureCopy {
        buffer_layout: wgt::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pitch),
            rows_per_image: Some(height),
        },
        texture_base: copy_base(0, [0, 0, 0]),
        size: wgpu_hal::CopyExtent {
            width,
            height,
            depth: 1,
        },
    };
    match (
        encoder.native.as_mut().expect("recording encoder"),
        source.native.as_ref().expect("live texture"),
        destination.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            NativeTexture::Dx12(source),
            NativeBuffer::Dx12(destination),
        ) => unsafe {
            encoder.copy_texture_to_buffer(
                source,
                wgt::TextureUses::COPY_SRC,
                destination,
                core::iter::once(copy),
            );
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            NativeTexture::Vulkan(source),
            NativeBuffer::Vulkan(destination),
        ) => unsafe {
            encoder.copy_texture_to_buffer(
                source,
                wgt::TextureUses::COPY_SRC,
                destination,
                core::iter::once(copy),
            );
        },
        _ => return Err("texture-to-buffer backend mismatch".into()),
    }
    Ok(())
}

#[cfg(test)]
fn clear_buffer(encoder: &mut CopyEncoder, buffer: &OwnedBuffer, size: u64) -> Result<(), String> {
    match (
        encoder.native.as_mut().expect("recording encoder"),
        buffer.native.as_ref().expect("live buffer"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), NativeBuffer::Dx12(buffer)) => unsafe {
            encoder.clear_buffer(buffer, 0..size);
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), NativeBuffer::Vulkan(buffer)) => unsafe {
            encoder.clear_buffer(buffer, 0..size);
        },
        _ => return Err("clear buffer backend mismatch".into()),
    }
    Ok(())
}

#[cfg(test)]
fn create_staging_buffer(
    owner: &Arc<OpenedDevice>,
    size: u64,
    usage: wgt::BufferUses,
) -> Result<OwnedBuffer, String> {
    let desc = wgpu_hal::BufferDescriptor {
        label: Some("fluxel test staging"),
        size,
        usage,
        memory_flags: wgpu_hal::MemoryFlags::PREFER_COHERENT,
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => NativeBuffer::Dx12(
            // SAFETY: non-zero in-bounds fixture size and valid staging flags.
            unsafe { device.create_buffer(&desc) }.map_err(|e| e.to_string())?,
        ),
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => NativeBuffer::Vulkan(
            // SAFETY: non-zero in-bounds fixture size and valid staging flags.
            unsafe { device.create_buffer(&desc) }.map_err(|e| e.to_string())?,
        ),
    };
    Ok(OwnedBuffer {
        native: Some(native),
        owner: Arc::clone(owner),
    })
}

#[cfg(test)]
fn with_mapped_write(buffer: &OwnedBuffer, bytes: &[u8]) -> Result<(), String> {
    let range = 0..bytes.len() as u64;
    match (
        &buffer.owner.native,
        buffer.native.as_ref().expect("live staging"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(native)) => unsafe {
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), mapping.ptr.as_ptr(), bytes.len());
            if !mapping.is_coherent {
                device.flush_mapped_ranges(native, core::iter::once(range));
            }
            device.unmap_buffer(native);
        },
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(native)) => unsafe {
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), mapping.ptr.as_ptr(), bytes.len());
            if !mapping.is_coherent {
                device.flush_mapped_ranges(native, core::iter::once(range));
            }
            device.unmap_buffer(native);
        },
        _ => return Err("staging backend mismatch".into()),
    }
    Ok(())
}

#[cfg(test)]
fn with_mapped_read(buffer: &OwnedBuffer, size: u64) -> Result<Vec<u8>, String> {
    let range = 0..size;
    let mut result = vec![0; size as usize];
    match (
        &buffer.owner.native,
        buffer.native.as_ref().expect("live staging"),
    ) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(native)) => unsafe {
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            if !mapping.is_coherent {
                device.invalidate_mapped_ranges(native, core::iter::once(range));
            }
            core::ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), result.as_mut_ptr(), result.len());
            device.unmap_buffer(native);
        },
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(native)) => unsafe {
            let mapping = device
                .map_buffer(native, range.clone())
                .map_err(|e| e.to_string())?;
            if !mapping.is_coherent {
                device.invalidate_mapped_ranges(native, core::iter::once(range));
            }
            core::ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), result.as_mut_ptr(), result.len());
            device.unmap_buffer(native);
        },
        _ => return Err("staging backend mismatch".into()),
    }
    Ok(result)
}

fn copy_base(mip_level: u32, origin: [u32; 3]) -> wgpu_hal::TextureCopyBase {
    wgpu_hal::TextureCopyBase {
        mip_level,
        array_layer: 0,
        origin: wgt::Origin3d {
            x: origin[0],
            y: origin[1],
            z: origin[2],
        },
        aspect: wgpu_hal::FormatAspects::COLOR,
    }
}

fn texture_range(
    descriptor: TextureDesc,
    range: TextureRange,
) -> Result<wgt::ImageSubresourceRange, String> {
    Ok(match range {
        TextureRange::Whole => wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(descriptor.mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(descriptor.array_layers),
        },
        TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        } => wgt::ImageSubresourceRange {
            aspect: match aspect {
                TextureAspect::Color => wgt::TextureAspect::All,
                TextureAspect::Depth => wgt::TextureAspect::DepthOnly,
                TextureAspect::Stencil => wgt::TextureAspect::StencilOnly,
                _ => return Err("unsupported texture aspect".into()),
            },
            base_mip_level,
            mip_level_count: Some(mip_level_count),
            base_array_layer,
            array_layer_count: Some(array_layer_count),
        },
    })
}

fn buffer_state(state: ResourceAccessState) -> Result<wgt::BufferUses, String> {
    Ok(match state {
        ResourceAccessState::Undefined => wgt::BufferUses::empty(),
        ResourceAccessState::ShaderStorageRead => wgt::BufferUses::STORAGE_READ_ONLY,
        ResourceAccessState::ShaderStorageWrite | ResourceAccessState::ShaderStorageReadWrite => {
            wgt::BufferUses::STORAGE_READ_WRITE
        }
        ResourceAccessState::UniformRead => wgt::BufferUses::UNIFORM,
        ResourceAccessState::VertexRead => wgt::BufferUses::VERTEX,
        ResourceAccessState::IndexRead => wgt::BufferUses::INDEX,
        ResourceAccessState::IndirectRead => wgt::BufferUses::INDIRECT,
        ResourceAccessState::CopySource => wgt::BufferUses::COPY_SRC,
        ResourceAccessState::CopyDestination => wgt::BufferUses::COPY_DST,
        _ => return Err("texture-only state used for buffer transition".into()),
    })
}

fn texture_state(state: ResourceAccessState) -> Result<wgt::TextureUses, String> {
    Ok(match state {
        ResourceAccessState::Undefined => wgt::TextureUses::UNINITIALIZED,
        ResourceAccessState::ColorAttachmentRead
        | ResourceAccessState::ColorAttachmentWrite
        | ResourceAccessState::ColorAttachmentReadWrite => wgt::TextureUses::COLOR_TARGET,
        ResourceAccessState::DepthStencilRead => wgt::TextureUses::DEPTH_STENCIL_READ,
        ResourceAccessState::DepthStencilWrite | ResourceAccessState::DepthStencilReadWrite => {
            wgt::TextureUses::DEPTH_STENCIL_WRITE
        }
        ResourceAccessState::ShaderSampledRead => wgt::TextureUses::RESOURCE,
        ResourceAccessState::ShaderStorageRead => wgt::TextureUses::STORAGE_READ_ONLY,
        ResourceAccessState::ShaderStorageWrite => wgt::TextureUses::STORAGE_WRITE_ONLY,
        ResourceAccessState::ShaderStorageReadWrite => wgt::TextureUses::STORAGE_READ_WRITE,
        ResourceAccessState::CopySource => wgt::TextureUses::COPY_SRC,
        ResourceAccessState::CopyDestination => wgt::TextureUses::COPY_DST,
        ResourceAccessState::Present => wgt::TextureUses::PRESENT,
        _ => return Err("buffer-only state used for texture transition".into()),
    })
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
