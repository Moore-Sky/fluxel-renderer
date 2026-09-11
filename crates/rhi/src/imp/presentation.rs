//! Private DX12 surface creation, acquisition, and presentation lowering.
//!
//! Surface objects never cross this module boundary.  A graph sees only an
//! owned texture wrapper and an opaque one-shot token; the token retains the
//! actual acquired swapchain image until it is consumed by `present` or
//! discarded on drop by the safe façade.

use super::*;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use wgpu_hal::Surface as _;

pub(crate) struct NativePresentationToken {
    pub(crate) owner: Arc<OpenedDevice>,
    pub(crate) surface: Arc<Mutex<NativeSurface>>,
    pub(crate) acquired: Option<NativeAcquiredSurfaceTexture>,
    pub(crate) fence: Option<NativeSurfaceFence>,
    /// Stays true from acquire until the native image has been discarded or
    /// the accepted submission's complete bundle has been retired.
    pub(crate) live_frame: Arc<std::sync::atomic::AtomicBool>,
    /// Type-erased `Arc<W>` from the public façade. It is intentionally held
    /// by the token because accepted-unknown quarantine can outlive Surface.
    pub(crate) _window: Arc<dyn std::any::Any>,
}

impl Drop for NativePresentationToken {
    fn drop(&mut self) {
        if let Some(acquired) = self.acquired.take() {
            let mut surface = self
                .surface
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match (&mut *surface, acquired) {
                #[cfg(feature = "dx12")]
                (NativeSurface::Dx12(surface), NativeAcquiredSurfaceTexture::Dx12(texture)) => unsafe {
                    // SAFETY: an unconsumed token owns the sole acquired image.
                    surface.discard_texture(texture);
                },
            }
        }
        if let Some(NativeSurfaceFence::Dx12(fence)) = self.fence.take() {
            if let NativeDevice::Dx12 { device, .. } = &self.owner.native {
                // No submit can still reference this fence when its token drops.
                unsafe { device.destroy_fence(fence) };
            }
        }
        self.live_frame
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

pub(crate) enum NativeAcquiredSurfaceTexture {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Texture),
}

pub(crate) enum NativeSurfaceFence {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Fence),
}

pub(crate) fn open_dx12_surface(
    options: DeviceOptions,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(OpenedDevice, NativeSurface), OpenError> {
    #[cfg(feature = "dx12")]
    {
        let descriptor = instance_descriptor(options.validation);
        // SAFETY: raw handles are borrowed from the caller's live window and
        // the returned safe surface lifetime is tied to that window by the
        // public façade.
        let instance = unsafe { wgpu_hal::dx12::Instance::init(&descriptor) }
            .map_err(|e| native_error(Backend::Dx12, e))?;
        // SAFETY: the instance and native window handles remain live while
        // surface creation performs no GPU work.
        let surface = unsafe { instance.create_surface(display, window) }
            .map_err(|e| native_error(Backend::Dx12, e))?;
        // SAFETY: the surface belongs to this live instance and constrains
        // enumeration to adapters which can present to this window.
        let adapters = unsafe { instance.enumerate_adapters(Some(&surface)) };
        let available_adapters = adapters.len();
        let exposed = adapters.into_iter().nth(options.adapter_index).ok_or(
            OpenError::AdapterUnavailable {
                backend: Backend::Dx12,
                adapter_index: options.adapter_index,
                available_adapters,
            },
        )?;
        // SAFETY: `surface` was created by this still-live `instance` from the
        // caller's live window handles. No configuration, acquisition, or
        // destruction can race this read-only compatibility query.
        if unsafe { exposed.adapter.surface_capabilities(&surface) }.is_none() {
            return Err(OpenError::NativeUnavailable {
                backend: Backend::Dx12,
                reason: "selected adapter cannot present to the supplied window".into(),
            });
        }
        let hardware = hardware(Backend::Dx12, &exposed.info);
        let rgba8_unorm_filterable = rgba8_unorm_filterable(&exposed.adapter);
        let rgba8_unorm_srgb_filterable = rgba8_unorm_srgb_filterable(&exposed.adapter);
        let capabilities = capabilities(
            exposed.features,
            &exposed.capabilities,
            rgba8_unorm_filterable,
            rgba8_unorm_srgb_filterable,
        );
        let requested_limits = required_limits(Backend::Dx12, &exposed.capabilities)?;
        let adapter = exposed.adapter;
        // SAFETY: requested limits were checked against this exact compatible
        // adapter; features remain intentionally empty for this fixed slice.
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
        Ok((
            OpenedDevice {
                native: NativeDevice::Dx12 {
                    queue,
                    device,
                    adapter,
                    instance,
                },
                queue_operations: Mutex::new(()),
                hardware,
                capabilities,
            },
            NativeSurface::Dx12(surface),
        ))
    }
    #[cfg(not(feature = "dx12"))]
    {
        let _ = (options, display, window);
        Err(OpenError::BackendDisabled {
            backend: Backend::Dx12,
        })
    }
}

pub(crate) fn configure_dx12_surface(
    owner: &Arc<OpenedDevice>,
    surface: &Mutex<NativeSurface>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut surface = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *surface) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeSurface::Dx12(surface)) => {
            let config = wgpu_hal::SurfaceConfiguration {
                maximum_frame_latency: 2,
                present_mode: wgt::PresentMode::Fifo,
                composite_alpha_mode: wgt::CompositeAlphaMode::Opaque,
                format: wgt::TextureFormat::Rgba8Unorm,
                color_space: wgt::SurfaceColorSpace::Srgb,
                extent: wgt::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                usage: wgt::TextureUses::COLOR_TARGET,
                view_formats: Vec::new(),
            };
            // SAFETY: the safe surface façade serializes configure with acquire
            // and requires prior frame shutdown before reconfiguration.
            unsafe { surface.configure(device, &config) }.map_err(|e| e.to_string())
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}

pub(crate) fn acquire_dx12_surface(
    owner: &Arc<OpenedDevice>,
    surface: Arc<Mutex<NativeSurface>>,
    window: Arc<dyn std::any::Any>,
    live_frame: Arc<std::sync::atomic::AtomicBool>,
    descriptor: TextureDesc,
    allowed_usage: TextureUsage,
) -> Result<(OwnedTexture, NativePresentationToken), String> {
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut guard = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *guard) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeSurface::Dx12(native_surface)) => {
            // SAFETY: this fence is uniquely associated with this one acquired
            // image and is passed to both acquire and the sole submit that may
            // use it.
            let fence = unsafe { device.create_fence() }.map_err(|e| e.to_string())?;
            // SAFETY: safe façade permits at most one outstanding frame and the
            // surface was configured before acquisition.
            let acquired = match unsafe { native_surface.acquire_texture(None, &fence) } {
                Ok(value) => value.texture,
                Err(error) => {
                    // SAFETY: acquisition failed before any queue submission
                    // can reference this uniquely-created fence; it remains
                    // owned solely by this error path and this device is live.
                    unsafe { device.destroy_fence(fence) };
                    return Err(error.to_string());
                }
            };
            // The graph texture is a distinct HAL wrapper around a cloned COM
            // resource. The token retains the original acquired texture, which
            // is the only value later passed to queue submit/present.
            // SAFETY: the cloned COM resource keeps the underlying DX12 image
            // alive; its format, dimension, extent, mip/layer counts exactly
            // match the fixed configured surface descriptor. The token keeps
            // the original acquired texture, surface, and device owner alive
            // until submit/present, discard, or accepted-unknown quarantine.
            let wrapper = unsafe {
                wgpu_hal::dx12::Device::texture_from_raw(
                    acquired.raw_resource().clone(),
                    wgt::TextureFormat::Rgba8Unorm,
                    wgt::TextureDimension::D2,
                    wgt::Extent3d {
                        width: descriptor.extent.width,
                        height: descriptor.extent.height,
                        depth_or_array_layers: 1,
                    },
                    1,
                    1,
                )
            };
            Ok((
                OwnedTexture {
                    native: Some(NativeTexture::Dx12(wrapper)),
                    owner: Arc::clone(owner),
                    descriptor,
                    allowed_usage,
                },
                NativePresentationToken {
                    owner: Arc::clone(owner),
                    surface: Arc::clone(&surface),
                    acquired: Some(NativeAcquiredSurfaceTexture::Dx12(acquired)),
                    fence: Some(NativeSurfaceFence::Dx12(fence)),
                    live_frame,
                    _window: window,
                },
            ))
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}

pub(crate) fn discard_presentation(token: NativePresentationToken) {
    drop(token);
}

/// Submits a command buffer and consumes exactly one acquired DX12 image.
///
/// After a successful queue submission every subsequent failure is returned as
/// completion state, never as a rejection, so graph leases remain quarantined.
pub(crate) fn submit_dx12_presented(
    mut buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
    mut token: NativePresentationToken,
) -> Result<NativeCompletion, String> {
    let finished = buffer.native.take().expect("finished command buffer");
    let NativeFinished::Dx12 {
        owner,
        mut encoder,
        command_buffer,
        render_views,
    } = finished
    else {
        reset_finished(finished);
        return Err("DX12 presentation received a non-DX12 command buffer".into());
    };
    let Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture)) = token.acquired.take() else {
        reset_finished(NativeFinished::Dx12 {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token was already consumed".into());
    };
    let Some(NativeSurfaceFence::Dx12(fence)) = token.fence.take() else {
        reset_finished(NativeFinished::Dx12 {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token has no DX12 fence".into());
    };
    let queue_owner = Arc::clone(&owner);
    let _queue_guard = lock_queue_operations(&queue_owner.queue_operations);
    let NativeDevice::Dx12 { device, queue, .. } = &owner.native else {
        unreachable!()
    };
    // SAFETY: command buffer, acquired texture and unique fence all belong to
    // this queue. The queue guard serializes submit/present with teardown.
    match unsafe { queue.submit(&[&command_buffer], &[&surface_texture], (&fence, 1)) } {
        Ok(()) => {
            let present = {
                let mut surface = token
                    .surface
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let NativeSurface::Dx12(surface) = &mut *surface;
                // SAFETY: queue accepted the command buffer and this is the
                // unique acquired texture for the configured surface.
                unsafe { queue.present(surface, surface_texture) }.err()
            };
            // `Queue::present` takes the acquired texture by value; successful
            // submit has therefore consumed the only surface-owned image.
            // The token now only carries the thread-affine window lease and
            // acquire gate, both of which may be released independently of
            // the ordinary, Send-capable command completion bundle.
            drop(token);
            Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
                NativeSubmission::Dx12 {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(fence),
                    leases,
                    staging_buffers: Vec::new(),
                    render_views,
                    failure: present.map(|_| CompletionFailure::ExecutionFailed),
                },
            ))))
        }
        Err(error) => {
            // A failed submit may still have executed command lists. If idle
            // cannot prove otherwise, retain the image token and every lease.
            if unsafe { queue.wait_for_idle() }.is_ok() {
                unsafe {
                    encoder.reset_all(core::iter::once(command_buffer));
                    device.destroy_fence(fence);
                }
                destroy_render_views(&owner, render_views);
                token.acquired = Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture));
                token.fence = None;
                discard_presentation(token);
                Err(error.to_string())
            } else {
                token.acquired = Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture));
                token.fence = Some(NativeSurfaceFence::Dx12(fence));
                // Neither queue acceptance nor completion can be disproven.
                // Quarantine the complete native bundle, including the
                // thread-affine window/surface token, for process lifetime.
                // Returning a terminal failure lets graph retirement progress
                // without pretending this unobservable work became safe.
                std::mem::forget((owner, encoder, command_buffer, leases, render_views, token));
                Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
                    NativeSubmission::TerminalFailure(CompletionFailure::DeviceLost),
                ))))
            }
        }
    }
}

pub(crate) fn unconfigure_dx12_surface(
    owner: &Arc<OpenedDevice>,
    surface: &Mutex<NativeSurface>,
) -> Result<(), String> {
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut surface = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *surface) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, queue, .. }, NativeSurface::Dx12(surface)) => {
            // SAFETY: shutdown has stopped acquisition and no frame token can
            // remain. Queue idle establishes the HAL unconfigure precondition.
            unsafe { queue.wait_for_idle() }.map_err(|e| e.to_string())?;
            unsafe { surface.unconfigure(device) };
            Ok(())
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}
