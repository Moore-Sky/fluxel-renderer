//! Rendering-owned DX12 presentation façade.
//!
//! This narrow Windows-only boundary binds a native window lifetime to one
//! DX12 surface. It intentionally exposes neither HWND nor DXGI/HAL types.
//! Explicit [`Dx12Surface::shutdown`] is the observable teardown operation.
//! Its `Drop` fallback only attempts the same idle-and-unconfigure sequence
//! when no acquired frame remains; accepted-unknown quarantine deliberately
//! keeps the token, native surface, and window lease alive instead.

use core::fmt;
use std::{
    any::Any,
    rc::Rc,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::{
    Device, DeviceOptions, MemoryPolicy, PhysicalResourceIdentity, Texture, TextureDescriptor,
    TextureLease, TextureShared, imp, next_identity,
};
use fluxel_rendergraph::{
    BoundSurfaceTexture, BoundTexture, ResourceAccessState, TextureDesc, TextureDimension,
    TextureFormat, TextureUsage, TextureUsageKind,
};

/// Failure while creating, configuring, acquiring, or shutting down a DX12
/// presentation surface.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SurfaceError {
    /// Window-handle extraction failed before native surface creation.
    WindowHandle(String),
    /// Device/surface bootstrap did not produce a present-compatible adapter.
    Open(crate::OpenError),
    /// The fixed surface configuration could not be installed.
    Configure(String),
    /// The next backbuffer could not be acquired before recording.
    Acquire(String),
    /// The caller tried to acquire while a prior frame is still outstanding.
    FrameOutstanding,
    /// Shutdown was requested while an acquired frame still exists.
    FrameStillAcquired,
    /// The surface was already shut down and cannot acquire another image.
    NotConfigured,
    /// A native synchronization or teardown operation failed.
    Shutdown(String),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WindowHandle(value) => write!(f, "window handle unavailable: {value}"),
            Self::Open(value) => write!(f, "DX12 surface open failed: {value}"),
            Self::Configure(value) => write!(f, "DX12 surface configure failed: {value}"),
            Self::Acquire(value) => write!(f, "DX12 surface acquire failed: {value}"),
            Self::FrameOutstanding => f.write_str("a surface frame is already acquired"),
            Self::FrameStillAcquired => {
                f.write_str("cannot shut down with an acquired surface frame")
            }
            Self::NotConfigured => f.write_str("surface is not configured"),
            Self::Shutdown(value) => write!(f, "DX12 surface shutdown failed: {value}"),
        }
    }
}

impl std::error::Error for SurfaceError {}

/// A rendering-owned DX12 surface that keeps its native window alive.
///
/// It is deliberately !Send and !Sync: window affinity and swapchain lifetime
/// stay on the harness thread in this first visible-image slice.
pub struct Dx12Surface {
    device: Device,
    native: Arc<Mutex<imp::NativeSurface>>,
    // The native DXGI surface may outlive an acquired token in accepted-unknown
    // quarantine. Retain the concrete window object, not merely a borrow, so
    // HWND destruction cannot race that token's surface teardown.
    window: Arc<dyn Any>,
    width: u32,
    height: u32,
    configured: bool,
    live_frame: Arc<AtomicBool>,
    _thread_affinity: Rc<()>,
}

impl Drop for Dx12Surface {
    fn drop(&mut self) {
        if !may_best_effort_shutdown(self.configured, self.live_frame.load(Ordering::Acquire)) {
            return;
        }
        // Drop must not turn an unwind/early-return cleanup path into a
        // second failure. `unconfigure_dx12_surface` waits for known queue
        // work before touching swapchain state. An error leaves retirement
        // unknown, so retain the complete ownership bundle for process
        // lifetime rather than letting this Drop release native/window state.
        let retained = (
            Arc::clone(&self.device.inner),
            Arc::clone(&self.native),
            Arc::clone(&self.window),
            Arc::clone(&self.live_frame),
        );
        if !quarantine_after_teardown_failure(
            imp::unconfigure_dx12_surface(&self.device.inner, &self.native),
            retained,
        ) {
            self.configured = false;
        }
    }
}

fn may_best_effort_shutdown(configured: bool, frame_outstanding: bool) -> bool {
    configured && !frame_outstanding
}

/// Returns whether a failed best-effort teardown quarantined `ownership`.
///
/// This deliberately leaks only on the branch where native retirement cannot
/// be proven. The caller's ordinary fields may then drop without releasing the
/// cloned device, surface, window, or frame-state ownership bundle.
fn quarantine_after_teardown_failure<T, E>(result: Result<(), E>, ownership: T) -> bool {
    if result.is_err() {
        std::mem::forget(ownership);
        true
    } else {
        false
    }
}

impl Dx12Surface {
    /// Opens a DX12 device selected specifically for this window and installs
    /// the fixed RGBA8 FIFO surface configuration.
    pub fn open<W>(
        window: Arc<W>,
        options: DeviceOptions,
        width: u32,
        height: u32,
    ) -> Result<Self, SurfaceError>
    where
        W: HasWindowHandle + HasDisplayHandle + 'static,
    {
        if width == 0 || height == 0 {
            return Err(SurfaceError::Configure(
                "surface extent must be non-zero".into(),
            ));
        }
        let display = window
            .display_handle()
            .map_err(|e| SurfaceError::WindowHandle(e.to_string()))?
            .as_raw();
        let native_window = window
            .window_handle()
            .map_err(|e| SurfaceError::WindowHandle(e.to_string()))?
            .as_raw();
        let (opened, native_surface) =
            imp::open_dx12_surface(options, display, native_window).map_err(SurfaceError::Open)?;
        let device = Device {
            hardware: opened.hardware.clone(),
            capabilities: opened.capabilities,
            inner: Arc::new(opened),
            identity: fluxel_rendergraph::DeviceIdentity::new(next_identity()),
        };
        let native = Arc::new(Mutex::new(native_surface));
        imp::configure_dx12_surface(&device.inner, &native, width, height)
            .map_err(SurfaceError::Configure)?;
        Ok(Self {
            device,
            native,
            width,
            height,
            configured: true,
            window,
            live_frame: Arc::new(AtomicBool::new(false)),
            _thread_affinity: Rc::new(()),
        })
    }

    /// Returns the device whose adapter was selected for this surface.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Creates the only RasterBackend that advertises this surface's fixed
    /// RGBA8 presentation contract; headless backend construction remains
    /// deliberately non-presentable.
    pub fn raster_backend(&self) -> crate::RasterBackend {
        crate::RasterBackend::for_dx12_surface(self.device())
    }

    /// Acquires the sole frame that may be recorded for this surface now.
    pub fn acquire(&mut self) -> Result<AcquiredSurfaceFrame, SurfaceError> {
        if !self.configured {
            return Err(SurfaceError::NotConfigured);
        }
        if self.live_frame.swap(true, Ordering::AcqRel) {
            return Err(SurfaceError::FrameOutstanding);
        }
        let descriptor = TextureDesc {
            dimension: TextureDimension::D2,
            extent: fluxel_rendergraph::Extent3d {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let usage = TextureUsage::from_kinds([
            TextureUsageKind::ColorAttachment,
            TextureUsageKind::Present,
        ]);
        let (native, token) = match imp::acquire_dx12_surface(
            &self.device.inner,
            Arc::clone(&self.native),
            Arc::clone(&self.window),
            Arc::clone(&self.live_frame),
            descriptor,
            usage,
        ) {
            Ok(value) => value,
            Err(error) => {
                self.live_frame.store(false, Ordering::Release);
                return Err(SurfaceError::Acquire(error));
            }
        };
        let texture = Texture(Arc::new(TextureShared {
            _native: native,
            descriptor: TextureDescriptor {
                texture: descriptor,
                usage,
                memory: MemoryPolicy::DeviceOnly,
            },
            allowed_usage: usage,
            identity: PhysicalResourceIdentity::new(next_identity()),
            device: self.device.identity(),
        }));
        Ok(AcquiredSurfaceFrame {
            texture,
            token: Some(PresentationToken {
                native: Some(token),
                // The native token owns the acquire flag and window lease
                // until discard or accepted-work retirement.
            }),
        })
    }

    /// Waits for known accepted work and releases the native swapchain.
    ///
    /// The caller must first drop or submit its acquired frame. This explicit
    /// shutdown ordering is the contract later resize generations build on.
    /// A later `Drop` is a no-op after success; if callers skip this method,
    /// `Drop` makes the same best-effort attempt only when it is safe.
    pub fn shutdown(&mut self) -> Result<(), SurfaceError> {
        if !self.configured {
            return Err(SurfaceError::NotConfigured);
        }
        if self.live_frame.load(Ordering::Acquire) {
            return Err(SurfaceError::FrameStillAcquired);
        }
        imp::unconfigure_dx12_surface(&self.device.inner, &self.native)
            .map_err(SurfaceError::Shutdown)?;
        self.configured = false;
        Ok(())
    }
}

/// One acquired presentable image and its unforgeable presentation permission.
pub struct AcquiredSurfaceFrame {
    texture: Texture,
    token: Option<PresentationToken>,
}

impl AcquiredSurfaceFrame {
    /// Converts this acquired image into the graph binding for one imported
    /// surface slot. The token is consumed with the binding, preventing a
    /// second present or a frame that records without a corresponding token.
    pub fn into_binding(mut self) -> BoundSurfaceTexture<Texture, TextureLease, PresentationToken> {
        let texture = self.texture;
        BoundSurfaceTexture {
            texture: BoundTexture {
                device: texture.device_identity(),
                identity: texture.identity(),
                physical: texture.clone(),
                descriptor: texture.descriptor().texture,
                usage: texture.allowed_usage(),
                initial_state: ResourceAccessState::Present,
                lease: texture.lease(),
            },
            presentation: self
                .token
                .take()
                .expect("acquired frame owns its presentation token"),
        }
    }
}

/// Opaque one-shot permission to present an acquired DX12 image.
pub struct PresentationToken {
    pub(crate) native: Option<imp::NativePresentationToken>,
}

impl PresentationToken {
    pub(crate) fn into_native(mut self) -> Option<imp::NativePresentationToken> {
        self.native.take()
    }
}

impl Drop for PresentationToken {
    fn drop(&mut self) {
        if let Some(native) = self.native.take() {
            imp::discard_presentation(native);
        }
    }
}

impl fmt::Debug for PresentationToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PresentationToken(..)")
    }
}

#[cfg(test)]
mod tests;
