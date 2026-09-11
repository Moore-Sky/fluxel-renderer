//! Stage 1 DX12 presentation proof using the host-owned Win32 window primitive.
//!
//! This example owns orchestration only. `fluxel-host` owns the HWND and message
//! pump; the renderer/RHI own the surface, swapchain, acquired image lifetime,
//! submission, and presentation. The renderer has no dependency on host.

#![deny(missing_docs)]

use std::{env, fmt, num::NonZeroIsize, sync::Arc, thread, time::Duration};

use fluxel_host::{Window, WindowConfig, WindowError};
use fluxel_renderer::{
    BasicMaterial, Camera, FixedFrameRenderer, Geometry, IndexedMeshSnapshot, IndexedMeshUpload,
    IndexedMeshUploadStatus, VisibleFrameStatus,
};
use fluxel_rhi::{
    Device, DeviceOptions, Dx12Surface, Validation,
    test_support::{clear_validation_diagnostics, validation_diagnostics},
};

const TITLE: &str = "Fluxel Stage 1.1 — DX12 presentation";
const CLIENT_WIDTH: u32 = 960;
const CLIENT_HEIGHT: u32 = 540;

/// Parsed finite-run controls. Kept separate so parser coverage needs no GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RunOptions {
    frames: Option<u64>,
    repeat: NonZeroIsize,
}

impl RunOptions {
    fn parse<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut frames = None;
        let mut repeat = NonZeroIsize::new(1).expect("one is nonzero");
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--frames" => frames = Some(parse_positive(&argument, args.next())?),
                "--repeat" => {
                    let count = parse_positive(&argument, args.next())?;
                    repeat = NonZeroIsize::new(count as isize)
                        .ok_or_else(|| "--repeat is too large for this platform".to_owned())?;
                }
                "--help" | "-h" => return Err(usage().to_owned()),
                _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
            }
        }
        Ok(Self { frames, repeat })
    }
}

fn parse_positive(flag: &str, value: Option<String>) -> Result<u64, String> {
    let value = value.ok_or_else(|| format!("{flag} requires a positive integer"))?;
    match value.parse::<u64>() {
        Ok(number) if number > 0 => Ok(number),
        _ => Err(format!("{flag} requires a positive integer, got `{value}`")),
    }
}

fn usage() -> &'static str {
    "usage: fluxel-windows-dx12-harness [--frames K] [--repeat R]"
}

fn run_one_lifecycle(options: RunOptions) -> Result<(), HarnessError> {
    let config =
        WindowConfig::new(TITLE, CLIENT_WIDTH, CLIENT_HEIGHT).map_err(HarnessError::Window)?;
    // The Arc is a same-thread lifetime lease retained by `Dx12Surface`, not a
    // cross-thread handoff. `Window` deliberately remains !Send + !Sync.
    #[allow(clippy::arc_with_non_send_sync)]
    let window = Arc::new(Window::new(config).map_err(HarnessError::Window)?);
    eprintln!("created host Window with client extent {CLIENT_WIDTH}x{CLIENT_HEIGHT}");

    let mut surface = Dx12Surface::open(
        Arc::clone(&window),
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
        CLIENT_WIDTH,
        CLIENT_HEIGHT,
    )
    .map_err(HarnessError::Surface)?;
    let device = surface.device();
    clear_validation_diagnostics(&device);
    eprintln!(
        "adapter: backend={:?}, name={}, vendor={:#06x}, device={:#06x}, driver={} {}",
        device.hardware().backend,
        device.hardware().name,
        device.hardware().vendor_id,
        device.hardware().device_id,
        device.hardware().driver,
        device.hardware().driver_info,
    );

    let mut presented = 0;
    let mut snapshot = None;
    let mut renderer = None;
    let work = (|| {
        snapshot = Some(upload_fixed_triangle(&device)?);
        renderer = Some(FixedFrameRenderer::for_surface(&surface));
        let camera = Camera::default();
        let material = BasicMaterial::new([0.15, 0.65, 1.0, 1.0])
            .map_err(|error| HarnessError::Renderer(error.to_string()))?;
        run_frame_loop(&window, options.frames, || {
            let acquired = surface.acquire().map_err(HarnessError::Surface)?;
            let mut submission = renderer
                .as_ref()
                .expect("renderer is installed before the frame loop")
                .draw_to_surface(
                    snapshot
                        .as_ref()
                        .expect("snapshot is installed before the frame loop"),
                    &camera,
                    &material,
                    acquired,
                )
                .map_err(|error| HarnessError::Renderer(error.to_string()))?;
            loop {
                match submission.poll() {
                    VisibleFrameStatus::Pending | VisibleFrameStatus::Busy => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    VisibleFrameStatus::Complete => break,
                    VisibleFrameStatus::Failed(error) => {
                        return Err(HarnessError::Renderer(format!(
                            "visible frame {presented} failed: {error}"
                        )));
                    }
                    status => {
                        return Err(HarnessError::Renderer(format!(
                            "visible frame {presented} returned unsupported status: {status:?}"
                        )));
                    }
                }
            }
            presented += 1;
            Ok(())
        })
    })();

    // No renderer resource or acquired image survives unconfigure. The surface
    // drains known accepted work before releasing its retained `Arc<Window>`.
    drop(renderer);
    drop(snapshot);
    drop(device);
    let shutdown = surface.shutdown().map_err(HarnessError::Surface);
    let diagnostics = validation_diagnostics(&surface.device());
    drop(surface);
    // `try_unwrap` proves that dropping the surface also dropped every native
    // surface/token lease. A mutable Window is required to invalidate future
    // raw-handle borrows before its HWND is destroyed.
    let mut window = Arc::try_unwrap(window).map_err(|_| HarnessError::WindowLeaseOutstanding)?;
    let close = window.close().map_err(HarnessError::Window);
    drop(window);
    eprintln!("frames presented: {presented}");
    if diagnostics.is_empty() {
        eprintln!("validation diagnostics: clean");
    } else {
        eprintln!("validation diagnostics ({}):", diagnostics.len());
        for diagnostic in &diagnostics {
            eprintln!("  {diagnostic}");
        }
    }
    shutdown?;
    if !diagnostics.is_empty() {
        return Err(HarnessError::Validation(diagnostics));
    }
    work?;
    close?;
    eprintln!("clean shutdown: surface unconfigured before host Window destruction");
    Ok(())
}

fn upload_fixed_triangle(device: &Device) -> Result<IndexedMeshSnapshot, HarnessError> {
    let geometry = Geometry::from_positions(vec![
        [-0.75, -0.75, 0.0],
        [0.75, -0.75, 0.0],
        [0.0, 0.75, 0.0],
    ])
    .with_indices(vec![0, 1, 2])
    .map_err(|error| HarnessError::Renderer(error.to_string()))?;
    let mut upload = IndexedMeshUpload::begin(device, &geometry)
        .map_err(|error| HarnessError::Renderer(error.to_string()))?;
    loop {
        match upload.poll() {
            IndexedMeshUploadStatus::Pending => thread::sleep(Duration::from_millis(1)),
            IndexedMeshUploadStatus::Ready => {
                return upload.ready_snapshot().ok_or_else(|| {
                    HarnessError::Renderer("ready mesh upload did not publish a snapshot".into())
                });
            }
            IndexedMeshUploadStatus::Failed(error) => {
                return Err(HarnessError::Renderer(format!(
                    "mesh upload failed: {error:?}"
                )));
            }
            status => {
                return Err(HarnessError::Renderer(format!(
                    "mesh upload returned unsupported status: {status:?}"
                )));
            }
        }
    }
}

fn run_frame_loop(
    window: &Window,
    frame_limit: Option<u64>,
    mut render_and_present: impl FnMut() -> Result<(), HarnessError>,
) -> Result<(), HarnessError> {
    let mut frames = 0;
    while !window.close_requested() && frame_limit.is_none_or(|limit| frames < limit) {
        window.poll_events().map_err(HarnessError::Window)?;
        if window.close_requested() {
            break;
        }
        render_and_present()?;
        frames += 1;
    }
    Ok(())
}

#[derive(Debug)]
enum HarnessError {
    Window(WindowError),
    WindowLeaseOutstanding,
    Surface(fluxel_rhi::SurfaceError),
    Renderer(String),
    Validation(Vec<String>),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Window(error) => write!(formatter, "host Window failed: {error}"),
            Self::WindowLeaseOutstanding => formatter
                .write_str("surface shutdown left an unexpected outstanding host Window lease"),
            Self::Surface(error) => write!(formatter, "DX12 surface failed: {error}"),
            Self::Renderer(error) => write!(formatter, "fixed renderer failed: {error}"),
            Self::Validation(diagnostics) => write!(
                formatter,
                "DX12 validation emitted {} diagnostic(s)",
                diagnostics.len()
            ),
        }
    }
}

fn main() {
    let options = match RunOptions::parse(env::args().skip(1)) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    for cycle in 0..options.repeat.get() {
        if let Err(error) = run_one_lifecycle(options) {
            eprintln!("lifecycle {} failed: {error}", cycle + 1);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RunOptions;

    #[test]
    fn finite_run_controls_parse_without_gpu() {
        let options =
            RunOptions::parse(["--frames", "12", "--repeat", "2"].map(str::to_owned)).unwrap();
        assert_eq!(options.frames, Some(12));
        assert_eq!(options.repeat.get(), 2);
    }

    #[test]
    fn zero_frames_is_rejected() {
        assert!(RunOptions::parse(["--frames", "0"].map(str::to_owned)).is_err());
    }
}
