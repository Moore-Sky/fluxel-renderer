//! Stage 1 DX12 presentation proof using the host-owned Win32 window primitive.
//!
//! This example owns orchestration only. `fluxel-host` owns the HWND and message
//! pump; the renderer/RHI own the surface, swapchain, acquired image lifetime,
//! submission, and presentation. The renderer has no dependency on host.

#![deny(missing_docs)]

mod frame_ring;

use std::{env, fmt, num::NonZeroIsize, sync::Arc, thread, time::Duration};

use fluxel_host::{Window, WindowConfig, WindowError, WindowEvent};
use fluxel_renderer::{
    BasicMaterial, Camera, FixedFrameRenderer, Geometry, IndexedMeshSnapshot, IndexedMeshUpload,
    IndexedMeshUploadStatus, VisibleFrameSubmission,
};
use fluxel_rhi::{
    Device, DeviceOptions, Dx12Surface, SurfaceExtent, SurfaceStatus, Validation,
    test_support::{
        NextDx12PresentationCompletionLatch, clear_validation_diagnostics,
        hold_next_dx12_presentation_completion, inject_dx12_presentation_accepted_unknown_once,
        validation_diagnostics,
    },
};
use frame_ring::{FrameIdentity, FrameRing, PollReport, StartError};

const TITLE: &str = "Fluxel Stage 1.3 — DX12 bounded frames in flight";
const CLIENT_WIDTH: u32 = 960;
const CLIENT_HEIGHT: u32 = 540;
const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// Parsed finite-run controls. Kept separate so parser coverage needs no GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RunOptions {
    frames: Option<u64>,
    repeat: NonZeroIsize,
    induce_back_pressure: bool,
    verify_accepted_unknown: bool,
    evidence_pause_ms: u64,
}

impl RunOptions {
    fn parse<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut frames = None;
        let mut repeat = NonZeroIsize::new(1).expect("one is nonzero");
        let mut induce_back_pressure = false;
        let mut verify_accepted_unknown = false;
        let mut evidence_pause_ms = 0;
        let mut args = args.into_iter();
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--frames" => frames = Some(parse_positive(&argument, args.next())?),
                "--repeat" => {
                    let count = parse_positive(&argument, args.next())?;
                    let count = isize::try_from(count)
                        .map_err(|_| "--repeat is too large for this platform".to_owned())?;
                    repeat = NonZeroIsize::new(count)
                        .ok_or_else(|| "--repeat is too large for this platform".to_owned())?;
                }
                "--induce-back-pressure" => induce_back_pressure = true,
                "--verify-accepted-unknown" => verify_accepted_unknown = true,
                "--evidence-pause-ms" => {
                    evidence_pause_ms = parse_positive(&argument, args.next())?;
                }
                "--help" | "-h" => return Err(usage().to_owned()),
                _ => return Err(format!("unknown argument `{argument}`\n{}", usage())),
            }
        }
        Ok(Self {
            frames,
            repeat,
            induce_back_pressure,
            verify_accepted_unknown,
            evidence_pause_ms,
        })
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
    "usage: fluxel-windows-dx12-harness [--frames K] [--repeat R] [--induce-back-pressure] [--evidence-pause-ms K] [--verify-accepted-unknown]"
}

fn run_accepted_unknown_oracle() -> Result<(), HarnessError> {
    let config =
        WindowConfig::new(TITLE, CLIENT_WIDTH, CLIENT_HEIGHT).map_err(HarnessError::Window)?;
    #[allow(clippy::arc_with_non_send_sync)]
    let window = Arc::new(Window::new(config).map_err(HarnessError::Window)?);
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
    let snapshot = upload_fixed_triangle(&device)?;
    let renderer = FixedFrameRenderer::for_surface(&surface);
    let camera = Camera::default();
    let material = BasicMaterial::new([0.15, 0.65, 1.0, 1.0])
        .map_err(|error| HarnessError::Renderer(error.to_string()))?;
    let mut ring = FrameRing::new(1).expect("fault oracle has one nonzero slot");

    inject_dx12_presentation_accepted_unknown_once();
    let reservation = ring
        .try_start(|reservation| {
            let acquired = surface.acquire().map_err(HarnessError::Surface)?;
            renderer
                .draw_to_surface(&snapshot, &camera, &material, acquired)
                .map_err(|error| {
                    HarnessError::Renderer(format!(
                        "accepted-unknown frame {} did not start: {error}",
                        reservation.serial()
                    ))
                })
        })
        .map_err(|error| match error {
            StartError::BackPressure => {
                HarnessError::Renderer("fault oracle unexpectedly lacked its first slot".into())
            }
            StartError::Start(error) => error,
        })?;
    let mut submitted = false;
    loop {
        match ring.poll_reserved_once(reservation) {
            Ok(report) => {
                submitted |= !report.submitted().is_empty();
                if !report.retired().is_empty() {
                    return Err(HarnessError::Renderer(
                        "accepted-unknown frame completed instead of reporting failure".into(),
                    ));
                }
                if !submitted {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            Err(error) => {
                if !submitted {
                    return Err(HarnessError::Renderer(format!(
                        "accepted-unknown failed before Submitted: {error}"
                    )));
                }
                eprintln!(
                    "accepted-unknown observed serial={} slot={} failure={error}",
                    reservation.serial(),
                    reservation.slot()
                );
                break;
            }
        }
    }

    let mut forbidden_start_calls = 0;
    if !matches!(
        ring.try_start(|_| {
            forbidden_start_calls += 1;
            Err::<VisibleFrameSubmission, _>(HarnessError::Renderer(
                "unreachable accepted-unknown replacement".into(),
            ))
        }),
        Err(StartError::BackPressure)
    ) || forbidden_start_calls != 0
    {
        return Err(HarnessError::Renderer(
            "accepted-unknown slot became reusable".into(),
        ));
    }
    let resize = surface.resize(SurfaceExtent::new(800, 600));
    if !matches!(resize, Err(fluxel_rhi::SurfaceError::FrameOutstanding)) {
        return Err(HarnessError::Renderer(format!(
            "accepted-unknown resize was not refused: {resize:?}"
        )));
    }
    let shutdown = surface.shutdown();
    if !matches!(shutdown, Err(fluxel_rhi::SurfaceError::FrameStillAcquired)) {
        return Err(HarnessError::Renderer(format!(
            "accepted-unknown shutdown was not refused: {shutdown:?}"
        )));
    }
    let diagnostics = validation_diagnostics(&device);
    if !diagnostics.is_empty() {
        return Err(HarnessError::Validation(diagnostics));
    }

    drop((ring, renderer, snapshot, device, surface));
    let retained_window_owners = Arc::strong_count(&window);
    if retained_window_owners < 2 {
        return Err(HarnessError::Renderer(
            "accepted-unknown surface drop failed to quarantine its window owner".into(),
        ));
    }
    eprintln!(
        "accepted-unknown oracle: slot_reuse=refused resize=FrameOutstanding shutdown=FrameStillAcquired window_owners={retained_window_owners} validation=clean result=pass"
    );
    // Process exit is intentional for this isolated fault fixture: the
    // quarantined owner must not be forged into a normal clean-close path.
    drop(window);
    Ok(())
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

    let mut counters = LifecycleCounters::default();
    let mut snapshot = None;
    let mut renderer = None;
    let mut ring = FrameRing::new(MAX_FRAMES_IN_FLIGHT).expect("frame capacity is nonzero");
    let work = (|| {
        snapshot = Some(upload_fixed_triangle(&device)?);
        renderer = Some(FixedFrameRenderer::for_surface(&surface));
        let camera = Camera::default();
        let material = BasicMaterial::new([0.15, 0.65, 1.0, 1.0])
            .map_err(|error| HarnessError::Renderer(error.to_string()))?;
        run_frame_loop(
            &window,
            &mut surface,
            options,
            &mut counters,
            &mut ring,
            |surface, presented| {
                let acquired = surface.acquire().map_err(HarnessError::Surface)?;
                renderer
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
                    .map_err(|error| {
                        HarnessError::Renderer(format!(
                            "visible frame {presented} did not start: {error}"
                        ))
                    })
            },
        )
    })();

    // No renderer resource or acquired image survives unconfigure. The surface
    // drains known accepted work before releasing its retained `Arc<Window>`.
    drop(renderer);
    drop(snapshot);
    eprintln!(
        "frame lifetime counters: capacity={} high_watermark={} started={} submitted={} retired={} back_pressure={} pending_observations={} controlled_releases={}",
        ring.capacity(),
        ring.high_watermark(),
        counters.started,
        counters.submitted,
        counters.presented,
        ring.back_pressure_count(),
        counters.pending_observations,
        counters.controlled_releases,
    );
    drop(ring);
    drop(device);
    let shutdown = surface.shutdown().map_err(HarnessError::Surface);
    let diagnostics = validation_diagnostics(&surface.device());
    drop(surface);
    if diagnostics.is_empty() {
        eprintln!("validation diagnostics: clean");
    } else {
        eprintln!("validation diagnostics ({}):", diagnostics.len());
        for diagnostic in &diagnostics {
            eprintln!("  {diagnostic}");
        }
    }
    // A poisoned shutdown deliberately retains the Window lease. Report that
    // original lifecycle failure instead of masking it with `try_unwrap`.
    shutdown?;
    // `try_unwrap` proves that dropping the surface also dropped every native
    // surface/token lease. A mutable Window is required to invalidate future
    // raw-handle borrows before its HWND is destroyed.
    let mut window = Arc::try_unwrap(window).map_err(|_| HarnessError::WindowLeaseOutstanding)?;
    let close = window.close().map_err(HarnessError::Window);
    drop(window);
    eprintln!(
        "lifecycle counters: events={} transitions={} presented={} suspended_iterations={}",
        counters.events, counters.transitions, counters.presented, counters.suspended_iterations,
    );
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

#[derive(Default)]
struct LifecycleCounters {
    events: u64,
    transitions: u64,
    presented: u64,
    started: u64,
    submitted: u64,
    suspended_iterations: u64,
    pending_observations: u64,
    controlled_releases: u64,
}

fn run_frame_loop(
    window: &Window,
    surface: &mut Dx12Surface,
    options: RunOptions,
    counters: &mut LifecycleCounters,
    ring: &mut FrameRing<VisibleFrameSubmission>,
    mut start_frame: impl FnMut(&mut Dx12Surface, u64) -> Result<VisibleFrameSubmission, HarnessError>,
) -> Result<(), HarnessError> {
    let mut induced = false;
    let mut held_completions: Vec<Option<NextDx12PresentationCompletionLatch>> =
        (0..ring.capacity()).map(|_| None).collect();
    let mut expected_reuse = None;
    while !window.close_requested() {
        let events = window.poll_events().map_err(HarnessError::Window)?;
        let close = process_event_batch(events, |event| {
            counters.events += 1;
            match event {
                WindowEvent::Resized { width, height }
                | WindowEvent::Restored { width, height } => {
                    release_controlled_completions(&mut held_completions);
                    drain_frame_ring(ring, counters)?;
                    let status = surface
                        .resize(SurfaceExtent::new(width, height))
                        .map_err(HarnessError::Surface)?;
                    counters.transitions += 1;
                    eprintln!(
                        "surface event={event:?} status={status:?} transitions={}",
                        counters.transitions
                    );
                }
                WindowEvent::Minimized => {
                    release_controlled_completions(&mut held_completions);
                    drain_frame_ring(ring, counters)?;
                    let status = surface
                        .resize(SurfaceExtent::new(0, 0))
                        .map_err(HarnessError::Surface)?;
                    counters.transitions += 1;
                    eprintln!(
                        "surface event=Minimized status={status:?} transitions={}",
                        counters.transitions
                    );
                }
                WindowEvent::CloseRequested => unreachable!("batch reducer consumes close"),
                _ => eprintln!("surface event={event:?} action=ignored-unknown-host-event"),
            }
            Ok(())
        })?;
        if close {
            counters.events += 1;
            eprintln!("surface event=CloseRequested action=stop-new-surface-work");
            break;
        }
        if options
            .frames
            .is_some_and(|limit| counters.started >= limit)
        {
            break;
        }
        match surface.status() {
            SurfaceStatus::Active { .. } => {
                let mut reused_controlled_slot = false;
                let reservation = match ring
                    .try_start(|reservation| start_frame(surface, reservation.serial()))
                {
                    Ok(reservation) => reservation,
                    Err(StartError::BackPressure) => {
                        eprintln!(
                            "frame back-pressure count={} live={} capacity={} action=poll-until-retirement",
                            ring.back_pressure_count(),
                            ring.live_count(),
                            ring.capacity()
                        );
                        if options.induce_back_pressure && !induced {
                            if ring.live_count() != ring.capacity()
                                || held_completions.iter().any(Option::is_none)
                            {
                                return Err(HarnessError::Renderer(
                                    "controlled pressure reached admission without N held presentation completions"
                                        .into(),
                                ));
                            }
                            evidence_pause(
                                "capacity-full",
                                options.evidence_pause_ms,
                                ring.live_count(),
                            );
                            let released_slot = held_completions
                                .iter()
                                .enumerate()
                                .nth(1)
                                .map(|(slot, _)| slot)
                                .expect("N=3 has a second controlled slot");
                            drop(held_completions[released_slot].take());
                            induced = true;
                            counters.controlled_releases += 1;
                            eprintln!(
                                "controlled completion release slot={released_slot} remaining_held={} action=poll-only-released-slot",
                                held_completions.iter().flatten().count(),
                            );
                            let retired = wait_for_retirement(ring, counters)?;
                            if retired.slot != released_slot {
                                return Err(HarnessError::Renderer(format!(
                                    "released slot {released_slot}, but slot {} retired",
                                    retired.slot
                                )));
                            }
                            expected_reuse = Some(released_slot);
                            release_controlled_completions(&mut held_completions);
                            continue;
                        }
                        wait_for_retirement(ring, counters)?;
                        continue;
                    }
                    Err(StartError::Start(error)) => return Err(error),
                };
                if let Some(expected_slot) = expected_reuse.take() {
                    if reservation.slot() != expected_slot {
                        return Err(HarnessError::Renderer(format!(
                            "completion released slot {expected_slot}, but admission reused slot {}",
                            reservation.slot()
                        )));
                    }
                    eprintln!(
                        "controlled slot reuse serial={} slot={} result=exact-match",
                        reservation.serial(),
                        reservation.slot()
                    );
                    reused_controlled_slot = true;
                }
                if options.induce_back_pressure && !induced {
                    held_completions[reservation.slot()] =
                        Some(hold_next_dx12_presentation_completion());
                    eprintln!(
                        "controlled completion armed serial={} slot={} held={}/{}",
                        reservation.serial(),
                        reservation.slot(),
                        held_completions.iter().flatten().count(),
                        ring.capacity()
                    );
                }
                eprintln!(
                    "frame submit serial={} slot={} live={}/{}",
                    reservation.serial(),
                    reservation.slot(),
                    ring.live_count(),
                    ring.capacity()
                );
                counters.started += 1;
                advance_to_submitted(ring, reservation, counters)?;
                if options.induce_back_pressure && counters.started == 1 {
                    evidence_pause("one-live", options.evidence_pause_ms, ring.live_count());
                } else if reused_controlled_slot {
                    evidence_pause("exact-reuse", options.evidence_pause_ms, ring.live_count());
                }
            }
            SurfaceStatus::Suspended => {
                counters.suspended_iterations += 1;
                thread::sleep(Duration::from_millis(5));
            }
            SurfaceStatus::Poisoned => {
                return Err(HarnessError::Renderer(
                    "surface entered terminal poisoned state".into(),
                ));
            }
            SurfaceStatus::Closed => {
                return Err(HarnessError::Renderer(
                    "surface closed before the frame loop ended".into(),
                ));
            }
        }
    }
    release_controlled_completions(&mut held_completions);
    drain_frame_ring(ring, counters)?;
    Ok(())
}

fn evidence_pause(stage: &str, pause_ms: u64, live: usize) {
    if pause_ms == 0 {
        return;
    }
    eprintln!("evidence stage={stage} live={live} pause_ms={pause_ms}");
    thread::sleep(Duration::from_millis(pause_ms));
}

fn release_controlled_completions(completions: &mut [Option<NextDx12PresentationCompletionLatch>]) {
    for completion in completions {
        drop(completion.take());
    }
}

fn advance_to_submitted(
    ring: &mut FrameRing<VisibleFrameSubmission>,
    reservation: frame_ring::FrameReservation,
    counters: &mut LifecycleCounters,
) -> Result<(), HarnessError> {
    loop {
        let report = ring
            .poll_reserved_once(reservation)
            .map_err(|error| HarnessError::Renderer(error.to_string()))?;
        let submitted = report
            .submitted()
            .iter()
            .any(|frame| frame.serial == reservation.serial());
        let completed = report
            .retired()
            .iter()
            .any(|frame| frame.serial == reservation.serial());
        record_poll(report, counters);
        if submitted {
            return Ok(());
        }
        if completed {
            return Err(HarnessError::Renderer(format!(
                "frame {} completed without reporting its submission milestone",
                reservation.serial()
            )));
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn wait_for_retirement(
    ring: &mut FrameRing<VisibleFrameSubmission>,
    counters: &mut LifecycleCounters,
) -> Result<FrameIdentity, HarnessError> {
    loop {
        let before = ring.live_count();
        let report = ring
            .poll_once()
            .map_err(|error| HarnessError::Renderer(error.to_string()))?;
        let retired = report.retired().first().copied();
        record_poll(report, counters);
        if ring.live_count() < before {
            return retired.ok_or_else(|| {
                HarnessError::Renderer("ring shrank without a retirement identity".into())
            });
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn drain_frame_ring(
    ring: &mut FrameRing<VisibleFrameSubmission>,
    counters: &mut LifecycleCounters,
) -> Result<(), HarnessError> {
    while !ring.is_empty() {
        let report = ring
            .poll_once()
            .map_err(|error| HarnessError::Renderer(error.to_string()))?;
        record_poll(report, counters);
        if !ring.is_empty() {
            thread::sleep(Duration::from_millis(1));
        }
    }
    Ok(())
}

fn record_poll(report: PollReport, counters: &mut LifecycleCounters) {
    counters.pending_observations += report.pending() as u64;
    for submitted in report.submitted() {
        counters.submitted += 1;
        eprintln!(
            "frame accepted serial={} slot={} submitted={} pending_observed={}",
            submitted.serial, submitted.slot, counters.submitted, counters.pending_observations
        );
    }
    for retired in report.retired() {
        counters.presented += 1;
        eprintln!(
            "frame retire serial={} slot={} retired={} pending_observed={}",
            retired.serial, retired.slot, counters.presented, counters.pending_observations
        );
    }
}

/// Applies ordered facts only until close, leaving all later queued events inert.
fn process_event_batch(
    events: Vec<WindowEvent>,
    mut apply: impl FnMut(WindowEvent) -> Result<(), HarnessError>,
) -> Result<bool, HarnessError> {
    // `poll_events` dispatches the whole native batch before returning. If it
    // contains terminal destruction, even an earlier queued size fact is no
    // longer safe to apply to the now-invalid HWND.
    if events.contains(&WindowEvent::CloseRequested) {
        return Ok(true);
    }
    for event in events {
        apply(event)?;
    }
    Ok(false)
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
        let result = if options.verify_accepted_unknown {
            run_accepted_unknown_oracle()
        } else {
            run_one_lifecycle(options)
        };
        if let Err(error) = result {
            eprintln!("lifecycle {} failed: {error}", cycle + 1);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests;
