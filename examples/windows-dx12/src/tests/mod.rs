//! Parser and lifecycle-batch tests for the Windows proof harness.

use fluxel_host::WindowEvent;

use super::{RunOptions, process_event_batch};

#[test]
fn finite_run_controls_parse_without_gpu() {
    let options = RunOptions::parse(
        [
            "--backend",
            "vulkan",
            "--frames",
            "12",
            "--repeat",
            "2",
            "--induce-back-pressure",
            "--evidence-pause-ms",
            "1500",
            "--verify-accepted-unknown",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(options.backend, fluxel_rhi::Backend::Vulkan);
    assert_eq!(options.frames, Some(12));
    assert_eq!(options.repeat.get(), 2);
    assert!(options.induce_back_pressure);
    assert!(options.verify_accepted_unknown);
    assert_eq!(options.evidence_pause_ms, 1500);
}

#[test]
fn zero_frames_is_rejected() {
    assert!(RunOptions::parse(["--frames", "0"].map(str::to_owned)).is_err());
}

#[test]
fn repeat_rejects_values_larger_than_platform_isize() {
    let error =
        RunOptions::parse(["--repeat", &u64::MAX.to_string()].map(str::to_owned)).unwrap_err();
    assert_eq!(error, "--repeat is too large for this platform");
}

#[test]
fn close_discards_surface_actions_from_the_entire_dispatched_batch() {
    let events = vec![
        WindowEvent::Resized {
            width: 800,
            height: 600,
        },
        WindowEvent::CloseRequested,
        WindowEvent::Restored {
            width: 960,
            height: 540,
        },
    ];
    let mut applied = Vec::new();
    let close = process_event_batch(events, |event| {
        applied.push(event);
        Ok(())
    })
    .unwrap();

    assert!(close);
    assert!(applied.is_empty());
}
