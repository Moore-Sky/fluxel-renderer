//! Submit one closed headless fixed-frame draw without reading pixels back.
//!
//! This is an API lifecycle example, not a windowed presentation demo or GPU
//! conformance test. It uses required native validation but intentionally does
//! not use test-only readback or assert a pixel oracle.

use std::{process::ExitCode, thread, time::Duration};

use fluxel_renderer::{
    BasicMaterial, Camera, FixedFrameRenderer, FixedFrameStatus, Geometry, IndexedMeshUpload,
    IndexedMeshUploadStatus,
};
use fluxel_rhi::{Backend, Device, DeviceOptions, Validation};

fn main() -> ExitCode {
    let Some(backend) = parse_backend() else {
        eprintln!(
            "usage: cargo run -p fluxel-renderer --features gpu-upload --example 01_headless_frame -- <dx12|vulkan>"
        );
        return ExitCode::FAILURE;
    };

    let device = match Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    ) {
        Ok(device) => device,
        Err(error) => return report("device open", error),
    };
    let geometry = match Geometry::from_positions(vec![
        [-0.75, -0.75, 0.0],
        [0.75, -0.75, 0.0],
        [0.0, 0.75, 0.0],
    ])
    .with_indices(vec![0, 1, 2])
    {
        Ok(geometry) => geometry,
        Err(error) => return report("geometry", error),
    };
    let material = match BasicMaterial::new([0.15, 0.65, 1.0, 1.0]) {
        Ok(material) => material,
        Err(error) => return report("material", error),
    };

    let mut upload = match IndexedMeshUpload::begin(&device, &geometry) {
        Ok(upload) => upload,
        Err(error) => return report("mesh upload start", error),
    };
    let snapshot = loop {
        match upload.poll() {
            IndexedMeshUploadStatus::Pending => thread::sleep(Duration::from_millis(1)),
            IndexedMeshUploadStatus::Ready => {
                break upload
                    .ready_snapshot()
                    .expect("Ready status publishes the immutable snapshot");
            }
            IndexedMeshUploadStatus::Failed(error) => return report("mesh upload", error),
            status => return report("mesh upload returned an unsupported status", status),
        }
    };

    let renderer = FixedFrameRenderer::new(device);
    let mut draw = match renderer.draw(&snapshot, &Camera::default(), &material, [64, 64]) {
        Ok(draw) => draw,
        Err(error) => return report("fixed draw start", error),
    };
    loop {
        match draw.poll() {
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                thread::sleep(Duration::from_millis(1));
            }
            FixedFrameStatus::Complete(image) => {
                println!(
                    "headless frame complete: extent={:?}, format={:?}",
                    image.extent(),
                    image.format()
                );
                return ExitCode::SUCCESS;
            }
            FixedFrameStatus::Failed(error) => return report("fixed draw", error),
            status => return report("fixed draw returned an unsupported status", status),
        }
    }
}

fn parse_backend() -> Option<Backend> {
    match std::env::args().nth(1).as_deref() {
        Some("dx12") => Some(Backend::Dx12),
        Some("vulkan") => Some(Backend::Vulkan),
        _ => None,
    }
}

fn report(stage: &str, error: impl std::fmt::Debug) -> ExitCode {
    eprintln!("{stage} failed: {error:#?}");
    ExitCode::FAILURE
}
