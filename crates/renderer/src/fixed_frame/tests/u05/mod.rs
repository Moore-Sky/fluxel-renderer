//! U05 explicit-UV native conformance fixtures.

use super::*;
use crate::{
    BaseColorTextureUpload, BaseColorTextureUploadStatus, BasicMaterial, Camera, Geometry,
    Rgba8Image, TexturedBasicMaterial, TexturedGeometry, TexturedIndexedMeshSnapshot,
    TexturedIndexedMeshUpload, TexturedIndexedMeshUploadStatus,
};
use fluxel_rhi::{Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test};
use std::{
    thread,
    time::{Duration, Instant},
};

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u05_explicit_uv_dx12() {
    run(Backend::Dx12);
}
#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u05_explicit_uv_vulkan() {
    run(Backend::Vulkan);
}
#[cfg(windows)]
#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u05_explicit_uv_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    let dx_mesh = ready_mesh(&dx);
    let vk_mesh = ready_mesh(&vk);
    let dx_texture = ready_texture(&dx);
    let vk_texture = ready_texture(&vk);
    let dx_capabilities = actual_raster_capabilities(&dx);
    assert_eq!(dx_capabilities, actual_raster_capabilities(&vk));
    let graph = Arc::new(
        build_uv_textured_camera_graph(
            &dx_mesh,
            &dx_texture,
            [8, 8],
            &dx_capabilities,
            RasterRecipe::UV_TEXTURE_LOAD,
        )
        .unwrap(),
    );
    let expected = oracle(Interpolation::ExplicitPerspective);
    let distinction = distinguish(&expected);
    for (backend, device, mesh, texture) in [
        (Backend::Dx12, &dx, &dx_mesh, &dx_texture),
        (Backend::Vulkan, &vk, &vk_mesh, &vk_texture),
    ] {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let mut submission = renderer
            .start_uv_textured(
                mesh,
                texture,
                Arc::clone(&graph),
                FrameUniform::new(&camera(), &BasicMaterial::default()).unwrap(),
                mesh.reserve_for_draw().unwrap(),
                texture.reserve_for_draw().unwrap(),
            )
            .unwrap();
        let actual = complete(device, &mut submission);
        let outgoing = outgoing_states(&submission, &graph);
        assert_eq!(
            actual.tight,
            expected.pixels,
            "U05 {backend:?} paired first difference {:?}",
            first_difference(&actual.tight, &expected.pixels)
        );
        assert!(fluxel_rhi::test_support::validation_diagnostics(device).is_empty());
        artifact(
            "paired-same-compiled-graph",
            backend,
            device,
            &graph,
            &actual,
            &expected,
            &distinction,
            outgoing,
        );
    }
}
#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with test-support fault injection"]
fn u05_two_gate_accepted_unknown_poison_contract() {
    let _guard = native_fixture_guard();
    let device = open(Backend::Dx12);
    let mesh = ready_mesh(&device);
    let texture = ready_texture(&device);
    let material = TexturedBasicMaterial::new(BasicMaterial::default(), texture.clone());
    let renderer = FixedFrameRenderer::new(device.clone());
    let mut submission = renderer
        .draw_textured_uv(&mesh, &camera(), &material, [8, 8])
        .unwrap();
    fluxel_rhi::test_support::inject_submit_accepted_unknown_once();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            FixedFrameStatus::Failed(FixedFrameFailure::RasterCompletion(_)) => break,
            other => panic!("unexpected U05 accepted-unknown state {other:?}"),
        }
    }
    let fresh_mesh = ready_mesh(&device);
    assert!(matches!(
        renderer.draw_textured_uv(&fresh_mesh, &camera(), &material, [8, 8]),
        Err(DrawStartError::TexturePoisoned)
    ));
    let fresh_texture = ready_texture(&device);
    let fresh_material = TexturedBasicMaterial::new(BasicMaterial::default(), fresh_texture);
    assert!(matches!(
        renderer.draw_textured_uv(&mesh, &camera(), &fresh_material, [8, 8]),
        Err(DrawStartError::SnapshotPoisoned)
    ));
}

#[cfg(windows)]
fn open(backend: Backend) -> Device {
    Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap()
}
#[cfg(windows)]
fn geometry() -> TexturedGeometry {
    let positions =
        Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap();
    // These deliberately differ from the legacy position-derived UVs.
    TexturedGeometry::new(positions, vec![[0.92, 0.08], [0.06, 0.82], [0.76, 0.94]]).unwrap()
}
#[cfg(windows)]
fn camera() -> Camera {
    Camera::new(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.1],
            [0.0, 0.0, 0.5, 1.0],
        ],
    )
}
#[cfg(windows)]
fn image() -> Rgba8Image {
    Rgba8Image::new(
        [3, 2],
        vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255, 255, 0, 255, 255, 0,
            255, 255, 255,
        ],
    )
    .unwrap()
}
#[cfg(windows)]
fn ready_mesh(device: &Device) -> TexturedIndexedMeshSnapshot {
    let mut upload = TexturedIndexedMeshUpload::begin(device, &geometry()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            TexturedIndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
            TexturedIndexedMeshUploadStatus::Pending => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            TexturedIndexedMeshUploadStatus::Failed(error) => {
                panic!("U05 mesh upload {error:?}")
            }
        }
    }
}
#[cfg(windows)]
fn ready_texture(device: &Device) -> BaseColorTextureSnapshot {
    let mut upload = BaseColorTextureUpload::begin(device, &image()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            BaseColorTextureUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
            BaseColorTextureUploadStatus::Pending => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            BaseColorTextureUploadStatus::Failed(error) => {
                panic!("U05 texture upload {error:?}")
            }
        }
    }
}
#[cfg(windows)]
fn run(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let mesh = ready_mesh(&device);
    let texture = ready_texture(&device);
    let material = TexturedBasicMaterial::new(BasicMaterial::default(), texture);
    fluxel_rhi::test_support::clear_validation_diagnostics(&device);
    let mut submission = FixedFrameRenderer::new(device.clone())
        .draw_textured_uv(&mesh, &camera(), &material, [8, 8])
        .unwrap();
    let actual = complete(&device, &mut submission);
    let expected = oracle(Interpolation::ExplicitPerspective);
    let distinction = distinguish(&expected);
    assert_eq!(
        actual.tight,
        expected.pixels,
        "U05 {backend:?} first difference {:?}",
        first_difference(&actual.tight, &expected.pixels)
    );
    let graph = submission.graph.as_ref().unwrap();
    let frame = submission.completed.as_ref().unwrap();
    assert_eq!(
        frame
            .exports
            .buffer(graph.position_export)
            .unwrap()
            .outgoing_state,
        ResourceAccessState::CopyDestination
    );
    assert_eq!(
        frame
            .exports
            .buffer(graph.index_export)
            .unwrap()
            .outgoing_state,
        ResourceAccessState::CopyDestination
    );
    assert_eq!(
        frame
            .exports
            .buffer(graph.texture_coordinate_export.unwrap())
            .unwrap()
            .outgoing_state,
        ResourceAccessState::CopyDestination
    );
    assert_eq!(
        frame
            .exports
            .texture(graph.texture_export.unwrap())
            .unwrap()
            .outgoing_state,
        ResourceAccessState::CopyDestination
    );
    let diagnostics = fluxel_rhi::test_support::validation_diagnostics(&device);
    assert!(diagnostics.is_empty(), "U05 {backend:?}: {diagnostics:?}");
    artifact(
        "independent-public-path",
        backend,
        &device,
        graph,
        &actual,
        &expected,
        &distinction,
        outgoing_states(&submission, graph),
    );
}
#[cfg(windows)]
fn complete(
    device: &Device,
    submission: &mut FixedFrameSubmission,
) -> fluxel_rhi::RasterTextureReadback {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            FixedFrameStatus::Complete(_) => break,
            FixedFrameStatus::Failed(error) => panic!("U05 completion {error:?}"),
        }
    }
    readback_exported_raster_texture_for_test(
        device,
        submission
            .completed
            .as_ref()
            .unwrap()
            .exports
            .texture(submission.target_export.unwrap())
            .unwrap(),
    )
    .unwrap()
}

#[cfg(windows)]
fn outgoing_states(
    submission: &FixedFrameSubmission,
    graph: &CameraGraph,
) -> [ResourceAccessState; 4] {
    let frame = submission
        .completed
        .as_ref()
        .expect("U05 completed frame retained");
    let states = [
        frame
            .exports
            .buffer(graph.position_export)
            .unwrap()
            .outgoing_state,
        frame
            .exports
            .buffer(graph.index_export)
            .unwrap()
            .outgoing_state,
        frame
            .exports
            .buffer(graph.texture_coordinate_export.unwrap())
            .unwrap()
            .outgoing_state,
        frame
            .exports
            .texture(graph.texture_export.unwrap())
            .unwrap()
            .outgoing_state,
    ];
    assert_eq!(states, [ResourceAccessState::CopyDestination; 4]);
    states
}
#[derive(Clone, Copy)]
enum Interpolation {
    ExplicitPerspective,
    PositionDerivedPerspective,
    ExplicitLinear,
}
#[allow(
    dead_code,
    reason = "all CPU oracle intermediates are deliberately preserved in the hardware artifact"
)]
#[derive(Clone, Debug)]
struct OracleSample {
    pixel: [u32; 2],
    winding: f32,
    barycentric: [f32; 3],
    uv: [f32; 2],
    multiplied: [f32; 2],
    floor_margin: [f32; 2],
    texel: [u32; 2],
    rgba: [u8; 4],
}
#[derive(Clone, Debug)]
struct OracleOutput {
    pixels: Vec<u8>,
    samples: Vec<Option<OracleSample>>,
}
#[derive(Clone, Debug)]
struct DistinctionEvidence {
    explicit: OracleSample,
    derived: OracleSample,
    linear: OracleSample,
}
#[cfg(windows)]
fn oracle(mode: Interpolation) -> OracleOutput {
    let geometry = geometry();
    let camera = camera();
    let image = image();
    let p = camera.projection();
    let v = camera.view();
    let mut m = [[0.0; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            m[c][r] = (0..4).map(|k| p[k][r] * v[c][k]).sum();
        }
    }
    let vertices = geometry
        .geometry()
        .positions()
        .iter()
        .zip(geometry.texture_coordinates())
        .map(|(position, uv)| {
            let input = [position[0], position[1], position[2], 1.0];
            let mut clip = [0.0; 4];
            for (r, component) in clip.iter_mut().enumerate() {
                *component = (0..4).map(|c| m[c][r] * input[c]).sum();
            }
            (
                (
                    (clip[0] / clip[3] + 1.0) * 4.0,
                    (1.0 - clip[1] / clip[3]) * 4.0,
                ),
                *uv,
                [position[0] * 0.5 + 0.5, position[1] * -0.5 + 0.5],
                clip[3],
            )
        })
        .collect::<Vec<_>>();
    let mut out = vec![0; 256];
    let mut samples = vec![None; 64];
    for px in out.chunks_exact_mut(4) {
        px.copy_from_slice(&[0, 0, 0, 255]);
    }
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let top_left = |a: (f32, f32), b: (f32, f32)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
    for triangle in geometry.geometry().indices().chunks_exact(3) {
        let q = [
            &vertices[triangle[0] as usize],
            &vertices[triangle[1] as usize],
            &vertices[triangle[2] as usize],
        ];
        let pts = [q[0].0, q[1].0, q[2].0];
        let area = edge(pts[0], pts[1], pts[2]);
        assert_ne!(area, 0.0);
        for y in 0..8 {
            for x in 0..8 {
                let s = (x as f32 + 0.5, y as f32 + 0.5);
                let raw = [
                    edge(pts[1], pts[2], s),
                    edge(pts[2], pts[0], s),
                    edge(pts[0], pts[1], s),
                ];
                let ds = [(pts[1], pts[2]), (pts[2], pts[0]), (pts[0], pts[1])];
                if !raw.iter().zip(ds).all(|(&e, (a, b))| {
                    if area > 0.0 {
                        e > 0.0 || (e == 0.0 && top_left(a, b))
                    } else {
                        e < 0.0 || (e == 0.0 && top_left(b, a))
                    }
                }) {
                    continue;
                };
                let b = [raw[0] / area, raw[1] / area, raw[2] / area];
                let source = match mode {
                    Interpolation::PositionDerivedPerspective => 1,
                    _ => 0,
                };
                let linear = matches!(mode, Interpolation::ExplicitLinear);
                let d: f32 = (0..3).map(|i| b[i] / q[i].3).sum();
                let uv = [0, 1].map(|axis| {
                    let value = (0..3)
                        .map(|i| {
                            b[i] * (if source == 0 {
                                q[i].1[axis]
                            } else {
                                q[i].2[axis]
                            })
                        })
                        .sum::<f32>();
                    if linear {
                        value
                    } else {
                        (0..3)
                            .map(|i| {
                                b[i] * (if source == 0 {
                                    q[i].1[axis]
                                } else {
                                    q[i].2[axis]
                                }) / q[i].3
                            })
                            .sum::<f32>()
                            / d
                    }
                });
                let multiplied = [uv[0].clamp(0.0, 1.0) * 3.0, uv[1].clamp(0.0, 1.0) * 2.0];
                let tx = multiplied[0].floor().min(2.0) as usize;
                let ty = multiplied[1].floor().min(1.0) as usize;
                let rgba: [u8; 4] = image.pixels()[(ty * 3 + tx) * 4..][..4].try_into().unwrap();
                out[(y * 8 + x) * 4..][..4].copy_from_slice(&rgba);
                let margin =
                    multiplied.map(|value| (value - value.floor()).min(value.ceil() - value));
                samples[y * 8 + x] = Some(OracleSample {
                    pixel: [x as u32, y as u32],
                    winding: area,
                    barycentric: b,
                    uv,
                    multiplied,
                    floor_margin: margin,
                    texel: [tx as u32, ty as u32],
                    rgba,
                });
            }
        }
    }
    OracleOutput {
        pixels: out,
        samples,
    }
}
#[cfg(windows)]
fn distinguish(expected: &OracleOutput) -> DistinctionEvidence {
    let derived = oracle(Interpolation::PositionDerivedPerspective);
    let linear = oracle(Interpolation::ExplicitLinear);
    let evidence = expected.samples.iter().zip(&derived.samples).zip(&linear.samples).find_map(|((explicit, derived), linear)| {
            let (Some(explicit), Some(derived), Some(linear)) = (explicit, derived, linear) else { return None; };
            (explicit.rgba != derived.rgba && explicit.rgba != linear.rgba
                && explicit.floor_margin.iter().all(|margin| *margin > 0.01)
                && derived.floor_margin.iter().all(|margin| *margin > 0.01)
                && linear.floor_margin.iter().all(|margin| *margin > 0.01))
                .then(|| DistinctionEvidence { explicit: explicit.clone(), derived: derived.clone(), linear: linear.clone() })
        }).expect("U05 fixture must have one covered pixel where E RGBA differs from both D and L with strict floor margins");
    assert_eq!(evidence.explicit.pixel, evidence.derived.pixel);
    assert_eq!(evidence.explicit.pixel, evidence.linear.pixel);
    assert!(evidence.explicit.winding != 0.0);
    assert_eq!(
        expected.pixels
            [(evidence.explicit.pixel[1] as usize * 8 + evidence.explicit.pixel[0] as usize) * 4..]
            [..4],
        evidence.explicit.rgba
    );
    evidence
}
#[cfg(windows)]
#[allow(
    clippy::too_many_arguments,
    reason = "the conformance artifact records each independent evidence source explicitly"
)]
fn artifact(
    mode: &str,
    backend: Backend,
    device: &Device,
    graph: &CameraGraph,
    actual: &fluxel_rhi::RasterTextureReadback,
    expected: &OracleOutput,
    distinction: &DistinctionEvidence,
    outgoing: [ResourceAccessState; 4],
) {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect("U05 exact SHA required");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    println!(
        "artifact schema=fluxel-u05-v2; case=U05; mode={mode}; commit={commit}; backend={backend:?}; hardware={:?}; driver={}; geometry={:?}; uvs={:?}; image={:?}; camera={:?}; identity={:?}; execution_plan={:?}; expected={:?}; actual={:?}; first_difference={:?}; e_pixel={:?}; coverage=true; winding={}; barycentric={:?}; e_uv={:?}; d_uv={:?}; l_uv={:?}; e_multiplied={:?}; d_multiplied={:?}; l_multiplied={:?}; e_floor_margin={:?}; d_floor_margin={:?}; l_floor_margin={:?}; e_texel={:?}; d_texel={:?}; l_texel={:?}; e_rgba={:?}; d_rgba={:?}; l_rgba={:?}; position_outgoing={:?}; index_outgoing={:?}; uv_outgoing={:?}; texture_outgoing={:?}; target_outgoing=CopySource; pitch={}; completion=Complete; diagnostics={:?}",
        device.hardware(),
        device.hardware().driver,
        geometry().geometry(),
        geometry().texture_coordinates(),
        image(),
        camera(),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv.portable_identity(),
        graph.compiled.execution_plan(),
        expected.pixels,
        actual.tight,
        first_difference(&actual.tight, &expected.pixels),
        distinction.explicit.pixel,
        distinction.explicit.winding,
        distinction.explicit.barycentric,
        distinction.explicit.uv,
        distinction.derived.uv,
        distinction.linear.uv,
        distinction.explicit.multiplied,
        distinction.derived.multiplied,
        distinction.linear.multiplied,
        distinction.explicit.floor_margin,
        distinction.derived.floor_margin,
        distinction.linear.floor_margin,
        distinction.explicit.texel,
        distinction.derived.texel,
        distinction.linear.texel,
        distinction.explicit.rgba,
        distinction.derived.rgba,
        distinction.linear.rgba,
        outgoing[0],
        outgoing[1],
        outgoing[2],
        outgoing[3],
        actual.bytes_per_row,
        fluxel_rhi::test_support::validation_diagnostics(device)
    );
}
#[cfg(windows)]
fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    actual.iter().zip(expected).position(|(a, b)| a != b)
}
