//! U04 textured fixed-draw native conformance fixtures.

use super::*;
use crate::{
    BaseColorTextureUpload, BaseColorTextureUploadStatus, BasicMaterial, Camera, Geometry,
    IndexedMeshUpload, IndexedMeshUploadStatus, Rgba8Image, TexturedBasicMaterial,
};
use fluxel_rhi::{Backend, DeviceOptions, Validation, readback_exported_raster_texture_for_test};
use std::{
    thread,
    time::{Duration, Instant},
};

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u04_textured_dx12() {
    run(Backend::Dx12);
}
#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u04_textured_vulkan() {
    run(Backend::Vulkan);
}
#[cfg(windows)]
#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u04_textured_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    let dx_mesh = ready_mesh(&dx);
    let dx_texture = ready_texture(&dx);
    let vk_mesh = ready_mesh(&vk);
    let vk_texture = ready_texture(&vk);
    // Both executions consume this exact allocation rather than equivalent recompilations.
    let dx_capabilities = actual_raster_capabilities(&dx);
    assert_eq!(dx_capabilities, actual_raster_capabilities(&vk));
    let graph = Arc::new(
        build_textured_camera_graph(&dx_mesh, &dx_texture, [8, 8], &dx_capabilities).unwrap(),
    );
    let fixture_geometry = mesh();
    let fixture_camera = camera();
    let fixture_image = image();
    let fixture_base_color = [1.0; 4];
    let expected = oracle(
        &fixture_geometry,
        &fixture_camera,
        &fixture_image,
        fixture_base_color,
        [8, 8],
    );
    assert_fixture_distinguishes_interpolation(
        &fixture_geometry,
        &fixture_camera,
        &fixture_image,
        fixture_base_color,
        [8, 8],
    );
    let commit = exact_commit();
    for (backend, device, mesh, texture) in [
        (Backend::Dx12, &dx, &dx_mesh, &dx_texture),
        (Backend::Vulkan, &vk, &vk_mesh, &vk_texture),
    ] {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let mut submission = renderer
            .start_textured(
                mesh,
                texture,
                Arc::clone(&graph),
                FrameUniform::new(&fixture_camera, &BasicMaterial::new(fixture_base_color))
                    .unwrap(),
                mesh.reserve_for_draw().unwrap(),
                texture.reserve_for_draw().unwrap(),
            )
            .unwrap();
        let readback = complete(device, &mut submission);
        assert_eq!(readback.bytes_per_row, 256);
        let actual = &readback.tight;
        assert_eq!(
            actual,
            &expected,
            "U04 {backend:?} paired first difference {:?}",
            first_difference(actual, &expected)
        );
        let diagnostics = fluxel_rhi::test_support::validation_diagnostics(device);
        assert!(diagnostics.is_empty());
        artifact(
            "paired-same-compiled-graph",
            commit.as_str(),
            backend,
            device,
            &graph,
            &fixture_geometry,
            &fixture_camera,
            &fixture_image,
            texture,
            fixture_base_color,
            [8, 8],
            &expected,
            &readback,
            &diagnostics,
        );
    }
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with test-support fault injection"]
fn u04_texture_gate_rejected_accepted_unknown_and_drop_contract() {
    let _guard = native_fixture_guard();
    let device = open(Backend::Dx12);
    let renderer = FixedFrameRenderer::new(device.clone());
    let camera = camera();

    // A rejected raster submit remains pre-accept: both the mesh and the
    // texture reservations must be reusable.
    let mesh = ready_mesh(&device);
    let texture = ready_texture(&device);
    let material = TexturedBasicMaterial::new(BasicMaterial::new([1.0; 4]), texture.clone());
    let mut rejected = renderer
        .draw_textured(&mesh, &camera, &material, [8, 8])
        .unwrap();
    fluxel_rhi::test_support::inject_submit_rejected_once();
    wait_for_failure(&mut rejected, |failure| {
        matches!(failure, FixedFrameFailure::RasterStart(_))
    });
    drop(rejected);
    drop(
        renderer
            .draw_textured(&mesh, &camera, &material, [8, 8])
            .unwrap(),
    );

    // Once submit is accepted but completion is unknown, the texture's
    // generation is poisoned just like the mesh generation.
    let unknown_mesh = ready_mesh(&device);
    let unknown_texture = ready_texture(&device);
    let unknown_material =
        TexturedBasicMaterial::new(BasicMaterial::new([1.0; 4]), unknown_texture.clone());
    let mut unknown = renderer
        .draw_textured(&unknown_mesh, &camera, &unknown_material, [8, 8])
        .unwrap();
    fluxel_rhi::test_support::inject_submit_accepted_unknown_once();
    wait_for_failure(&mut unknown, |failure| {
        matches!(failure, FixedFrameFailure::RasterCompletion(_))
    });
    let fresh_mesh = ready_mesh(&device);
    assert!(matches!(
        renderer.draw_textured(&fresh_mesh, &camera, &unknown_material, [8, 8]),
        Err(DrawStartError::TexturePoisoned)
    ));

    // Early Drop after acceptance is conservatively indistinguishable
    // from unknown completion, so it poisons the texture generation too.
    let dropped_mesh = ready_mesh(&device);
    let dropped_texture = ready_texture(&device);
    let dropped_material =
        TexturedBasicMaterial::new(BasicMaterial::new([1.0; 4]), dropped_texture.clone());
    let mut accepted = renderer
        .draw_textured(&dropped_mesh, &camera, &dropped_material, [8, 8])
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(accepted.phase, CameraPhase::RasterAccepted(_)) {
        assert!(Instant::now() < deadline);
        assert!(matches!(
            accepted.poll(),
            FixedFrameStatus::Pending | FixedFrameStatus::Busy
        ));
        thread::sleep(Duration::from_millis(1));
    }
    drop(accepted);
    let fresh_mesh = ready_mesh(&device);
    assert!(matches!(
        renderer.draw_textured(&fresh_mesh, &camera, &dropped_material, [8, 8]),
        Err(DrawStartError::TexturePoisoned)
    ));
}

#[cfg(windows)]
fn wait_for_failure(
    submission: &mut FixedFrameSubmission,
    expected: impl Fn(&FixedFrameFailure) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            FixedFrameStatus::Failed(failure) if expected(&failure) => break,
            other => panic!("unexpected U04 texture gate state {other:?}"),
        }
    }
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
fn image() -> Rgba8Image {
    // 3 * 4 = 12-byte tight rows deliberately exercise internal 256-byte staging pitch.
    // Endpoint-only channels make UNORM decode, white modulation, and
    // attachment encode bit-exact on both native backends.
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
fn mesh() -> Geometry {
    // No triangle edge crosses an 8x8 pixel center, avoiding a top-left
    // fill-rule ambiguity in the exact CPU oracle.
    Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
        .with_indices(vec![0, 1, 2])
        .unwrap()
}
#[cfg(windows)]
fn camera() -> Camera {
    // clip.w = 1 + 0.1 * model.z.  The three vertices therefore exercise
    // a legal, non-constant w range (0.95, 1.0, 1.05), not affine UVs.
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
fn ready_mesh(device: &Device) -> IndexedMeshSnapshot {
    let mut upload = IndexedMeshUpload::begin(device, &mesh()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            IndexedMeshUploadStatus::Ready => return upload.ready_snapshot().unwrap(),
            IndexedMeshUploadStatus::Pending => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            IndexedMeshUploadStatus::Failed(error) => panic!("U04 mesh upload {error:?}"),
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
                panic!("U04 texture upload {error:?}")
            }
        }
    }
}
#[cfg(windows)]
fn run(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let snapshot = ready_mesh(&device);
    let texture = ready_texture(&device);
    let material = TexturedBasicMaterial::new(BasicMaterial::new([1.0; 4]), texture);
    fluxel_rhi::test_support::clear_validation_diagnostics(&device);
    let renderer = FixedFrameRenderer::new(device.clone());
    let mut submission = renderer
        .draw_textured(&snapshot, &camera(), &material, [8, 8])
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                assert!(Instant::now() < deadline);
                thread::sleep(Duration::from_millis(1));
            }
            FixedFrameStatus::Complete(_) => break,
            FixedFrameStatus::Failed(error) => panic!("U04 {backend:?} failed {error:?}"),
        }
    }
    let frame = submission
        .completed
        .as_ref()
        .expect("test completion retains executed frame");
    let exported = frame
        .exports
        .texture(submission.target_export.unwrap())
        .unwrap();
    assert_eq!(exported.outgoing_state, ResourceAccessState::CopySource);
    let readback = readback_exported_raster_texture_for_test(&device, exported).unwrap();
    assert_eq!(readback.bytes_per_row, 256);
    let fixture_geometry = mesh();
    let fixture_camera = camera();
    let fixture_image = image();
    let fixture_base_color = [1.0; 4];
    let expected = oracle(
        &fixture_geometry,
        &fixture_camera,
        &fixture_image,
        fixture_base_color,
        [8, 8],
    );
    assert_fixture_distinguishes_interpolation(
        &fixture_geometry,
        &fixture_camera,
        &fixture_image,
        fixture_base_color,
        [8, 8],
    );
    assert_eq!(
        readback.tight,
        expected,
        "U04 {backend:?} first difference {:?}",
        first_difference(&readback.tight, &expected)
    );
    assert!(fluxel_rhi::test_support::validation_diagnostics(&device).is_empty());
    let commit = exact_commit();
    let graph = submission.graph.as_ref().unwrap();
    let source_export = graph
        .texture_export
        .expect("textured graph exports its source");
    assert_eq!(
        frame.exports.texture(source_export).unwrap().outgoing_state,
        ResourceAccessState::CopyDestination
    );
    let diagnostics = fluxel_rhi::test_support::validation_diagnostics(&device);
    artifact(
        "independent-public-path",
        commit.as_str(),
        backend,
        &device,
        graph,
        &fixture_geometry,
        &fixture_camera,
        &fixture_image,
        material.base_color_texture(),
        fixture_base_color,
        [8, 8],
        &expected,
        &readback,
        &diagnostics,
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
            FixedFrameStatus::Failed(error) => panic!("U04 completion {error:?}"),
        }
    }
    let frame = submission.completed.as_ref().unwrap();
    let graph = submission.graph.as_ref().unwrap();
    let source_export = graph
        .texture_export
        .expect("textured graph exports its source");
    assert_eq!(
        frame.exports.texture(source_export).unwrap().outgoing_state,
        ResourceAccessState::CopyDestination
    );
    readback_exported_raster_texture_for_test(
        device,
        frame
            .exports
            .texture(submission.target_export.unwrap())
            .unwrap(),
    )
    .unwrap()
}
#[cfg(windows)]
fn oracle(
    geometry: &Geometry,
    camera: &Camera,
    image: &Rgba8Image,
    base_color: [f32; 4],
    extent: [u32; 2],
) -> Vec<u8> {
    oracle_with_interpolation(geometry, camera, image, base_color, extent, true).0
}
#[cfg(windows)]
fn assert_fixture_distinguishes_interpolation(
    geometry: &Geometry,
    camera: &Camera,
    image: &Rgba8Image,
    base_color: [f32; 4],
    extent: [u32; 2],
) {
    let (perspective, perspective_texels) =
        oracle_with_interpolation(geometry, camera, image, base_color, extent, true);
    let (linear, linear_texels) =
        oracle_with_interpolation(geometry, camera, image, base_color, extent, false);
    assert_ne!(
        perspective, linear,
        "U04 fixture must distinguish perspective-correct from linear interpolation"
    );
    assert!(
        perspective_texels
            .iter()
            .zip(&linear_texels)
            .any(|(perspective, linear)| perspective.is_some() && perspective != linear),
        "U04 fixture must choose at least one different textureLoad texel under perspective interpolation"
    );
}
#[cfg(windows)]
fn oracle_with_interpolation(
    geometry: &Geometry,
    camera: &Camera,
    image: &Rgba8Image,
    base_color: [f32; 4],
    extent: [u32; 2],
    perspective_correct: bool,
) -> (Vec<u8>, Vec<Option<[u32; 2]>>) {
    // This intentionally consumes only public fixture inputs. It neither
    // serializes FrameUniform nor shares validation/WGSL helpers with the
    // renderer: P*V, rasterization, interpolation, textureLoad mapping,
    // and RGBA8 encoding are all spelled out here.
    let view = camera.view();
    let projection = camera.projection();
    let mut projection_times_view = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            projection_times_view[column][row] =
                (0..4).map(|k| projection[k][row] * view[column][k]).sum();
        }
    }
    let vertices = geometry
        .positions()
        .iter()
        .map(|position| {
            let position4 = [position[0], position[1], position[2], 1.0];
            let mut clip = [0.0; 4];
            for (row, component) in clip.iter_mut().enumerate() {
                *component = (0..4)
                    .map(|column| projection_times_view[column][row] * position4[column])
                    .sum();
            }
            assert!(clip[3] > 0.0, "U04 fixture has positive clip w");
            let ndc_x = clip[0] / clip[3];
            let ndc_y = clip[1] / clip[3];
            (
                (
                    (ndc_x + 1.0) * extent[0] as f32 * 0.5,
                    (1.0 - ndc_y) * extent[1] as f32 * 0.5,
                ),
                [position[0] * 0.5 + 0.5, position[1] * -0.5 + 0.5],
                clip[3],
            )
        })
        .collect::<Vec<_>>();
    let mut out = vec![0; extent[0] as usize * extent[1] as usize * 4];
    let mut texels = vec![None; extent[0] as usize * extent[1] as usize];
    for pixel in out.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0, 0, 0, 255]);
    }
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let top_left = |a: (f32, f32), b: (f32, f32)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
    for triangle in geometry.indices().chunks_exact(3) {
        let v = [
            &vertices[triangle[0] as usize],
            &vertices[triangle[1] as usize],
            &vertices[triangle[2] as usize],
        ];
        let points = [v[0].0, v[1].0, v[2].0];
        let area = edge(points[0], points[1], points[2]);
        assert_ne!(area, 0.0, "U04 fixture has non-degenerate triangles");
        for y in 0..extent[1] {
            for x in 0..extent[0] {
                let sample = (x as f32 + 0.5, y as f32 + 0.5);
                let raw = [
                    edge(points[1], points[2], sample),
                    edge(points[2], points[0], sample),
                    edge(points[0], points[1], sample),
                ];
                let directed = [
                    (points[1], points[2]),
                    (points[2], points[0]),
                    (points[0], points[1]),
                ];
                let covered = raw.iter().zip(directed).all(|(&value, (a, b))| {
                    if area > 0.0 {
                        value > 0.0 || (value == 0.0 && top_left(a, b))
                    } else {
                        value < 0.0 || (value == 0.0 && top_left(b, a))
                    }
                });
                if !covered {
                    continue;
                }
                let barycentric = [raw[0] / area, raw[1] / area, raw[2] / area];
                let denominator: f32 = (0..3).map(|i| barycentric[i] / v[i].2).sum();
                let uv = [
                    if perspective_correct {
                        (0..3)
                            .map(|i| barycentric[i] * v[i].1[0] / v[i].2)
                            .sum::<f32>()
                            / denominator
                    } else {
                        (0..3).map(|i| barycentric[i] * v[i].1[0]).sum()
                    },
                    if perspective_correct {
                        (0..3)
                            .map(|i| barycentric[i] * v[i].1[1] / v[i].2)
                            .sum::<f32>()
                            / denominator
                    } else {
                        (0..3).map(|i| barycentric[i] * v[i].1[1]).sum()
                    },
                ];
                let [width, height] = image.extent();
                let texel_x = (uv[0].clamp(0.0, 1.0) * width as f32)
                    .floor()
                    .min((width - 1) as f32) as usize;
                let texel_y = (uv[1].clamp(0.0, 1.0) * height as f32)
                    .floor()
                    .min((height - 1) as f32) as usize;
                texels[y as usize * extent[0] as usize + x as usize] =
                    Some([texel_x as u32, texel_y as u32]);
                let source = &image.pixels()[(texel_y * width as usize + texel_x) * 4..][..4];
                let output = &mut out[(y as usize * extent[0] as usize + x as usize) * 4..][..4];
                for channel in 0..4 {
                    output[channel] =
                        ((base_color[channel].clamp(0.0, 1.0) * source[channel] as f32 / 255.0)
                            * 255.0)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (out, texels)
}
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn artifact(
    mode: &str,
    commit: &str,
    backend: Backend,
    device: &Device,
    graph: &CameraGraph,
    geometry: &Geometry,
    camera: &Camera,
    image: &Rgba8Image,
    texture: &BaseColorTextureSnapshot,
    base_color: [f32; 4],
    extent: [u32; 2],
    expected: &[u8],
    readback: &fluxel_rhi::RasterTextureReadback,
    diagnostics: &[String],
) {
    let image_extent = image.extent();
    let image_bytes = image.pixels();
    let positions = geometry.positions();
    let indices = geometry.indices();
    let camera_view = camera.view();
    let camera_projection = camera.projection();
    let source_tight_pitch = image_extent[0] * 4;
    let actual_readback_bytes_per_row = readback.bytes_per_row;
    let actual_readback_padded = &readback.padded;
    let (uploaded_tight, uploaded_padded, uploaded_bytes_per_row) =
        fluxel_rhi::test_support::readback_uploaded_texture(device, texture.texture())
            .expect("U04 source texture readback must succeed");
    assert_eq!(uploaded_bytes_per_row, 256);
    assert_eq!(uploaded_tight, image_bytes, "U04 source tight RGBA8 bytes");
    let source_row_bytes = usize::try_from(source_tight_pitch).unwrap();
    assert!(
        uploaded_padded
            .chunks_exact(uploaded_bytes_per_row as usize)
            .all(|row| row[source_row_bytes..].iter().all(|byte| *byte == 0)),
        "U04 source readback padding must be zeroed"
    );
    assert_eq!(
        texture.texture().outgoing_state(),
        ResourceAccessState::CopyDestination
    );
    let raster_identity =
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture.portable_identity();
    let execution_plan = graph.compiled.execution_plan();
    let actual = &readback.tight;
    let first_difference = first_difference(actual, expected);
    println!(
        "artifact schema=fluxel-u04-v3; case=U04; mode={mode}; commit={commit}; backend={backend:?}; hardware={hardware:?}; driver={driver}; image_extent={image_extent:?}; image_bytes={image_bytes:?}; positions={positions:?}; indices={indices:?}; camera_view={camera_view:?}; camera_projection={camera_projection:?}; base_color={base_color:?}; target_extent={extent:?}; source_tight_pitch={source_tight_pitch}; source_actual_readback_bytes_per_row={uploaded_bytes_per_row}; source_actual_tight={uploaded_tight:?}; source_actual_readback_padded={uploaded_padded:?}; actual_readback_bytes_per_row={actual_readback_bytes_per_row}; actual_readback_padded={actual_readback_padded:?}; raster_identity={raster_identity:?}; execution_plan={execution_plan:?}; expected={expected:?}; actual={actual:?}; first_difference={first_difference:?}; texture_incoming=CopyDestination; texture_outgoing=CopyDestination; target_outgoing=CopySource; completion=Complete; diagnostics={diagnostics:?}",
        hardware = device.hardware(),
        driver = device.hardware().driver,
    );
}
#[cfg(windows)]
fn exact_commit() -> String {
    let commit = std::env::var("FLUXEL_TEST_COMMIT")
        .expect("U04 evidence requires FLUXEL_TEST_COMMIT at the exact tested SHA");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    commit
}
#[cfg(windows)]
fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    actual.iter().zip(expected).position(|(a, b)| a != b)
}
