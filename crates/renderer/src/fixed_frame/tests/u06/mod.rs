//! U06 deliberately uses endpoint-only RGBA8 values.  The two audited
//! samples make U and V clamp independently observable.
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
fn u06_linear_clamp_dx12() {
    run(Backend::Dx12);
}
#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u06_linear_clamp_vulkan() {
    run(Backend::Vulkan);
}
#[cfg(windows)]
#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u06_linear_clamp_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    assert_eq!(
        actual_raster_capabilities(&dx),
        actual_raster_capabilities(&vk)
    );
    let dx_mesh = mesh(&dx);
    let vk_mesh = mesh(&vk);
    let dx_tex = texture(&dx);
    let vk_tex = texture(&vk);
    let caps = actual_raster_capabilities(&dx);
    let graph = Arc::new(
        build_uv_textured_camera_graph(
            &dx_mesh,
            &dx_tex,
            [8, 8],
            &caps,
            RasterRecipe::UV_LINEAR_CLAMP_UNORM,
        )
        .unwrap(),
    );
    let expected = oracle();
    let audit = audit(&expected);
    let mut paired_actual: Option<Vec<u8>> = None;
    for (backend, device, mesh, texture) in [
        (Backend::Dx12, &dx, &dx_mesh, &dx_tex),
        (Backend::Vulkan, &vk, &vk_mesh, &vk_tex),
    ] {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let renderer = FixedFrameRenderer::new(device.clone());
        let mut submission = renderer
            .start_uv_with_recipe(
                mesh,
                FrameTextureSnapshot::Linear(texture.clone()),
                UvStartRequest {
                    graph: Arc::clone(&graph),
                    uniform: FrameUniform::new(&camera(), &BasicMaterial::default()).unwrap(),
                    reservation: mesh.reserve_for_draw().unwrap(),
                    texture_reservation: texture.reserve_for_draw().unwrap(),
                    recipe: RasterRecipe::UV_LINEAR_CLAMP_UNORM,
                },
            )
            .unwrap();
        let actual = complete(device, &mut submission);
        assert_witnesses(backend, &actual.tight, &expected, &audit);
        if let Some(first) = &paired_actual {
            assert_eq!(
                actual.tight,
                *first,
                "U06 paired full-readback difference {:?}",
                first_difference(&actual.tight, first),
            );
        } else {
            paired_actual = Some(actual.tight.clone());
        }
        assert_required_validation_clean(device);
        artifact(
            "paired-same-compiled-graph",
            backend,
            device,
            &graph,
            &actual,
            &expected,
            &audit,
            outgoing(&submission, &graph),
        );
    }
}
#[cfg(windows)]
pub(super) fn open(backend: Backend) -> Device {
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
pub(super) fn geometry() -> TexturedGeometry {
    // All clip w are positive and unequal: 0.95, 1.05, 1.0.
    let positions =
        Geometry::from_positions(vec![[-0.9, -0.9, -0.5], [0.8, -0.9, 0.5], [-0.9, 0.8, 0.0]])
            .with_indices(vec![0, 1, 2])
            .unwrap();
    TexturedGeometry::new(positions, vec![[-0.28, 0.38], [1.28, 0.42], [0.46, 1.34]]).unwrap()
}
#[cfg(windows)]
pub(super) fn camera() -> Camera {
    Camera::new(
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0.05, 0.1],
            [0., 0., 0.5, 1.],
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
pub(super) fn mesh(device: &Device) -> TexturedIndexedMeshSnapshot {
    let mut upload = TexturedIndexedMeshUpload::begin(device, &geometry()).unwrap();
    wait(10, || match upload.poll() {
        TexturedIndexedMeshUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        TexturedIndexedMeshUploadStatus::Pending => None,
        TexturedIndexedMeshUploadStatus::Failed(e) => panic!("U06 mesh {e:?}"),
    })
}
#[cfg(windows)]
fn texture(device: &Device) -> BaseColorTextureSnapshot {
    let mut upload = BaseColorTextureUpload::begin(device, &image()).unwrap();
    wait(10, || match upload.poll() {
        BaseColorTextureUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        BaseColorTextureUploadStatus::Pending => None,
        BaseColorTextureUploadStatus::Failed(e) => panic!("U06 texture {e:?}"),
    })
}
#[cfg(windows)]
pub(super) fn wait<T>(seconds: u64, mut poll: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(v) = poll() {
            return v;
        };
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
}
#[cfg(windows)]
fn run(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let mesh = mesh(&device);
    let tex = texture(&device);
    let material = TexturedBasicMaterial::new(BasicMaterial::default(), tex);
    fluxel_rhi::test_support::clear_validation_diagnostics(&device);
    let mut submission = FixedFrameRenderer::new(device.clone())
        .draw_textured_uv_linear_clamp(&mesh, &camera(), &material, [8, 8])
        .unwrap();
    let actual = complete(&device, &mut submission);
    let expected = oracle();
    let evidence = audit(&expected);
    assert_witnesses(backend, &actual.tight, &expected, &evidence);
    let graph = submission.graph.as_ref().unwrap();
    let states = outgoing(&submission, graph);
    assert_required_validation_clean(&device);
    artifact(
        "independent-public-path",
        backend,
        &device,
        graph,
        &actual,
        &expected,
        &evidence,
        states,
    );
}
#[cfg(windows)]
pub(super) fn complete(
    device: &Device,
    submission: &mut FixedFrameSubmission,
) -> fluxel_rhi::RasterTextureReadback {
    wait(10, || match submission.poll() {
        FixedFrameStatus::Complete(_) => Some(()),
        FixedFrameStatus::Pending | FixedFrameStatus::Busy => None,
        FixedFrameStatus::Failed(e) => panic!("U06 completion {e:?}"),
    });
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
pub(super) fn outgoing(
    submission: &FixedFrameSubmission,
    graph: &CameraGraph,
) -> [ResourceAccessState; 4] {
    let e = &submission.completed.as_ref().unwrap().exports;
    let s = [
        e.buffer(graph.position_export).unwrap().outgoing_state,
        e.buffer(graph.index_export).unwrap().outgoing_state,
        e.buffer(graph.texture_coordinate_export.unwrap())
            .unwrap()
            .outgoing_state,
        e.texture(graph.texture_export.unwrap())
            .unwrap()
            .outgoing_state,
    ];
    assert_eq!(s, [ResourceAccessState::CopyDestination; 4]);
    s
}
#[allow(
    dead_code,
    reason = "the complete audit record is emitted through Debug in the hardware artifact"
)]
#[derive(Clone, Debug)]
struct Sample {
    pixel: [u32; 2],
    winding: f32,
    barycentric: [f32; 3],
    uv: [f32; 2],
    p: [f32; 2],
    floor: [f32; 2],
    texels: [[i32; 2]; 4],
    weights: [f32; 4],
    e: [u8; 4],
    n: [u8; 4],
    repeat: [u8; 4],
    screen_linear: [u8; 4],
    edge_margin: f32,
}
#[derive(Clone, Debug)]
struct Output {
    pixels: Vec<u8>,
    samples: Vec<Option<Sample>>,
}
#[cfg(windows)]
fn oracle() -> Output {
    let g = geometry();
    let im = image();
    let positions = g.geometry().positions();
    // Preserve f32 operation order from the fixed camera transform instead
    // of baking decimal screen-coordinate approximations.
    let vertex = |position: [f32; 3]| {
        let w = 1.0 + position[2] * 0.1;
        (
            w,
            ((position[0] / w + 1.0) * 4.0, (1.0 - position[1] / w) * 4.0),
        )
    };
    let vertices = [
        vertex(positions[0]),
        vertex(positions[1]),
        vertex(positions[2]),
    ];
    let ws = vertices.map(|vertex| vertex.0);
    let ps = vertices.map(|vertex| vertex.1);
    let uvs = g.texture_coordinates();
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let area = edge(ps[0], ps[1], ps[2]);
    let mut pixels = vec![0; 256];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&[0, 0, 0, 255])
    }
    let mut samples = vec![None; 64];
    for y in 0..8 {
        for x in 0..8 {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let raw = [
                edge(ps[1], ps[2], p),
                edge(ps[2], ps[0], p),
                edge(ps[0], ps[1], p),
            ];
            if raw.iter().any(|v| *v <= 0.0) {
                continue;
            };
            let b = raw.map(|v| v / area);
            let d: f32 = (0..3).map(|i| b[i] / ws[i]).sum();
            let uv = [0, 1].map(|a| (0..3).map(|i| b[i] * uvs[i][a] / ws[i]).sum::<f32>() / d);
            let screen_linear_uv = [0, 1].map(|a| (0..3).map(|i| b[i] * uvs[i][a]).sum::<f32>());
            let e = sample(&im, uv, [false, false]);
            let n = nearest(&im, uv);
            let ru = sample(&im, uv, [true, false]);
            let rv = sample(&im, uv, [false, true]);
            let screen_linear = sample(&im, screen_linear_uv, [false, false]);
            let ptex = [uv[0] * 3.0 - 0.5, uv[1] * 2.0 - 0.5];
            let ix = ptex[0].floor() as i32;
            let iy = ptex[1].floor() as i32;
            let fx = ptex[0] - ix as f32;
            let fy = ptex[1] - iy as f32;
            let texels = [[ix, iy], [ix + 1, iy], [ix, iy + 1], [ix + 1, iy + 1]];
            let weights = [
                (1.0 - fx) * (1.0 - fy),
                fx * (1.0 - fy),
                (1.0 - fx) * fy,
                fx * fy,
            ];
            let floor = [fx.min(1.0 - fx), fy.min(1.0 - fy)];
            let edge_margin = raw
                .iter()
                .map(|v| (v / area).abs())
                .fold(f32::INFINITY, f32::min);
            samples[y * 8 + x] = Some(Sample {
                pixel: [x as u32, y as u32],
                winding: area,
                barycentric: b,
                uv,
                p: ptex,
                floor,
                texels,
                weights,
                e,
                n,
                repeat: if uv[0] < 0.0 || uv[0] > 1.0 { ru } else { rv },
                screen_linear,
                edge_margin,
            });
            pixels[(y * 8 + x) * 4..][..4].copy_from_slice(&e);
        }
    }
    Output { pixels, samples }
}
#[cfg(windows)]
fn texel(im: &Rgba8Image, x: i32, y: i32, repeat: [bool; 2]) -> [u8; 4] {
    let x = if repeat[0] {
        x.rem_euclid(3)
    } else {
        x.clamp(0, 2)
    };
    let y = if repeat[1] {
        y.rem_euclid(2)
    } else {
        y.clamp(0, 1)
    };
    im.pixels()[((y * 3 + x) * 4) as usize..][..4]
        .try_into()
        .unwrap()
}
#[cfg(windows)]
fn sample(im: &Rgba8Image, uv: [f32; 2], repeat: [bool; 2]) -> [u8; 4] {
    let p = [uv[0] * 3.0 - 0.5, uv[1] * 2.0 - 0.5];
    let x = p[0].floor() as i32;
    let y = p[1].floor() as i32;
    let fx = p[0] - x as f32;
    let fy = p[1] - y as f32;
    let c = [
        texel(im, x, y, repeat),
        texel(im, x + 1, y, repeat),
        texel(im, x, y + 1, repeat),
        texel(im, x + 1, y + 1, repeat),
    ];
    [0, 1, 2, 3].map(|a| {
        let value = (c[0][a] as f32 * (1.0 - fx) + c[1][a] as f32 * fx) * (1.0 - fy)
            + (c[2][a] as f32 * (1.0 - fx) + c[3][a] as f32 * fx) * fy;
        value.round().clamp(0.0, 255.0) as u8
    })
}
#[cfg(windows)]
fn nearest(im: &Rgba8Image, uv: [f32; 2]) -> [u8; 4] {
    texel(
        im,
        (uv[0] * 3.).floor() as i32,
        (uv[1] * 2.).floor() as i32,
        [false, false],
    )
}
#[cfg(windows)]
fn audit(output: &Output) -> [Sample; 2] {
    let mut u = None;
    let mut v = None;
    for s in output.samples.iter().flatten() {
        let common = s.edge_margin > 0.01
            && s.floor.iter().all(|m| *m > 0.01)
            && channel_distance(s.e, s.n) > 2
            && channel_distance(s.e, s.screen_linear) > 2;
        if common
            && s.uv[0] < 0.0
            && s.uv[1] > 0.0
            && s.uv[1] < 1.0
            && s.e != s.repeat
            && s.pixel == [0, 6]
        {
            u = Some(s.clone())
        };
        if common
            && s.uv[1] > 1.0
            && s.uv[0] > 0.0
            && s.uv[0] < 1.0
            && s.e != s.repeat
            && s.pixel == [1, 2]
        {
            v = Some(s.clone())
        }
    }
    let e = [
        u.expect("U06 fixed U clamp witness"),
        v.expect("U06 fixed V clamp witness"),
    ];
    assert_eq!(e[0].pixel, [0, 6]);
    assert_eq!(e[1].pixel, [1, 2]);
    e
}
#[cfg(windows)]
fn assert_witnesses(backend: Backend, actual: &[u8], expected: &Output, witnesses: &[Sample; 2]) {
    for witness in witnesses {
        let offset = (witness.pixel[1] as usize * 8 + witness.pixel[0] as usize) * 4;
        assert!(
            actual[offset..offset + 4]
                .iter()
                .zip(witness.e)
                .all(|(&a, e)| a.abs_diff(e) <= 1),
            "U06 {backend:?} witness {:?}; actual={:?}; E={:?}; full-frame first difference {:?}",
            witness.pixel,
            &actual[offset..offset + 4],
            witness.e,
            first_difference(actual, &expected.pixels)
        );
        assert!(channel_distance(witness.e, witness.n) > 2);
        assert!(channel_distance(witness.e, witness.repeat) > 2);
        assert!(channel_distance(witness.e, witness.screen_linear) > 2);
    }
}
#[cfg(windows)]
pub(super) fn channel_distance(a: [u8; 4], b: [u8; 4]) -> u8 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| a.abs_diff(b))
        .max()
        .unwrap()
}
#[cfg(windows)]
pub(super) fn assert_required_validation_clean(device: &Device) {
    let diagnostics = fluxel_rhi::test_support::validation_diagnostics(device);
    // DX12 #820 only reports the known missing optimized clear value; it
    // is a performance hint and does not describe an invalid command.
    let unexpected: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| {
            !(diagnostic.contains("severity=D3D12_MESSAGE_SEVERITY(2)")
                && diagnostic.contains("category=D3D12_MESSAGE_CATEGORY(9)")
                && diagnostic.contains("id=D3D12_MESSAGE_ID(820)")
                && diagnostic
                    .contains("The application did not pass any clear value to resource creation"))
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "U06 validation diagnostics: {unexpected:?}"
    );
}
#[cfg(windows)]
#[allow(
    clippy::too_many_arguments,
    reason = "U06 artifact intentionally preserves all independent oracle evidence"
)]
fn artifact(
    mode: &str,
    backend: Backend,
    device: &Device,
    graph: &CameraGraph,
    actual: &fluxel_rhi::RasterTextureReadback,
    expected: &Output,
    evidence: &[Sample; 2],
    outgoing: [ResourceAccessState; 4],
) {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect("U06 exact SHA required");
    assert!(commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()));
    println!(
        "artifact schema=fluxel-u06-v1; case=U06; mode={mode}; commit={commit}; backend={backend:?}; hardware={:?}; driver={}; geometry={:?}; uvs={:?}; image={:?}; camera={:?}; identity={:?}; execution_plan={:?}; expected={:?}; actual={:?}; first_difference={:?}; u_sample={:?}; v_sample={:?}; u_actual={:?}; v_actual={:?}; u_abs_delta={:?}; v_abs_delta={:?}; position_outgoing={:?}; index_outgoing={:?}; uv_outgoing={:?}; texture_outgoing={:?}; target_outgoing=CopySource; pitch={}; completion=Complete; diagnostics={:?}",
        device.hardware(),
        device.hardware().driver,
        geometry().geometry(),
        geometry().texture_coordinates(),
        image(),
        camera(),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            .portable_identity(),
        graph.compiled.execution_plan(),
        expected.pixels,
        actual.tight,
        first_difference(&actual.tight, &expected.pixels),
        evidence[0],
        evidence[1],
        actual_rgba(&actual.tight, evidence[0].pixel),
        actual_rgba(&actual.tight, evidence[1].pixel),
        rgba_delta(actual_rgba(&actual.tight, evidence[0].pixel), evidence[0].e),
        rgba_delta(actual_rgba(&actual.tight, evidence[1].pixel), evidence[1].e),
        outgoing[0],
        outgoing[1],
        outgoing[2],
        outgoing[3],
        actual.bytes_per_row,
        fluxel_rhi::test_support::validation_diagnostics(device)
    );
}
#[cfg(windows)]
fn actual_rgba(bytes: &[u8], pixel: [u32; 2]) -> [u8; 4] {
    bytes[(pixel[1] as usize * 8 + pixel[0] as usize) * 4..][..4]
        .try_into()
        .unwrap()
}
#[cfg(windows)]
pub(super) fn rgba_delta(actual: [u8; 4], expected: [u8; 4]) -> [u8; 4] {
    [
        actual[0].abs_diff(expected[0]),
        actual[1].abs_diff(expected[1]),
        actual[2].abs_diff(expected[2]),
        actual[3].abs_diff(expected[3]),
    ]
}
#[cfg(windows)]
pub(super) fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[cfg(windows)]
pub(super) fn srgb_image() -> crate::Srgba8Image {
    crate::Srgba8Image::new(
        [3, 2],
        vec![
            16, 64, 128, 255, 224, 32, 96, 255, 48, 192, 240, 255, 200, 80, 24, 255, 100, 220, 40,
            255, 252, 140, 72, 255,
        ],
    )
    .unwrap()
}
