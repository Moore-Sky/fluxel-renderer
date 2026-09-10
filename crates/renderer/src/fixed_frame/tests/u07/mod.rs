//! U07 sRGB linear-clamp native conformance fixtures.

use super::u06::{
    assert_required_validation_clean, camera, channel_distance, complete, first_difference,
    geometry, mesh, open, outgoing, rgba_delta, srgb_image, wait,
};
use super::*;
use crate::BasicMaterial;
use fluxel_rhi::Backend;

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u07_srgb_linear_clamp_dx12() {
    run_u07(Backend::Dx12);
}
#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u07_srgb_linear_clamp_vulkan() {
    run_u07(Backend::Vulkan);
}
#[cfg(windows)]
#[test]
#[ignore = "requires DX12 and Vulkan devices with required validation"]
fn u07_srgb_linear_clamp_paired_same_compiled_graph() {
    let _guard = native_fixture_guard();
    let commit = u07_commit();
    let dx = open(Backend::Dx12);
    let vk = open(Backend::Vulkan);
    let caps = actual_raster_capabilities(&dx);
    assert_eq!(caps, actual_raster_capabilities(&vk));
    let dx_mesh = mesh(&dx);
    let vk_mesh = mesh(&vk);
    let dx_texture = srgb_texture(&dx);
    let vk_texture = srgb_texture(&vk);
    let source = FrameTextureSnapshot::Srgb(dx_texture.clone());
    let graph = Arc::new(
        build_uv_textured_camera_graph(
            &dx_mesh,
            &source,
            [8, 8],
            &caps,
            RasterRecipe::UV_LINEAR_CLAMP_SRGB,
        )
        .unwrap(),
    );
    let mut first: Option<Vec<u8>> = None;
    for (backend, device, mesh, texture) in [
        (Backend::Dx12, &dx, &dx_mesh, &dx_texture),
        (Backend::Vulkan, &vk, &vk_mesh, &vk_texture),
    ] {
        fluxel_rhi::test_support::clear_validation_diagnostics(device);
        let mut submission = FixedFrameRenderer::new(device.clone())
            .start_uv_with_recipe(
                mesh,
                FrameTextureSnapshot::Srgb(texture.clone()),
                UvStartRequest {
                    graph: Arc::clone(&graph),
                    uniform: FrameUniform::new(&camera(), &BasicMaterial::default()).unwrap(),
                    reservation: mesh.reserve_for_draw().unwrap(),
                    texture_reservation: texture.reserve_for_draw().unwrap(),
                    recipe: RasterRecipe::UV_LINEAR_CLAMP_SRGB,
                },
            )
            .unwrap();
        let actual = complete(device, &mut submission);
        let witnesses = assert_u07_witnesses(backend, &actual.tight);
        if let Some(reference) = &first {
            assert_eq!(
                actual.tight,
                *reference,
                "U07 paired full-readback difference {:?}",
                first_difference(&actual.tight, reference)
            );
        } else {
            first = Some(actual.tight.clone());
        }
        assert_required_validation_clean(device);
        let states = outgoing(&submission, &graph);
        let witness_actual = u07_actual_witnesses(&actual.tight, &witnesses);
        println!(
            "artifact schema=fluxel-u07-v3; case=U07; mode=paired-same-compiled-graph; commit={commit}; os={}; backend={backend:?}; hardware={:?}; driver={}; source_format=Rgba8UnormSrgb; target_format=Rgba8Unorm; capability_facts={caps:?}; identity={:?}; plan={:?}; encoded_bytes={:?}; witnesses={witnesses:?}; actual_witnesses={witness_actual:?}; actual_readback={:?}; outgoing={states:?}; target_outgoing=CopySource; pitch={}; completion=Complete; paired_first_difference=None; diagnostics={:?}",
            std::env::consts::OS,
            device.hardware(),
            device.hardware().driver,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                .portable_identity(),
            graph.compiled.execution_plan(),
            srgb_image().pixels(),
            actual.tight,
            actual.bytes_per_row,
            fluxel_rhi::test_support::validation_diagnostics(device)
        );
    }
}

#[cfg(windows)]
fn srgb_texture(device: &Device) -> crate::SrgbBaseColorTextureSnapshot {
    let mut upload = crate::SrgbBaseColorTextureUpload::begin(device, &srgb_image()).unwrap();
    wait(10, || match upload.poll() {
        crate::SrgbBaseColorTextureUploadStatus::Ready => Some(upload.ready_snapshot().unwrap()),
        crate::SrgbBaseColorTextureUploadStatus::Pending => None,
        crate::SrgbBaseColorTextureUploadStatus::Failed(error) => {
            panic!("U07 texture {error:?}")
        }
    })
}

#[cfg(windows)]
fn run_u07(backend: Backend) {
    let _guard = native_fixture_guard();
    let device = open(backend);
    let mesh = mesh(&device);
    let material =
        crate::SrgbTexturedBasicMaterial::new(BasicMaterial::default(), srgb_texture(&device));
    fluxel_rhi::test_support::clear_validation_diagnostics(&device);
    let mut submission = FixedFrameRenderer::new(device.clone())
        .draw_textured_uv_linear_clamp_srgb(&mesh, &camera(), &material, [8, 8])
        .unwrap();
    let actual = complete(&device, &mut submission);
    let witnesses = assert_u07_witnesses(backend, &actual.tight);
    assert_required_validation_clean(&device);
    let graph = submission.graph.as_ref().unwrap();
    let states = outgoing(&submission, graph);
    let witness_actual = u07_actual_witnesses(&actual.tight, &witnesses);
    let commit = u07_commit();
    println!(
        "artifact schema=fluxel-u07-v3; case=U07; mode=independent-public-path; commit={commit}; backend={backend:?}; hardware={:?}; driver={}; raw_encoded_bytes={:?}; source_format=Rgba8UnormSrgb; target_format=Rgba8Unorm; capability_facts={:?}; identity={:?}; execution_plan={:?}; witnesses={witnesses:?}; actual_witnesses={witness_actual:?}; actual_readback={:?}; outgoing={states:?}; target_outgoing=CopySource; pitch={}; completion=Complete; diagnostics={:?}",
        device.hardware(),
        device.hardware().driver,
        srgb_image().pixels(),
        actual_raster_capabilities(&device),
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            .portable_identity(),
        graph.compiled.execution_plan(),
        actual.tight,
        actual.bytes_per_row,
        fluxel_rhi::test_support::validation_diagnostics(&device)
    );
}

#[cfg(windows)]
fn assert_u07_witnesses(backend: Backend, actual: &[u8]) -> [U07Sample; 2] {
    let witnesses = u07_witnesses();
    for witness in &witnesses {
        let offset = (witness.pixel[1] as usize * 8 + witness.pixel[0] as usize) * 4;
        let got: [u8; 4] = actual[offset..offset + 4].try_into().unwrap();
        assert!(
            got.into_iter()
                .zip(witness.e)
                .all(|(a, e)| a.abs_diff(e) <= 1),
            "U07 {backend:?} {:?}: actual={got:?}; E={:?}",
            witness.pixel,
            witness.e
        );
        for (name, counter) in [("C", witness.c), ("N", witness.n), ("R", witness.r)] {
            assert!(
                channel_distance(witness.e, counter) > 2,
                "U07 {backend:?} {:?}: E={:?}; {name}={counter:?}",
                witness.pixel,
                witness.e
            );
        }
        if witness.pixel == [1, 2] {
            assert!(
                channel_distance(witness.e, witness.a) > 2,
                "U07 {backend:?} V witness {:?}: E={:?}; A={:?}",
                witness.pixel,
                witness.e,
                witness.a
            );
        }
    }
    witnesses
}
#[cfg(windows)]
fn u07_actual_witnesses(actual: &[u8], witnesses: &[U07Sample; 2]) -> [([u8; 4], [u8; 4]); 2] {
    witnesses.each_ref().map(|witness| {
        let offset = (witness.pixel[1] as usize * 8 + witness.pixel[0] as usize) * 4;
        let value: [u8; 4] = actual[offset..offset + 4].try_into().unwrap();
        (value, rgba_delta(value, witness.e))
    })
}

#[cfg(windows)]
fn u07_witnesses() -> [U07Sample; 2] {
    let witnesses = [u07_oracle([0, 7]), u07_oracle([1, 2])];
    assert!(witnesses[0].uv[0] < 0. && witnesses[0].uv[1] > 0. && witnesses[0].uv[1] < 1.);
    assert!(witnesses[1].uv[1] > 1. && witnesses[1].uv[0] > 0. && witnesses[1].uv[0] < 1.);
    assert!(witnesses.iter().all(|sample| sample.edge_margin > 0.01));
    assert!(witnesses.iter().all(|sample| sample.texel_margin > 0.01));
    assert!(witnesses.iter().all(|sample| {
        [sample.c, sample.n, sample.r]
            .into_iter()
            .all(|counter| channel_distance(sample.e, counter) > 2)
    }));
    assert!(
        witnesses
            .iter()
            .all(|sample| sample.quantization_margin > 0.01),
        "U07 quantization margins: {:?}",
        witnesses
            .each_ref()
            .map(|sample| sample.quantization_margin)
    );
    witnesses
}

#[cfg(windows)]
#[allow(
    dead_code,
    reason = "all CPU oracle intermediates are deliberately preserved in the hardware artifact"
)]
#[derive(Clone, Debug)]
struct U07Sample {
    pixel: [u32; 2],
    uv: [f32; 2],
    p: [f32; 2],
    weights: [f32; 4],
    barycentric: [f32; 3],
    edge_margin: f32,
    texel_margin: f32,
    quantization_margin: f32,
    e: [u8; 4],
    c: [u8; 4],
    n: [u8; 4],
    r: [u8; 4],
    a: [u8; 4],
}

#[cfg(windows)]
fn u07_oracle(pixel: [u32; 2]) -> U07Sample {
    let g = geometry();
    let positions = g.geometry().positions();
    let uvs = g.texture_coordinates();
    let vertex = |p: [f32; 3]| {
        let w = 1. + p[2] * 0.1;
        (w, ((p[0] / w + 1.) * 4., (1. - p[1] / w) * 4.))
    };
    let v = [
        vertex(positions[0]),
        vertex(positions[1]),
        vertex(positions[2]),
    ];
    let edge = |a: (f32, f32), b: (f32, f32), p: (f32, f32)| {
        (p.0 - a.0) * (b.1 - a.1) - (p.1 - a.1) * (b.0 - a.0)
    };
    let p = (pixel[0] as f32 + 0.5, pixel[1] as f32 + 0.5);
    let area = edge(v[0].1, v[1].1, v[2].1);
    let raw = [
        edge(v[1].1, v[2].1, p),
        edge(v[2].1, v[0].1, p),
        edge(v[0].1, v[1].1, p),
    ];
    if raw.iter().any(|x| *x <= 0.) {
        return U07Sample {
            pixel,
            uv: [0.; 2],
            p: [0.; 2],
            weights: [0.; 4],
            barycentric: [0.; 3],
            edge_margin: 0.,
            texel_margin: 0.,
            quantization_margin: 0.,
            e: [0; 4],
            c: [0; 4],
            n: [0; 4],
            r: [0; 4],
            a: [0; 4],
        };
    }
    let b = raw.map(|x| x / area);
    let d: f32 = (0..3).map(|i| b[i] / v[i].0).sum();
    let uv = [0, 1].map(|a| (0..3).map(|i| b[i] * uvs[i][a] / v[i].0).sum::<f32>() / d);
    let affine = [0, 1].map(|a| (0..3).map(|i| b[i] * uvs[i][a]).sum::<f32>());
    let ptex = [uv[0] * 3. - 0.5, uv[1] * 2. - 0.5];
    let floor = [ptex[0].floor(), ptex[1].floor()];
    let frac = [ptex[0] - floor[0], ptex[1] - floor[1]];
    let repeat = if uv[0] < 0. || uv[0] > 1. {
        [true, false]
    } else {
        [false, true]
    };
    U07Sample {
        pixel,
        uv,
        p: ptex,
        weights: [
            (1. - frac[0]) * (1. - frac[1]),
            frac[0] * (1. - frac[1]),
            (1. - frac[0]) * frac[1],
            frac[0] * frac[1],
        ],
        barycentric: b,
        edge_margin: b.into_iter().map(f32::abs).fold(f32::INFINITY, f32::min),
        texel_margin: frac
            .into_iter()
            .flat_map(|value| [value, 1. - value])
            .fold(f32::INFINITY, f32::min),
        quantization_margin: quantization_margin(srgb_sample_linear(uv, [false, false], true)),
        e: srgb_sample(uv, [false, false], true),
        c: srgb_sample(uv, [false, false], false),
        n: srgb_nearest(uv),
        r: srgb_sample(uv, repeat, true),
        a: srgb_sample(affine, [false, false], true),
    }
}
#[cfg(windows)]
fn srgb_sample(uv: [f32; 2], repeat: [bool; 2], decode_each: bool) -> [u8; 4] {
    srgb_sample_linear(uv, repeat, decode_each)
        .map(|value| (value * 255.).round().clamp(0., 255.) as u8)
}
#[cfg(windows)]
fn srgb_sample_linear(uv: [f32; 2], repeat: [bool; 2], decode_each: bool) -> [f32; 4] {
    let p = [uv[0] * 3. - 0.5, uv[1] * 2. - 0.5];
    let x = p[0].floor() as i32;
    let y = p[1].floor() as i32;
    let fx = p[0] - x as f32;
    let fy = p[1] - y as f32;
    let t = [
        srgb_texel(x, y, repeat),
        srgb_texel(x + 1, y, repeat),
        srgb_texel(x, y + 1, repeat),
        srgb_texel(x + 1, y + 1, repeat),
    ];
    [0, 1, 2, 3].map(|a| {
        let f = |b: u8| {
            if a == 3 || !decode_each {
                b as f32 / 255.
            } else {
                srgb_decode(b)
            }
        };
        let l = (f(t[0][a]) * (1. - fx) + f(t[1][a]) * fx) * (1. - fy)
            + (f(t[2][a]) * (1. - fx) + f(t[3][a]) * fx) * fy;
        if a == 3 || decode_each {
            l
        } else {
            // Counter C is deliberately the mathematically wrong path:
            // filter normalized encoded values in f32, then decode once.
            // Do not introduce a byte quantization between those steps.
            srgb_decode_unit(l)
        }
    })
}
#[cfg(windows)]
fn quantization_margin(linear: [f32; 4]) -> f32 {
    linear
        .into_iter()
        .map(|value| {
            let scaled = value * 255.;
            (scaled - (scaled.floor() + 0.5)).abs()
        })
        .fold(f32::INFINITY, f32::min)
}
#[cfg(windows)]
fn srgb_decode(b: u8) -> f32 {
    srgb_decode_unit(b as f32 / 255.)
}
#[cfg(windows)]
fn srgb_decode_unit(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
#[cfg(windows)]
fn srgb_texel(x: i32, y: i32, repeat: [bool; 2]) -> [u8; 4] {
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
    srgb_image().pixels()[((y * 3 + x) * 4) as usize..][..4]
        .try_into()
        .unwrap()
}
#[cfg(windows)]
fn u07_commit() -> String {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect("U07 exact SHA required");
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    if commit.bytes().all(|byte| byte == b'0') {
        return "working-tree".into();
    }
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("U07 needs git rev-parse HEAD");
    assert!(head.status.success(), "U07 git rev-parse HEAD failed");
    assert_eq!(String::from_utf8(head.stdout).unwrap().trim(), commit);
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .expect("U07 needs git status");
    assert!(status.status.success(), "U07 git status failed");
    assert!(
        status.stdout.is_empty(),
        "U07 exact-SHA evidence requires a clean worktree: {}",
        String::from_utf8_lossy(&status.stdout)
    );
    commit
}
#[cfg(windows)]
fn srgb_nearest(uv: [f32; 2]) -> [u8; 4] {
    let raw = srgb_texel(
        (uv[0] * 3.).floor() as i32,
        (uv[1] * 2.).floor() as i32,
        [false, false],
    );
    [0, 1, 2, 3].map(|channel| {
        if channel == 3 {
            raw[channel]
        } else {
            (srgb_decode(raw[channel]) * 255.).round() as u8
        }
    })
}
