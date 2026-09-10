//! Tests canonical device-capability fingerprinting.

use super::*;

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(7),
            QueueCapabilities::new(false, true, true, false),
        ))
        .queue(QueueDescriptor::new(
            QueueId::new(3),
            QueueCapabilities::new(true, false, true, true),
        ))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .attachments(true, false, vec![4, 1, 2, 4])
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
                .attachments(false, true, vec![4, 1])
                .build(),
        )
        .surface(SurfaceCapabilities::new(
            vec![
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba8Unorm,
                TextureFormat::Bgra8Unorm,
            ],
            true,
            true,
        ))
        .build()
}

#[test]
fn capability_fingerprint_normalizes_unordered_capability_entries() {
    let capabilities = capabilities();
    let mut reordered = capabilities.clone();
    reordered.queues.reverse();
    reordered.texture_formats.reverse();
    for format in &mut reordered.texture_formats {
        format.attachment_sample_counts.reverse();
    }
    reordered.surface.as_mut().unwrap().formats.reverse();

    assert_eq!(capabilities.fingerprint(), reordered.fingerprint());
}

#[test]
fn capability_fingerprint_retains_duplicate_entries() {
    let capabilities = capabilities();
    let mut with_duplicate = capabilities.clone();
    with_duplicate
        .texture_formats
        .push(with_duplicate.texture_formats[0].clone());

    assert_ne!(capabilities.fingerprint(), with_duplicate.fingerprint());
}

#[test]
fn capability_fingerprint_includes_compute_dispatch_limits() {
    let capabilities = capabilities();
    let mut different_limit = capabilities.clone();
    different_limit.limits =
        DeviceLimits::new(0, 1).with_max_compute_workgroups_per_dimension([65_535, 65_535, 64]);

    assert_ne!(capabilities.fingerprint(), different_limit.fingerprint());
}

#[test]
fn capability_fingerprint_keeps_srgb_format_and_facts_distinct_from_unorm() {
    let unorm = DeviceCapabilities::builder()
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .copies(true, true)
                .build(),
        )
        .build();
    let srgb = DeviceCapabilities::builder()
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8UnormSrgb)
                .sampled(true, true)
                .copies(true, true)
                .build(),
        )
        .build();

    assert_ne!(unorm.fingerprint(), srgb.fingerprint());
}
