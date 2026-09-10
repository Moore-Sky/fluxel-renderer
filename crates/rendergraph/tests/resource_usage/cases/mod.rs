//! Shared descriptors and resource-usage test groups.

//! Resource-usage summary regression tests.

use fluxel_rendergraph::{
    test_rhi::{TestBuffer, TestRegistry, TestRhi, TestTexture, TestTraceEvent},
    *,
};

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(true, true, true))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(true, true)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
                .attachments(false, true, vec![1])
                .build(),
        )
        .build()
}

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 64 }
}

mod declaration;
mod execution;
