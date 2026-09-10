//! Shared helpers for the numbered compile-only render-graph examples.

#![allow(dead_code)]

use fluxel_rendergraph::{
    BufferCapabilities, BufferDesc, DeviceCapabilities, DeviceLimits, Extent3d, ExternalOwnership,
    ImportBufferContract, ImportTextureContract, QueueCapabilities, QueueDescriptor, QueueId,
    RecordingCapabilities, RecordingModel, ResourceAccessState, SurfaceCapabilities,
    SurfaceTextureContract, SynchronizationCapabilities, TextureDesc, TextureDimension,
    TextureFormat, TextureFormatCapabilities, TimestampCapabilities, TransientResourceCapabilities,
    TransitionCapabilities,
};

/// Returns the MVP single-queue capability fixture used by the compile-only examples.
pub fn single_queue_capabilities() -> DeviceCapabilities {
    capabilities(
        true,
        RecordingModel::DeferredCommandBuffers,
        true,
        true,
        true,
    )
}

/// Returns a single-sampled two-dimensional RGBA texture descriptor.
pub fn rgba8_texture(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

/// Returns a single-sampled two-dimensional depth texture descriptor.
pub fn depth_texture(width: u32, height: u32) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width,
            height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Depth32Float,
    }
}

/// Returns a buffer descriptor with the supplied byte size.
pub fn buffer(size: u64) -> BufferDesc {
    BufferDesc { size }
}

/// Returns an ordinary caller-owned imported buffer contract.
pub fn imported_buffer(size: u64, initial_state: ResourceAccessState) -> ImportBufferContract {
    ImportBufferContract {
        descriptor: buffer(size),
        initial_state,
        ownership: ExternalOwnership::Caller,
        initial_contents: fluxel_rendergraph::InitialContents::Defined,
    }
}

/// Returns an ordinary caller-owned imported texture contract.
pub fn imported_texture(
    width: u32,
    height: u32,
    initial_state: ResourceAccessState,
) -> ImportTextureContract {
    ImportTextureContract {
        descriptor: rgba8_texture(width, height),
        initial_state,
        ownership: ExternalOwnership::Caller,
        initial_contents: fluxel_rendergraph::InitialContents::Defined,
    }
}

/// Returns a surface-owned imported texture contract.
pub fn imported_surface(width: u32, height: u32) -> SurfaceTextureContract {
    SurfaceTextureContract {
        descriptor: rgba8_texture(width, height),
    }
}

/// Returns the format facts used by the RGBA8 examples.
pub fn rgba8_capabilities() -> TextureFormatCapabilities {
    TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
        .sampled(true, true)
        .storage(true, true)
        .attachments(true, false, vec![1, 4])
        .copies(true, true)
        .build()
}

/// Returns the format facts used by the depth examples.
pub fn depth32_capabilities() -> TextureFormatCapabilities {
    TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
        .sampled(true, false)
        .storage(false, false)
        .attachments(false, true, vec![1, 4])
        .copies(true, false)
        .build()
}

/// Returns a WebGL2-style owner-thread capability fixture without compute.
pub fn webgl2_capabilities() -> DeviceCapabilities {
    capabilities(false, RecordingModel::ImmediateContext, false, false, false)
}

fn capabilities(
    compute: bool,
    recording_model: RecordingModel,
    storage: bool,
    indirect: bool,
    surface_copy: bool,
) -> DeviceCapabilities {
    let rgba8 = TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
        .sampled(true, true)
        .storage(storage, storage)
        .attachments(true, false, vec![1])
        .copies(true, true)
        .build();
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, compute, true, true),
        ))
        .recording(RecordingCapabilities::new(recording_model, false))
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(storage, storage, indirect))
        .texture_format(rgba8)
        .texture_format(depth32_capabilities())
        .surface(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            true,
            surface_copy,
        ))
        .build()
}
