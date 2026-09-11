//! Trace types and opaque handles for the deterministic CPU-only backend.

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    access::{BufferCopyRegion, BufferRange, TextureCopyRegion, TextureRange},
    pass::{ScissorRect, Viewport},
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, IndexFormat, QueueId, ResourceAccessState, TextureDesc},
};

/// An opaque physical texture selected by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestTexture(u64);

impl TestTexture {
    /// Creates a deterministic test texture handle.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(super) fn identity(self) -> crate::PhysicalResourceIdentity {
        crate::PhysicalResourceIdentity::new(self.0)
    }
}

/// An opaque physical buffer selected by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestBuffer(u64);

impl TestBuffer {
    /// Creates a deterministic test buffer handle.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub(super) fn identity(self) -> crate::PhysicalResourceIdentity {
        crate::PhysicalResourceIdentity::new(self.0)
    }
}

/// An opaque raster pipeline selected by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestRasterPipeline(u64);

impl TestRasterPipeline {
    /// Creates a deterministic test raster-pipeline handle.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// An opaque compute pipeline selected by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestComputePipeline(u64);

impl TestComputePipeline {
    /// Creates a deterministic test compute-pipeline handle.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// An opaque renderer binding object selected by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestBindings(u64);

impl TestBindings {
    /// Creates a deterministic test binding-object handle.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// A test submission completion token.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TestCompletion(u64);

impl TestCompletion {
    pub(super) fn new(_raw: u64) -> Self {
        Self(_raw)
    }

    /// Returns the submission sequence number carried by this token.
    pub fn sequence(&self) -> u64 {
        self.0
    }
}

/// One acquired test presentation token.
#[derive(Debug)]
pub struct TestPresentationToken {
    value: u64,
    cancellation_count: Arc<AtomicUsize>,
    presented: bool,
}

impl TestPresentationToken {
    /// Creates an opaque test presentation token without a cancellation probe.
    pub fn new(raw: u64) -> Self {
        Self::fresh(raw).0
    }

    /// Creates a token with a probe that observes abandoned acquisitions.
    pub fn fresh(raw: u64) -> (Self, TestPresentationProbe) {
        let cancellation_count = Arc::new(AtomicUsize::new(0));
        (
            Self {
                value: raw,
                cancellation_count: Arc::clone(&cancellation_count),
                presented: false,
            },
            TestPresentationProbe(cancellation_count),
        )
    }

    /// Returns the deterministic value carried by this token.
    pub fn value(&self) -> u64 {
        self.value
    }

    pub(crate) fn mark_presented(mut self) {
        self.presented = true;
    }
}

impl Drop for TestPresentationToken {
    fn drop(&mut self) {
        if !self.presented {
            self.cancellation_count.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// A probe for acquisitions cancelled because their token was not submitted.
#[derive(Clone, Debug)]
pub struct TestPresentationProbe(Arc<AtomicUsize>);

impl TestPresentationProbe {
    /// Returns the number of abandoned acquisitions observed so far.
    pub fn cancellation_count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// Owned color-attachment data captured when a raster pass begins.
#[derive(Clone, Debug, PartialEq)]
pub struct TestColorAttachment {
    /// Zero-based attachment index.
    pub index: u32,
    /// Physical texture selected by the execution adapter.
    pub texture: TestTexture,
    /// Selected texture subresources.
    pub range: TextureRange,
    /// Recorded load, store, clear, and write-coverage operations.
    pub operations: crate::pass::AttachmentOps<[f32; 4]>,
}

/// Owned depth/stencil-attachment data captured when a raster pass begins.
#[derive(Clone, Debug, PartialEq)]
pub struct TestDepthStencilAttachment {
    /// Physical texture selected by the execution adapter.
    pub texture: TestTexture,
    /// Selected texture subresources.
    pub range: TextureRange,
    /// Depth operations, when the depth aspect is attached.
    pub depth: Option<crate::pass::AttachmentOps<f32>>,
    /// Stencil operations, when the stencil aspect is attached.
    pub stencil: Option<crate::pass::AttachmentOps<u32>>,
}

/// One structured event emitted by [`crate::test_rhi::TestRhi`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TestTraceEvent {
    /// A transient texture was created.
    CreateTexture {
        /// Created physical texture.
        texture: TestTexture,
        /// Requested texture descriptor.
        descriptor: TextureDesc,
        /// Operations required when creating this texture.
        usage: TextureUsage,
    },
    /// A transient buffer was created.
    CreateBuffer {
        /// Created physical buffer.
        buffer: TestBuffer,
        /// Requested buffer descriptor.
        descriptor: BufferDesc,
        /// Operations required when creating this buffer.
        usage: BufferUsage,
    },
    /// One ordered encoder was opened.
    BeginEncoder {
        /// Queue selected for recording.
        queue: QueueId,
    },
    /// A texture state transition was recorded.
    TransitionTexture {
        /// Transitioned texture.
        texture: TestTexture,
        /// Transitioned subresources.
        range: TextureRange,
        /// State before the transition.
        before: ResourceAccessState,
        /// State after the transition.
        after: ResourceAccessState,
    },
    /// A buffer state transition was recorded.
    TransitionBuffer {
        /// Transitioned buffer.
        buffer: TestBuffer,
        /// Transitioned bytes.
        range: BufferRange,
        /// State before the transition.
        before: ResourceAccessState,
        /// State after the transition.
        after: ResourceAccessState,
    },
    /// A raster pass was opened.
    BeginRaster {
        /// Raster-pass label.
        label: String,
        /// Number of color attachments.
        color_attachments: usize,
        /// Whether a depth-stencil attachment is present.
        has_depth_stencil: bool,
        /// Complete owned color attachment descriptors in index order.
        colors: Vec<TestColorAttachment>,
        /// Complete owned depth/stencil descriptor, when present.
        depth_stencil: Option<TestDepthStencilAttachment>,
    },
    /// A raster pass was closed.
    EndRaster,
    /// A compute pass was opened.
    BeginCompute {
        /// Compute-pass label.
        label: String,
    },
    /// A compute pass was closed.
    EndCompute,
    /// A copy pass was opened.
    BeginCopy {
        /// Copy-pass label.
        label: String,
    },
    /// A copy pass was closed.
    EndCopy,
    /// A raster pipeline was selected.
    SetRasterPipeline {
        /// Selected raster pipeline.
        pipeline: TestRasterPipeline,
    },
    /// A compute pipeline was selected.
    SetComputePipeline {
        /// Selected compute pipeline.
        pipeline: TestComputePipeline,
    },
    /// A binding object was selected.
    SetBindings {
        /// Selected binding object.
        bindings: TestBindings,
    },
    /// A vertex buffer was selected.
    SetVertexBuffer {
        /// Vertex slot.
        slot: u32,
        /// Selected buffer.
        buffer: TestBuffer,
        /// First bound byte.
        offset: u64,
    },
    /// An index buffer was selected.
    SetIndexBuffer {
        /// Selected buffer.
        buffer: TestBuffer,
        /// First bound byte.
        offset: u64,
        /// Index element format.
        format: IndexFormat,
    },
    /// The raster viewport changed.
    SetViewport {
        /// Selected viewport.
        viewport: Viewport,
    },
    /// The raster scissor changed.
    SetScissor {
        /// Selected scissor rectangle.
        scissor: ScissorRect,
    },
    /// A non-indexed draw was recorded.
    Draw {
        /// Drawn vertex range.
        vertices: Range<u32>,
        /// Drawn instance range.
        instances: Range<u32>,
    },
    /// An indexed draw was recorded.
    DrawIndexed {
        /// Drawn index range.
        indices: Range<u32>,
        /// Added vertex index base.
        base_vertex: i32,
        /// Drawn instance range.
        instances: Range<u32>,
    },
    /// A compute dispatch was recorded.
    Dispatch {
        /// Workgroup counts by axis.
        groups: [u32; 3],
    },
    /// A texture copy was recorded.
    CopyTexture {
        /// Source texture.
        source: TestTexture,
        /// Destination texture.
        destination: TestTexture,
        /// Copied region.
        region: TextureCopyRegion,
    },
    /// A buffer copy was recorded.
    CopyBuffer {
        /// Source buffer.
        source: TestBuffer,
        /// Destination buffer.
        destination: TestBuffer,
        /// Copied region.
        region: BufferCopyRegion,
    },
    /// An encoder was finished.
    Finish {
        /// Queue used by the finished encoder.
        queue: QueueId,
    },
    /// A finished command buffer was submitted.
    Submit {
        /// Queue receiving the command buffer.
        queue: QueueId,
        /// New submission completion token.
        completion: TestCompletion,
        /// Presentation roots and tokens accepted with this submission.
        presentations: Vec<(crate::PresentTarget, u64)>,
    },
    /// A completion and its leases were moved to retirement.
    Retire {
        /// Completion guarded by the retirement entry.
        completion: TestCompletion,
        /// Number of retained leases.
        lease_count: usize,
    },
}
