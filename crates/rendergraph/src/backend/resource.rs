//! Physical-resource, attachment, and provider contract types.

use crate::{
    access::{
        BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureRange,
        TextureReadUse, TextureReadWriteUse, TextureWriteUse,
    },
    error::RecordingError,
    handles::{
        BindingSetId, BufferBindingId, ComputePipelineId, PresentTarget, RasterPipelineId,
        SurfaceBindingId, TextureBindingId,
    },
    pass::AttachmentOps,
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc},
};

use super::{DeviceIdentity, ExecutionBackend, FrameBindingError};

/// Stable identity of one physical resource generation within its resource kind.
///
/// Providers must return the same value for aliases of the same native object
/// and a different value after that native handle is recycled for a new
/// generation. Texture and buffer identities occupy separate namespaces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PhysicalResourceIdentity(u64);

impl PhysicalResourceIdentity {
    /// Creates one backend-defined resource-generation identity.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// A texture attachment passed to a raster-pass begin operation.
pub struct RasterColorAttachment<'a, T> {
    /// Zero-based color attachment slot.
    pub index: u32,
    /// Physical texture selected as the attachment.
    pub texture: &'a T,
    /// Selected texture subresources.
    pub range: TextureRange,
    /// Content operations for the attachment.
    pub operations: AttachmentOps<[f32; 4]>,
}

/// A depth-stencil attachment passed to a raster-pass begin operation.
pub struct RasterDepthStencilAttachment<'a, T> {
    /// Physical texture selected as the attachment.
    pub texture: &'a T,
    /// Selected texture subresources.
    pub range: TextureRange,
    /// Depth content operations, when depth is used.
    pub depth: Option<AttachmentOps<f32>>,
    /// Stencil content operations, when stencil is used.
    pub stencil: Option<AttachmentOps<u32>>,
}

/// Backend-independent raster-pass begin information.
pub struct RasterPassDescriptor<'a, T> {
    /// Optional diagnostic label for the pass.
    pub label: &'a str,
    /// Color attachments selected by the compiled pass.
    pub colors: &'a [RasterColorAttachment<'a, T>],
    /// Optional depth-stencil attachment selected by the compiled pass.
    pub depth_stencil: Option<RasterDepthStencilAttachment<'a, T>>,
}

/// A resolved texture together with the lease that keeps it valid for a frame.
pub struct BoundTexture<T, L> {
    /// Backend device that owns this texture.
    pub device: DeviceIdentity,
    /// Physical object generation used to reject unmodeled aliases.
    pub identity: PhysicalResourceIdentity,
    /// Backend-native physical texture object.
    pub physical: T,
    /// Description used to create or validate the physical texture.
    pub descriptor: TextureDesc,
    /// Domain-level operations this physical texture actually permits.
    ///
    /// Providers derive this from their resource-creation contract and known
    /// native facts such as descriptors, flags, format, views, and allocation
    /// constraints. It must describe the physical object, not merely echo the
    /// compiled requirement passed to an allocation request.
    pub usage: TextureUsage,
    /// Semantic state observed when this binding was resolved.
    pub initial_state: ResourceAccessState,
    /// Lease retaining the physical texture until submission completion.
    pub lease: L,
}

/// One acquired, graph-imported presentable image.
///
/// The texture is the only part visible while passes are recorded. The opaque
/// token remains owned by the execution adapter until it is transferred to
/// [`ExecutionBackend::submit`](crate::backend::ExecutionBackend::submit),
/// which is the only boundary allowed to present it. This prevents a renderer
/// callback from presenting an image before the graph's final transition.
pub struct BoundSurfaceTexture<T, L, P> {
    /// The acquired image validated against the graph surface contract.
    pub texture: BoundTexture<T, L>,
    /// One-shot backend presentation token for this acquired image.
    pub presentation: P,
}

/// Result of resolving one acquired presentable image.
pub type SurfaceBindingResult<B> = Result<
    BoundSurfaceTexture<
        <B as ExecutionBackend>::Texture,
        <B as ExecutionBackend>::Lease,
        <B as ExecutionBackend>::PresentationToken,
    >,
    FrameBindingError,
>;

/// A one-shot acquired-image token paired with the graph presentation root it
/// satisfies.
///
/// It contains no native surface or swapchain detail. Backends use the token
/// to perform their native present after accepting the command buffer. Dropping
/// an unsubmitted submission drops the token and therefore cancels acquisition.
pub struct PresentationSubmission<P> {
    /// Graph-declared presentation root associated with this token.
    pub target: PresentTarget,
    /// Backend-owned token returned when the image was acquired.
    pub token: P,
}

/// A resolved buffer together with the lease that keeps it valid for a frame.
pub struct BoundBuffer<B, L> {
    /// Backend device that owns this buffer.
    pub device: DeviceIdentity,
    /// Physical object generation used to reject unmodeled aliases.
    pub identity: PhysicalResourceIdentity,
    /// Backend-native physical buffer object.
    pub physical: B,
    /// Description used to create or validate the physical buffer.
    pub descriptor: BufferDesc,
    /// Domain-level operations this physical buffer actually permits.
    ///
    /// Providers derive this from their resource-creation contract and known
    /// native facts such as descriptors, flags, views, and allocation
    /// constraints. It must describe the physical object, not merely echo the
    /// compiled requirement passed to an allocation request.
    pub usage: BufferUsage,
    /// Semantic state observed when this binding was resolved.
    pub initial_state: ResourceAccessState,
    /// Lease retaining the physical buffer until submission completion.
    pub lease: L,
}

/// A renderer-owned raster pipeline resolved for one backend device.
pub struct BoundRasterPipeline<P, L> {
    /// Backend device that owns this pipeline.
    pub device: DeviceIdentity,
    /// Backend-native raster pipeline object.
    pub physical: P,
    /// Lease retaining the pipeline while commands refer to it.
    pub lease: L,
}

/// A renderer-owned compute pipeline resolved for one backend device.
pub struct BoundComputePipeline<P, L> {
    /// Backend device that owns this pipeline.
    pub device: DeviceIdentity,
    /// Backend-native compute pipeline object.
    pub physical: P,
    /// Lease retaining the pipeline while commands refer to it.
    pub lease: L,
}

/// A renderer-owned binding object resolved for one backend device.
pub struct BoundBindings<B, L> {
    /// Backend device that owns this binding object.
    pub device: DeviceIdentity,
    /// Backend-native binding object.
    pub physical: B,
    /// Lease retaining binding metadata and referenced objects.
    pub lease: L,
}

/// The graph-declared usage which authorizes a resource binding.
///
/// This is deliberately distinct from backend resource states: it retains the
/// domain-level use needed by a renderer to select an SRV, UAV, or analogous
/// native view without exposing compiler-internal access declarations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BindingResourceSemantic {
    /// A read-only texture use.
    TextureRead(TextureReadUse),
    /// A write-only texture use.
    TextureWrite(TextureWriteUse),
    /// A read-modify-write texture use.
    TextureReadWrite(TextureReadWriteUse),
    /// A read-only buffer use.
    BufferRead(BufferReadUse),
    /// A write-only buffer use.
    BufferWrite(BufferWriteUse),
    /// A read-modify-write buffer use.
    BufferReadWrite(BufferReadWriteUse),
}

/// Physical resources made available to a renderer binding recipe.
///
/// Each entry retains the exact graph-authorized range and semantic use. A
/// provider must use those fields when constructing backend views rather than
/// treating the complete physical allocation as implicitly authorized.
pub enum ResolvedBindingResource<'a, T, B> {
    /// A texture selected by a graph-declared access.
    Texture {
        /// Backend-native physical texture object.
        physical: &'a T,
        /// Selected texture subresources.
        range: TextureRange,
        /// Graph-declared usage authorizing this binding.
        semantic: BindingResourceSemantic,
    },
    /// A buffer selected by a graph-declared access.
    Buffer {
        /// Backend-native physical buffer object.
        physical: &'a B,
        /// Selected byte range.
        range: BufferRange,
        /// Graph-declared usage authorizing this binding.
        semantic: BindingResourceSemantic,
    },
}

/// Resolves opaque per-frame resource identities into backend objects.
///
/// The execution adapter calls this provider only after it has checked that a
/// frame input contains every import retained by the compiled plan. Surface
/// bindings model one acquisition: providers must consume a `SurfaceBindingId`
/// exactly once and return a token that cannot be reused for a later frame.
pub trait FrameResourceProvider<B: ExecutionBackend> {
    /// Resolves a caller-owned imported texture identity.
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<B::Texture, B::Lease>, FrameBindingError>;

    /// Resolves a caller-owned imported buffer identity.
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<B::Buffer, B::Lease>, FrameBindingError>;

    /// Resolves one acquired presentable image and its one-shot presentation
    /// token.
    ///
    /// The default keeps providers that only support ordinary imports source
    /// compatible. Implementations must consume `id`, rather than returning a
    /// cloneable cached acquisition. A graph that actually retains a surface slot still fails
    /// closed with [`FrameBindingErrorKind::MissingSurface`](super::FrameBindingErrorKind::MissingSurface).
    fn surface(&self, id: SurfaceBindingId) -> SurfaceBindingResult<B> {
        Err(FrameBindingError {
            kind: super::FrameBindingErrorKind::MissingSurface,
            texture_slot: None,
            buffer_slot: None,
            resource: None,
            surface_binding: Some(id),
            detail: "surface binding is not supported by this frame resource provider".to_owned(),
        })
    }
}

/// Resolves renderer-owned opaque pipelines and binding recipes.
///
/// Binding-resource validation remains the execution adapter's responsibility.
/// The provider receives physical objects together with their graph-authorized
/// ranges and semantic uses, and is responsible for validating its opaque
/// recipe against that information and its native pipeline metadata.
pub trait RenderObjectProvider<B: ExecutionBackend> {
    /// Resolves a raster pipeline opaque identity.
    fn raster_pipeline(
        &self,
        id: RasterPipelineId,
    ) -> Result<BoundRasterPipeline<B::RasterPipeline, B::Lease>, RecordingError>;

    /// Resolves a compute pipeline opaque identity.
    fn compute_pipeline(
        &self,
        id: ComputePipelineId,
    ) -> Result<BoundComputePipeline<B::ComputePipeline, B::Lease>, RecordingError>;

    /// Resolves one binding recipe against validated physical resources.
    fn bindings(
        &self,
        id: BindingSetId,
        resources: &[ResolvedBindingResource<'_, B::Texture, B::Buffer>],
        dynamic_offsets: &[u32],
    ) -> Result<BoundBindings<B::Bindings, B::Lease>, RecordingError>;
}
