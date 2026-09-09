//! Safe owned-resource creation and lifetime contracts.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, PhysicalResourceIdentity, TextureDesc,
    TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};

use crate::{Backend, Device};

/// Host visibility policy for an owned resource.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MemoryPolicy {
    /// The resource is not exposed for host mapping by this API.
    #[default]
    DeviceOnly,
}

/// Description of an owned buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferDescriptor {
    /// Logical buffer shape.
    pub buffer: BufferDesc,
    /// Operations the resource must permit.
    pub usage: BufferUsage,
    /// Host visibility policy.
    pub memory: MemoryPolicy,
}

/// Description of an owned texture.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureDescriptor {
    /// Logical texture shape and format.
    pub texture: TextureDesc,
    /// Operations the resource must permit.
    pub usage: TextureUsage,
    /// Host visibility policy.
    pub memory: MemoryPolicy,
}

/// The resource kind involved in a creation error.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceKind {
    /// A buffer.
    Buffer,
    /// A texture.
    Texture,
}

/// A validated reason why a resource descriptor was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidResourceReason {
    /// A buffer has zero size.
    ZeroSize,
    /// No operation was requested.
    EmptyUsage,
    /// A texture extent contains zero.
    ZeroExtent,
    /// Mip-level count is zero or exceeds the extent's full chain.
    InvalidMipLevels,
    /// Array layers are zero or incompatible with the dimension.
    InvalidArrayLayers,
    /// Sample count is unsupported or incompatible with the descriptor.
    InvalidSampleCount,
    /// Extent components are incompatible with the texture dimension.
    InvalidDimension,
    /// This resource slice supports only two-dimensional textures.
    UnsupportedDimension,
    /// The resource exceeds a limit reported by the selected device.
    ExceedsDeviceLimit,
    /// The requested operation cannot be used with this resource shape or format.
    IncompatibleUsage,
    /// Presentation is reserved for acquired surface images.
    PresentRequiresSurface,
}

/// Why an owned resource could not be created.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResourceCreateError {
    /// The safe descriptor contract was invalid.
    InvalidDescriptor {
        /// Resource kind.
        resource: ResourceKind,
        /// Stable rejection reason.
        reason: InvalidResourceReason,
    },
    /// The native backend failed after portable validation succeeded.
    NativeFailure {
        /// Backend performing the operation.
        backend: Backend,
        /// Native diagnostic captured at the private boundary.
        reason: String,
    },
}

impl fmt::Display for ResourceCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor { resource, reason } => {
                write!(f, "invalid {resource:?} descriptor: {reason:?}")
            }
            Self::NativeFailure { backend, reason } => {
                write!(f, "{backend:?} resource creation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for ResourceCreateError {}

/// Why creation of a fixed compute artifact was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ComputeCreateError {
    /// The device cannot support the fixed shader's declared workgroup.
    UnsupportedComputeLimits,
    /// The pipeline, buffer, or binding operation crossed device identities.
    ForeignDevice,
    /// The requested storage-buffer view is empty, unaligned, overflowing, or out of bounds.
    InvalidBindingRange,
    /// The buffer's actual native usage cannot provide read-write storage access.
    StorageUsageRequired,
    /// The selected fixed artifact does not accept this binding recipe.
    BindingRecipeMismatch,
    /// The fixed WGSL artifact failed Naga parsing or validation.
    ShaderValidation(String),
    /// The backend rejected shader lowering or compute-pipeline compilation.
    ShaderCompilation(String),
    /// The backend could not create a fixed layout or another non-shader object.
    NativeObjectCreation(String),
    /// Native creation of a validated fixed binding object failed.
    NativeFailure(String),
}

impl fmt::Display for ComputeCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedComputeLimits => {
                f.write_str("device does not support this fixed compute workgroup")
            }
            Self::ForeignDevice => f.write_str("compute objects belong to different devices"),
            Self::InvalidBindingRange => f.write_str("invalid compute storage-buffer range"),
            Self::StorageUsageRequired => {
                f.write_str("compute bindings require read-write storage usage")
            }
            Self::BindingRecipeMismatch => {
                f.write_str("compute bindings do not match the fixed artifact recipe")
            }
            Self::ShaderValidation(reason) => {
                write!(f, "compute shader validation failed: {reason}")
            }
            Self::ShaderCompilation(reason) => {
                write!(f, "compute shader compilation failed: {reason}")
            }
            Self::NativeObjectCreation(reason) => {
                write!(f, "native compute object creation failed: {reason}")
            }
            Self::NativeFailure(reason) => {
                write!(f, "native compute binding creation failed: {reason}")
            }
        }
    }
}
impl std::error::Error for ComputeCreateError {}

/// Why creation of a fixed raster artifact was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RasterCreateError {
    /// The pipeline, attachment, or buffer operation crossed device identities.
    ForeignDevice,
    /// The fixed WGSL artifact failed Naga parsing or validation.
    ShaderValidation(String),
    /// The backend rejected shader lowering or raster-pipeline compilation.
    ShaderCompilation(String),
    /// The backend could not create a fixed layout or another non-shader object.
    NativeObjectCreation(String),
}

impl fmt::Display for RasterCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignDevice => f.write_str("raster objects belong to different devices"),
            Self::ShaderValidation(reason) => {
                write!(f, "raster shader validation failed: {reason}")
            }
            Self::ShaderCompilation(reason) => {
                write!(f, "raster shader compilation failed: {reason}")
            }
            Self::NativeObjectCreation(reason) => {
                write!(f, "native raster object creation failed: {reason}")
            }
        }
    }
}
impl std::error::Error for RasterCreateError {}

#[cfg(windows)]
fn map_compute_pipeline_create_error(
    error: crate::imp::ComputePipelineCreateError,
) -> ComputeCreateError {
    match error {
        crate::imp::ComputePipelineCreateError::ShaderValidation(reason) => {
            ComputeCreateError::ShaderValidation(reason)
        }
        crate::imp::ComputePipelineCreateError::ShaderCompilation(reason) => {
            ComputeCreateError::ShaderCompilation(reason)
        }
        crate::imp::ComputePipelineCreateError::NativeObjectCreation(reason) => {
            ComputeCreateError::NativeObjectCreation(reason)
        }
    }
}

#[cfg(windows)]
fn map_raster_pipeline_create_error(
    error: crate::imp::RasterPipelineCreateError,
) -> RasterCreateError {
    match error {
        crate::imp::RasterPipelineCreateError::ShaderValidation(reason) => {
            RasterCreateError::ShaderValidation(reason)
        }
        crate::imp::RasterPipelineCreateError::ShaderCompilation(reason) => {
            RasterCreateError::ShaderCompilation(reason)
        }
        crate::imp::RasterPipelineCreateError::NativeObjectCreation(reason) => {
            RasterCreateError::NativeObjectCreation(reason)
        }
    }
}

struct BufferShared {
    _native: crate::imp::OwnedBuffer,
    descriptor: BufferDescriptor,
    allowed_usage: BufferUsage,
    identity: PhysicalResourceIdentity,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// One opaque owned native buffer.
#[derive(Clone)]
pub struct Buffer(Arc<BufferShared>);

/// A cloneable lifetime token for a buffer.
#[derive(Clone)]
pub struct BufferLease(Arc<BufferShared>);

struct TextureShared {
    _native: crate::imp::OwnedTexture,
    descriptor: TextureDescriptor,
    allowed_usage: TextureUsage,
    identity: PhysicalResourceIdentity,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// One opaque owned native texture.
#[derive(Clone)]
pub struct Texture(Arc<TextureShared>);

/// A cloneable lifetime token for a texture.
#[derive(Clone)]
pub struct TextureLease(Arc<TextureShared>);

/// The fixed, deterministic compute artifacts supported by this milestone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeKernel {
    /// Adds one to every addressed `u32` with wrapping arithmetic.
    WrappingAdd,
    /// Multiplies every addressed `u32` by three with wrapping arithmetic.
    WrappingMultiply,
    /// Packs every texel of one `Rgba8Unorm` texture into row-major `u32`s.
    TexturePackRgba8,
}

/// Portable identity of one fixed compute artifact.
///
/// This identifies the validated source-level artifact shared by every native
/// backend. It deliberately does not identify backend-specific DXIL or SPIR-V
/// binaries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeArtifactIdentity {
    /// Hash of the complete embedded WGSL module.
    pub module_source_hash: u64,
    /// Selected entry point within that module.
    pub entry_point: &'static str,
    /// Declared local workgroup shape.
    pub workgroup_size: [u32; 3],
    /// Version of the fixed one-RW-storage-buffer binding recipe.
    pub binding_recipe_version: u32,
}

impl ComputeKernel {
    /// Returns the fixed WGSL entry point for this artifact.
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::WrappingAdd => "wrapping_add",
            Self::WrappingMultiply => "wrapping_multiply",
            Self::TexturePackRgba8 => "pack_rgba8",
        }
    }

    /// Returns the fixed workgroup shape baked into this artifact.
    pub const fn workgroup_size(self) -> [u32; 3] {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => [64, 1, 1],
            Self::TexturePackRgba8 => [8, 8, 1],
        }
    }

    /// Returns the source hash recorded by conformance fixtures.
    pub const fn source_hash(self) -> u64 {
        // FNV-1a is an artifact identifier, not a security primitive. Keeping
        // the calculation adjacent to the embedded source makes accidental
        // source/hash drift impossible.
        fnv1a64(self.wgsl_source().as_bytes())
    }

    /// Returns the portable identity shared by DX12 and Vulkan lowering.
    pub const fn portable_identity(self) -> ComputeArtifactIdentity {
        ComputeArtifactIdentity {
            module_source_hash: self.source_hash(),
            entry_point: self.entry_point(),
            workgroup_size: self.workgroup_size(),
            binding_recipe_version: self.binding_recipe_version(),
        }
    }

    /// Returns the version of this slice's one read-write storage binding recipe.
    pub const fn binding_recipe_version(self) -> u32 {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => 1,
            Self::TexturePackRgba8 => 2,
        }
    }

    pub(crate) const fn wgsl_source(self) -> &'static str {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => {
                "@group(0) @binding(0) var<storage, read_write> values: array<u32>;\n\
         @compute @workgroup_size(64)\n\
         fn wrapping_add(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             if (id.x < arrayLength(&values)) { values[id.x] = values[id.x] + 1u; }\n\
         }\n\
         @compute @workgroup_size(64)\n\
         fn wrapping_multiply(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             if (id.x < arrayLength(&values)) { values[id.x] = values[id.x] * 3u; }\n\
         }"
            }
            Self::TexturePackRgba8 => {
                "@group(0) @binding(0) var source: texture_2d<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> destination: array<u32>;\n\
         @compute @workgroup_size(8, 8, 1)\n\
         fn pack_rgba8(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             let dimensions = textureDimensions(source);\n\
             if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }\n\
             let pixel = textureLoad(source, vec2<i32>(id.xy), 0);\n\
             let r = u32(round(pixel.r * 255.0));\n\
             let g = u32(round(pixel.g * 255.0));\n\
             let b = u32(round(pixel.b * 255.0));\n\
             let a = u32(round(pixel.a * 255.0));\n\
             destination[id.y * dimensions.x + id.x] = r | (g << 8u) | (b << 16u) | (a << 24u);\n\
         }"
            }
        }
    }
}

pub(crate) const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

struct ComputePipelineShared {
    _native: crate::imp::NativeComputePipeline,
    kernel: ComputeKernel,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque, device-affine pipeline for one fixed compute artifact.
#[derive(Clone)]
pub struct ComputePipeline(Arc<ComputePipelineShared>);

/// A cloneable strong lease for a compute pipeline and its native layout/module.
#[derive(Clone)]
pub struct ComputePipelineLease(Arc<ComputePipelineShared>);

struct ComputeBindingsShared {
    _native: crate::imp::NativeComputeBindings,
    pipeline: ComputePipeline,
    _buffer: BufferLease,
    offset: u64,
    size: u64,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque binding object for exactly one in-place RW storage buffer.
#[derive(Clone)]
pub struct ComputeBindings(Arc<ComputeBindingsShared>);

/// A cloneable strong lease for a binding object, its pipeline, and its buffer.
#[derive(Clone)]
pub struct ComputeBindingsLease(Arc<ComputeBindingsShared>);

/// The two deterministic raster artifacts supported by this milestone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterKernel {
    /// A fixed, non-indexed triangle with no resource bindings.
    Triangle,
    /// An indexed mesh using `float32x2 position + unorm8x4 color` vertices.
    IndexedPositionColor,
}

/// The non-configurable vertex layout selected by a fixed raster artifact.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterVertexLayout {
    /// The vertex shader derives its three fixed positions from `vertex_index`.
    None,
    /// One vertex is `float32x2 position` at byte zero followed by
    /// `unorm8x4 color` at byte eight, for a 12-byte stride.
    PositionFloat32x2ColorUnorm8x4,
}

/// Portable identity of one fixed raster artifact.
///
/// It describes the source-level recipe shared by DX12 and Vulkan, rather
/// than their intentionally different native binaries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RasterArtifactIdentity {
    /// Hash of the complete embedded WGSL module.
    pub module_source_hash: u64,
    /// Selected vertex entry point.
    pub vertex_entry_point: &'static str,
    /// Selected fragment entry point.
    pub fragment_entry_point: &'static str,
    /// Target format required by this artifact.
    pub target_format: TextureFormat,
    /// Complete non-configurable vertex layout of this artifact.
    pub vertex_layout: RasterVertexLayout,
    /// Bytes between vertices; zero means the fixed shader uses vertex index.
    pub vertex_stride: u32,
    /// Version of the fixed vertex/index/target recipe.
    pub recipe_version: u32,
}

impl RasterKernel {
    /// Returns the fixed vertex entry point.
    pub const fn vertex_entry_point(self) -> &'static str {
        match self {
            Self::Triangle => "triangle_vertex",
            Self::IndexedPositionColor => "position_color_vertex",
        }
    }

    /// Returns the shared fixed fragment entry point.
    pub const fn fragment_entry_point(self) -> &'static str {
        "color_fragment"
    }

    /// Returns the target format supported by every fixed artifact.
    pub const fn target_format(self) -> TextureFormat {
        TextureFormat::Rgba8Unorm
    }

    /// Returns the byte stride of the fixed vertex recipe.
    pub const fn vertex_stride(self) -> u32 {
        match self {
            Self::Triangle => 0,
            Self::IndexedPositionColor => 12,
        }
    }

    /// Returns the complete non-configurable vertex layout.
    pub const fn vertex_layout(self) -> RasterVertexLayout {
        match self {
            Self::Triangle => RasterVertexLayout::None,
            Self::IndexedPositionColor => RasterVertexLayout::PositionFloat32x2ColorUnorm8x4,
        }
    }

    /// Returns the portable identity recorded by conformance fixtures.
    pub const fn portable_identity(self) -> RasterArtifactIdentity {
        RasterArtifactIdentity {
            module_source_hash: fnv1a64(self.wgsl_source().as_bytes()),
            vertex_entry_point: self.vertex_entry_point(),
            fragment_entry_point: self.fragment_entry_point(),
            target_format: self.target_format(),
            vertex_layout: self.vertex_layout(),
            vertex_stride: self.vertex_stride(),
            recipe_version: 1,
        }
    }

    pub(crate) const fn wgsl_source(self) -> &'static str {
        match self {
            Self::Triangle => {
                "struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         @location(0) color: vec4<f32>,\n\
         };\n\
         @vertex fn triangle_vertex(@builtin(vertex_index) index: u32) -> VertexOutput {\n\
             if (index == 0u) {\n\
                 return VertexOutput(vec4(-0.70, -0.60, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             } else if (index == 1u) {\n\
                 return VertexOutput(vec4(0.70, -0.60, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             } else {\n\
                 return VertexOutput(vec4(0.0, 0.70, 0.0, 1.0), vec4(64.0 / 255.0, 160.0 / 255.0, 1.0, 1.0));\n\
             }\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }"
            }
            Self::IndexedPositionColor => {
                "struct VertexInput {\n\
         @location(0) position: vec2<f32>,\n\
         @location(1) color: vec4<f32>,\n\
         };\n\
         struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         @location(0) color: vec4<f32>,\n\
         };\n\
         @vertex fn position_color_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(vec4(input.position, 0.0, 1.0), input.color);\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return input.color; }"
            }
        }
    }
}

struct RasterPipelineShared {
    _native: crate::imp::NativeRasterPipeline,
    kernel: RasterKernel,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque, device-affine pipeline for one fixed raster artifact.
#[derive(Clone)]
pub struct RasterPipeline(Arc<RasterPipelineShared>);

/// A cloneable strong lease for a raster pipeline and its native artifacts.
#[derive(Clone)]
pub struct RasterPipelineLease(Arc<RasterPipelineShared>);

struct TexturePackBindingsShared {
    _native: crate::imp::NativeTexturePackBindings,
    pipeline: ComputePipeline,
    _texture: TextureLease,
    _buffer: BufferLease,
    offset: u64,
    size: u64,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed X01 bindings: one complete sampled `Rgba8Unorm` texture and one RW
/// storage buffer receiving row-major packed pixels.
#[derive(Clone)]
pub struct TexturePackBindings(Arc<TexturePackBindingsShared>);

/// A cloneable strong lease for an X01 binding object and all its inputs.
#[derive(Clone)]
pub struct TexturePackBindingsLease(Arc<TexturePackBindingsShared>);

/// A type-erased strong lease retained by native frame submissions.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ResourceLease {
    /// Retains one buffer allocation.
    Buffer(BufferLease),
    /// Retains one texture allocation.
    Texture(TextureLease),
    /// Retains a native compute pipeline, module, and layout.
    ComputePipeline(ComputePipelineLease),
    /// Retains a native compute binding object, pipeline, and buffer.
    ComputeBindings(ComputeBindingsLease),
    /// Retains a native raster pipeline, module, and layout.
    RasterPipeline(RasterPipelineLease),
    /// Retains the closed sampled-texture-to-storage-buffer binding object.
    TexturePackBindings(TexturePackBindingsLease),
}

macro_rules! resource_accessors {
    ($resource:ident, $lease:ident, $shared:ident, $desc:ty, $usage:ty) => {
        impl $resource {
            /// Returns the descriptor validated at creation.
            pub fn descriptor(&self) -> $desc {
                self.0.descriptor
            }
            /// Returns operations proven by the final native creation facts.
            ///
            /// This is an authorization upper bound, not necessarily the exact
            /// requested set. Native normalization may widen a request; for
            /// example, buffer storage write is reported as storage read/write.
            pub fn allowed_usage(&self) -> $usage {
                self.0.allowed_usage
            }
            /// Returns this physical generation's opaque identity.
            pub fn identity(&self) -> PhysicalResourceIdentity {
                self.0.identity
            }
            /// Returns the owning device identity.
            pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
                self.0.device
            }
            /// Acquires a token that keeps the native object and device alive.
            pub fn lease(&self) -> $lease {
                $lease(Arc::clone(&self.0))
            }
        }
        impl fmt::Debug for $resource {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($resource))
                    .field("descriptor", &self.0.descriptor)
                    .field("allowed_usage", &self.0.allowed_usage)
                    .field("identity", &self.0.identity)
                    .finish_non_exhaustive()
            }
        }
        impl fmt::Debug for $lease {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($lease))
                    .field(&self.0.identity)
                    .finish()
            }
        }
    };
}

resource_accessors!(
    Buffer,
    BufferLease,
    BufferShared,
    BufferDescriptor,
    BufferUsage
);
resource_accessors!(
    Texture,
    TextureLease,
    TextureShared,
    TextureDescriptor,
    TextureUsage
);

impl BufferLease {
    #[allow(
        dead_code,
        reason = "the test-only exported-buffer wrapper consumes this exact lease identity"
    )]
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    #[allow(
        dead_code,
        reason = "the test-only exported-buffer wrapper consumes this exact lease identity"
    )]
    pub(crate) fn identity(&self) -> PhysicalResourceIdentity {
        self.0.identity
    }
}

impl TextureLease {
    #[allow(
        dead_code,
        reason = "the test-only exported-texture wrapper consumes this exact lease identity"
    )]
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }

    #[allow(
        dead_code,
        reason = "the test-only exported-texture wrapper consumes this exact lease identity"
    )]
    pub(crate) fn identity(&self) -> PhysicalResourceIdentity {
        self.0.identity
    }
}

impl From<BufferLease> for ResourceLease {
    fn from(value: BufferLease) -> Self {
        Self::Buffer(value)
    }
}

impl From<TextureLease> for ResourceLease {
    fn from(value: TextureLease) -> Self {
        Self::Texture(value)
    }
}

impl From<ComputePipelineLease> for ResourceLease {
    fn from(value: ComputePipelineLease) -> Self {
        Self::ComputePipeline(value)
    }
}

impl From<ComputeBindingsLease> for ResourceLease {
    fn from(value: ComputeBindingsLease) -> Self {
        Self::ComputeBindings(value)
    }
}

impl From<RasterPipelineLease> for ResourceLease {
    fn from(value: RasterPipelineLease) -> Self {
        Self::RasterPipeline(value)
    }
}

impl From<TexturePackBindingsLease> for ResourceLease {
    fn from(value: TexturePackBindingsLease) -> Self {
        Self::TexturePackBindings(value)
    }
}

impl Buffer {
    pub(crate) fn native(&self) -> &crate::imp::OwnedBuffer {
        &self.0._native
    }
}

impl Texture {
    pub(crate) fn native(&self) -> &crate::imp::OwnedTexture {
        &self.0._native
    }
}

impl ComputePipeline {
    /// Returns the fixed kernel selected during creation.
    pub fn kernel(&self) -> ComputeKernel {
        self.0.kernel
    }
    pub(crate) fn same_object(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the native pipeline and its artifacts.
    pub fn lease(&self) -> ComputePipelineLease {
        ComputePipelineLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeComputePipeline {
        &self.0._native
    }
}

impl ComputeBindings {
    /// Returns the pipeline this binding object was created for.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.0.pipeline
    }
    /// Returns the exact authorized storage-buffer byte range.
    pub fn range(&self) -> (u64, u64) {
        (self.0.offset, self.0.size)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining all native objects referenced by this binding.
    pub fn lease(&self) -> ComputeBindingsLease {
        ComputeBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeComputeBindings {
        &self.0._native
    }
}

impl RasterPipeline {
    /// Returns the fixed artifact selected during creation.
    pub fn kernel(&self) -> RasterKernel {
        self.0.kernel
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the native pipeline and its artifacts.
    pub fn lease(&self) -> RasterPipelineLease {
        RasterPipelineLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterPipeline {
        &self.0._native
    }
}

impl TexturePackBindings {
    /// Returns the `TexturePackRgba8` pipeline selected during creation.
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.0.pipeline
    }
    /// Returns the exact storage-buffer range receiving packed pixels.
    pub fn range(&self) -> (u64, u64) {
        (self.0.offset, self.0.size)
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining all native objects referenced by this binding.
    pub fn lease(&self) -> TexturePackBindingsLease {
        TexturePackBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeTexturePackBindings {
        &self.0._native
    }
}

impl fmt::Debug for ComputePipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePipeline")
            .field("kernel", &self.0.kernel)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for ComputeBindings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputeBindings")
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for ComputePipelineLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ComputePipelineLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}
impl fmt::Debug for ComputeBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ComputeBindingsLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}

impl fmt::Debug for RasterPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RasterPipeline")
            .field("kernel", &self.0.kernel)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for RasterPipelineLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RasterPipelineLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}
impl fmt::Debug for TexturePackBindings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TexturePackBindings")
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for TexturePackBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TexturePackBindingsLease")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}

impl Device {
    /// Creates one fixed-artifact compute pipeline for this device.
    pub fn create_compute_pipeline(
        &self,
        kernel: ComputeKernel,
    ) -> Result<ComputePipeline, ComputeCreateError> {
        validate_compute_workgroup_limits(
            kernel.workgroup_size(),
            self.capabilities.max_compute_workgroup_size,
            self.capabilities.max_compute_invocations_per_workgroup,
        )?;
        #[cfg(windows)]
        let native = crate::imp::create_compute_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.entry_point(),
        )
        .map_err(map_compute_pipeline_create_error)?;
        #[cfg(not(windows))]
        let native = crate::imp::create_compute_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.entry_point(),
        )
        .map_err(ComputeCreateError::NativeObjectCreation)?;
        Ok(ComputePipeline(Arc::new(ComputePipelineShared {
            _native: native,
            kernel,
            device: self.identity,
        })))
    }

    /// Creates the only binding layout accepted by the fixed compute artifacts.
    pub fn create_compute_bindings(
        &self,
        pipeline: &ComputePipeline,
        buffer: &Buffer,
        offset: u64,
        size: u64,
    ) -> Result<ComputeBindings, ComputeCreateError> {
        if pipeline.kernel() == ComputeKernel::TexturePackRgba8 {
            return Err(ComputeCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity || buffer.device_identity() != self.identity
        {
            return Err(ComputeCreateError::ForeignDevice);
        }
        validate_compute_binding_range(
            offset,
            size,
            buffer.descriptor().buffer.size,
            self.capabilities.min_storage_buffer_offset_alignment,
            self.capabilities.max_storage_buffer_binding_size,
        )?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::StorageRead)
            || !buffer
                .allowed_usage()
                .contains(BufferUsageKind::StorageWrite)
        {
            return Err(ComputeCreateError::StorageUsageRequired);
        }
        let native = crate::imp::create_compute_bindings(
            &self.inner,
            pipeline.native(),
            buffer.native(),
            offset,
            size,
        )
        .map_err(ComputeCreateError::NativeFailure)?;
        Ok(ComputeBindings(Arc::new(ComputeBindingsShared {
            _native: native,
            pipeline: pipeline.clone(),
            _buffer: buffer.lease(),
            offset,
            size,
            device: self.identity,
        })))
    }

    /// Creates the only binding layout accepted by the X01 texture-pack artifact.
    ///
    /// The complete `Rgba8Unorm` texture is sampled with `textureLoad`; each
    /// pixel occupies exactly one little-endian RGBA8-packed `u32` in the
    /// authorized destination range.
    pub fn create_texture_pack_bindings(
        &self,
        pipeline: &ComputePipeline,
        texture: &Texture,
        buffer: &Buffer,
        offset: u64,
        size: u64,
    ) -> Result<TexturePackBindings, ComputeCreateError> {
        if pipeline.kernel() != ComputeKernel::TexturePackRgba8 {
            return Err(ComputeCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || buffer.device_identity() != self.identity
        {
            return Err(ComputeCreateError::ForeignDevice);
        }
        validate_texture_pack_texture(texture)?;
        validate_compute_binding_range(
            offset,
            size,
            buffer.descriptor().buffer.size,
            self.capabilities.min_storage_buffer_offset_alignment,
            self.capabilities.max_storage_buffer_binding_size,
        )?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::StorageRead)
            || !buffer
                .allowed_usage()
                .contains(BufferUsageKind::StorageWrite)
        {
            return Err(ComputeCreateError::StorageUsageRequired);
        }
        let required_size = texture_pack_required_size(texture.descriptor().texture)?;
        if size < required_size {
            return Err(ComputeCreateError::InvalidBindingRange);
        }
        let native = crate::imp::create_texture_pack_bindings(
            &self.inner,
            pipeline.native(),
            texture.native(),
            buffer.native(),
            offset,
            size,
        )
        .map_err(ComputeCreateError::NativeFailure)?;
        Ok(TexturePackBindings(Arc::new(TexturePackBindingsShared {
            _native: native,
            pipeline: pipeline.clone(),
            _texture: texture.lease(),
            _buffer: buffer.lease(),
            offset,
            size,
            device: self.identity,
        })))
    }

    /// Creates one fixed-artifact raster pipeline for this device.
    pub fn create_raster_pipeline(
        &self,
        kernel: RasterKernel,
    ) -> Result<RasterPipeline, RasterCreateError> {
        #[cfg(windows)]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel.vertex_stride() != 0,
        )
        .map_err(map_raster_pipeline_create_error)?;
        #[cfg(not(windows))]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel.vertex_stride() != 0,
        )
        .map_err(RasterCreateError::NativeObjectCreation)?;
        Ok(RasterPipeline(Arc::new(RasterPipelineShared {
            _native: native,
            kernel,
            device: self.identity,
        })))
    }
    /// Creates a native buffer after validating and lowering its portable contract.
    pub fn create_buffer(
        &self,
        descriptor: BufferDescriptor,
    ) -> Result<Buffer, ResourceCreateError> {
        validate_buffer(descriptor)?;
        let (native, allowed_usage) = crate::imp::create_buffer(&self.inner, descriptor)?;
        Ok(Buffer(Arc::new(BufferShared {
            _native: native,
            descriptor,
            allowed_usage,
            identity: PhysicalResourceIdentity::new(crate::next_identity()),
            device: self.identity,
        })))
    }

    /// Creates a native texture after validating and lowering its portable contract.
    pub fn create_texture(
        &self,
        descriptor: TextureDescriptor,
    ) -> Result<Texture, ResourceCreateError> {
        validate_texture(descriptor)?;
        let extent = descriptor.texture.extent;
        if descriptor.texture.dimension == TextureDimension::D2
            && (extent.width > self.capabilities.max_texture_dimension_2d
                || extent.height > self.capabilities.max_texture_dimension_2d)
        {
            return Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::ExceedsDeviceLimit,
            ));
        }
        let (native, allowed_usage) = crate::imp::create_texture(&self.inner, descriptor)?;
        Ok(Texture(Arc::new(TextureShared {
            _native: native,
            descriptor,
            allowed_usage,
            identity: PhysicalResourceIdentity::new(crate::next_identity()),
            device: self.identity,
        })))
    }
}

/// Checks the fixed shader declaration against raw native workgroup facts.
///
/// This is intentionally kept at the RHI boundary: the declaration is a
/// property of the fixed shader artifact, not of RenderGraph dispatch syntax.
fn validate_compute_workgroup_limits(
    workgroup_size: [u32; 3],
    maximum_size: [u32; 3],
    maximum_invocations: u32,
) -> Result<(), ComputeCreateError> {
    if workgroup_size
        .into_iter()
        .zip(maximum_size)
        .any(|(requested, maximum)| requested == 0 || requested > maximum)
    {
        return Err(ComputeCreateError::UnsupportedComputeLimits);
    }
    let invocations = workgroup_size
        .into_iter()
        .try_fold(1_u32, |total, component| total.checked_mul(component));
    if invocations.is_none_or(|total| total > maximum_invocations) {
        return Err(ComputeCreateError::UnsupportedComputeLimits);
    }
    Ok(())
}

/// Checks a fixed RW-storage binding against portable and native device facts.
///
/// The fixed WGSL artifacts require four-byte elements, while the native
/// offset must additionally meet the device's storage-buffer alignment.
fn validate_compute_binding_range(
    offset: u64,
    size: u64,
    buffer_size: u64,
    minimum_storage_offset_alignment: u32,
    maximum_storage_binding_size: u64,
) -> Result<(), ComputeCreateError> {
    let required_offset_alignment = u64::from(minimum_storage_offset_alignment.max(4));
    let end = offset
        .checked_add(size)
        .ok_or(ComputeCreateError::InvalidBindingRange)?;
    if size == 0
        || !offset.is_multiple_of(required_offset_alignment)
        || !size.is_multiple_of(4)
        || size > maximum_storage_binding_size
        || end > buffer_size
    {
        return Err(ComputeCreateError::InvalidBindingRange);
    }
    Ok(())
}

/// Validates the closed X01 sampled-texture side of the binding recipe.
fn validate_texture_pack_texture(texture: &Texture) -> Result<(), ComputeCreateError> {
    validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
}

/// Validates the descriptor facts the X01 texture binding cannot generalize.
fn validate_texture_pack_texture_desc(
    image: TextureDesc,
    allowed_usage: TextureUsage,
) -> Result<(), ComputeCreateError> {
    if image.dimension != TextureDimension::D2
        || image.format != TextureFormat::Rgba8Unorm
        || image.extent.depth != 1
        || image.mip_levels != 1
        || image.array_layers != 1
        || image.sample_count != 1
        || !allowed_usage.contains(TextureUsageKind::Sampled)
    {
        return Err(ComputeCreateError::BindingRecipeMismatch);
    }
    Ok(())
}

/// Returns the exact byte extent of X01's one-`u32`-per-pixel output.
fn texture_pack_required_size(image: TextureDesc) -> Result<u64, ComputeCreateError> {
    u64::from(image.extent.width)
        .checked_mul(u64::from(image.extent.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(ComputeCreateError::InvalidBindingRange)
}

fn invalid(resource: ResourceKind, reason: InvalidResourceReason) -> ResourceCreateError {
    ResourceCreateError::InvalidDescriptor { resource, reason }
}

fn validate_buffer(desc: BufferDescriptor) -> Result<(), ResourceCreateError> {
    if desc.buffer.size == 0 {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::ZeroSize,
        ));
    }
    if !buffer_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    Ok(())
}

fn buffer_has_usage(usage: BufferUsage) -> bool {
    [
        BufferUsageKind::Uniform,
        BufferUsageKind::StorageRead,
        BufferUsageKind::StorageWrite,
        BufferUsageKind::Vertex,
        BufferUsageKind::Index,
        BufferUsageKind::Indirect,
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

fn texture_has_usage(usage: TextureUsage) -> bool {
    [
        TextureUsageKind::Sampled,
        TextureUsageKind::StorageRead,
        TextureUsageKind::StorageWrite,
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::DepthStencilAttachment,
        TextureUsageKind::CopySource,
        TextureUsageKind::CopyDestination,
        TextureUsageKind::Present,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

fn validate_texture(desc: TextureDescriptor) -> Result<(), ResourceCreateError> {
    let d = desc.texture;
    if !texture_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    if desc.usage.contains(TextureUsageKind::Present) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::PresentRequiresSurface,
        ));
    }
    if d.extent.width == 0 || d.extent.height == 0 || d.extent.depth == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::ZeroExtent,
        ));
    }
    if d.array_layers == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidArrayLayers,
        ));
    }
    if d.dimension != TextureDimension::D2 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::UnsupportedDimension,
        ));
    }
    let dimensions_valid = match d.dimension {
        TextureDimension::D1 => false,
        TextureDimension::D2 => d.extent.depth == 1,
        TextureDimension::D3 => false,
        _ => false,
    };
    if !dimensions_valid {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidDimension,
        ));
    }
    let max_dimension = d.extent.width.max(d.extent.height).max(d.extent.depth);
    let max_mips = u32::BITS - max_dimension.leading_zeros();
    if d.mip_levels == 0 || d.mip_levels > max_mips {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidMipLevels,
        ));
    }
    if !matches!(d.sample_count, 1 | 2 | 4 | 8 | 16)
        || (d.sample_count > 1
            && (d.dimension != TextureDimension::D2 || d.mip_levels != 1 || d.array_layers != 1))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidSampleCount,
        ));
    }
    let depth = d.format == TextureFormat::Depth32Float;
    let color_attachment = desc.usage.contains(TextureUsageKind::ColorAttachment);
    let depth_attachment = desc
        .usage
        .contains(TextureUsageKind::DepthStencilAttachment);
    let storage = desc.usage.contains(TextureUsageKind::StorageRead)
        || desc.usage.contains(TextureUsageKind::StorageWrite);
    let copy = desc.usage.contains(TextureUsageKind::CopySource)
        || desc.usage.contains(TextureUsageKind::CopyDestination);
    if (depth && (color_attachment || storage))
        || (!depth && depth_attachment)
        || (d.sample_count > 1 && (storage || copy))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::IncompatibleUsage,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_rendergraph::{Extent3d, TextureDimension};

    #[cfg(windows)]
    #[test]
    fn compute_pipeline_creation_stages_map_to_public_error_variants() {
        let cases = [
            (
                crate::imp::ComputePipelineCreateError::ShaderValidation("parse".into()),
                ComputeCreateError::ShaderValidation("parse".into()),
            ),
            (
                crate::imp::ComputePipelineCreateError::ShaderCompilation("compile".into()),
                ComputeCreateError::ShaderCompilation("compile".into()),
            ),
            (
                crate::imp::ComputePipelineCreateError::NativeObjectCreation("layout".into()),
                ComputeCreateError::NativeObjectCreation("layout".into()),
            ),
        ];
        for (native, public) in cases {
            assert_eq!(map_compute_pipeline_create_error(native), public);
        }
    }

    #[cfg(windows)]
    #[test]
    fn raster_pipeline_creation_stages_map_to_public_error_variants() {
        let cases = [
            (
                crate::imp::RasterPipelineCreateError::ShaderValidation("parse".into()),
                RasterCreateError::ShaderValidation("parse".into()),
            ),
            (
                crate::imp::RasterPipelineCreateError::ShaderCompilation("compile".into()),
                RasterCreateError::ShaderCompilation("compile".into()),
            ),
            (
                crate::imp::RasterPipelineCreateError::NativeObjectCreation("layout".into()),
                RasterCreateError::NativeObjectCreation("layout".into()),
            ),
        ];
        for (native, public) in cases {
            assert_eq!(map_raster_pipeline_create_error(native), public);
        }
    }

    #[test]
    fn rejects_zero_buffer_and_empty_usage() {
        let empty = BufferDescriptor {
            buffer: BufferDesc { size: 0 },
            usage: BufferUsage::empty(),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_buffer(empty),
            Err(invalid(
                ResourceKind::Buffer,
                InvalidResourceReason::ZeroSize
            ))
        );
        assert_eq!(
            validate_buffer(BufferDescriptor {
                buffer: BufferDesc { size: 4 },
                ..empty
            }),
            Err(invalid(
                ResourceKind::Buffer,
                InvalidResourceReason::EmptyUsage
            ))
        );
    }

    #[test]
    fn rejects_surface_and_incompatible_texture_contracts() {
        let base = TextureDescriptor {
            texture: TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
            },
            usage: TextureUsage::empty().with(TextureUsageKind::Present),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_texture(base),
            Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::PresentRequiresSurface
            ))
        );
        let depth_color = TextureDescriptor {
            texture: TextureDesc {
                format: TextureFormat::Depth32Float,
                ..base.texture
            },
            usage: TextureUsage::empty().with(TextureUsageKind::ColorAttachment),
            ..base
        };
        assert_eq!(
            validate_texture(depth_color),
            Err(invalid(
                ResourceKind::Texture,
                InvalidResourceReason::IncompatibleUsage
            ))
        );
    }

    #[test]
    fn fixed_workgroup_requires_each_native_dimension_and_total_invocations() {
        assert_eq!(
            validate_compute_workgroup_limits([64, 1, 1], [63, 1, 1], 64),
            Err(ComputeCreateError::UnsupportedComputeLimits)
        );
        assert_eq!(
            validate_compute_workgroup_limits([64, 1, 1], [64, 1, 1], 63),
            Err(ComputeCreateError::UnsupportedComputeLimits)
        );
        assert_eq!(
            validate_compute_workgroup_limits([64, 0, 1], [64, 1, 1], 64),
            Err(ComputeCreateError::UnsupportedComputeLimits)
        );
        assert_eq!(
            validate_compute_workgroup_limits([64, 1, 1], [64, 1, 1], 64),
            Ok(())
        );
    }

    #[test]
    fn storage_bindings_require_native_alignment_and_limited_in_bounds_ranges() {
        let validate = |offset, size, buffer_size, alignment, maximum| {
            validate_compute_binding_range(offset, size, buffer_size, alignment, maximum)
        };
        assert_eq!(
            validate(4, 256, 512, 256, 256),
            Err(ComputeCreateError::InvalidBindingRange)
        );
        assert_eq!(validate(0, 256, 256, 256, 256), Ok(()));
        assert_eq!(validate(256, 256, 512, 256, 256), Ok(()));
        assert_eq!(
            validate(0, 260, 512, 256, 256),
            Err(ComputeCreateError::InvalidBindingRange)
        );
        assert_eq!(
            validate(0, 256, 512, 256, 252),
            Err(ComputeCreateError::InvalidBindingRange)
        );
        assert_eq!(
            validate(u64::MAX - 3, 4, u64::MAX, 4, u64::MAX),
            Err(ComputeCreateError::InvalidBindingRange)
        );
    }

    #[test]
    fn portable_compute_identity_shares_module_but_distinguishes_entries() {
        let add = ComputeKernel::WrappingAdd.portable_identity();
        let multiply = ComputeKernel::WrappingMultiply.portable_identity();
        assert_eq!(add.module_source_hash, multiply.module_source_hash);
        assert_ne!(add.entry_point, multiply.entry_point);
        assert_eq!(add.workgroup_size, [64, 1, 1]);
        assert_eq!(add.binding_recipe_version, 1);
    }

    #[test]
    fn texture_pack_is_a_distinct_closed_compute_recipe() {
        let pack = ComputeKernel::TexturePackRgba8.portable_identity();
        let add = ComputeKernel::WrappingAdd.portable_identity();
        assert_ne!(pack.module_source_hash, add.module_source_hash);
        assert_eq!(pack.entry_point, "pack_rgba8");
        assert_eq!(pack.workgroup_size, [8, 8, 1]);
        assert_eq!(pack.binding_recipe_version, 2);
    }

    #[test]
    fn raster_identities_are_portable_but_recipes_remain_distinct() {
        let triangle = RasterKernel::Triangle.portable_identity();
        let indexed = RasterKernel::IndexedPositionColor.portable_identity();
        assert_eq!(triangle.target_format, TextureFormat::Rgba8Unorm);
        assert_eq!(triangle.vertex_stride, 0);
        assert_eq!(triangle.vertex_layout, RasterVertexLayout::None);
        assert_eq!(indexed.vertex_stride, 12);
        assert_eq!(
            indexed.vertex_layout,
            RasterVertexLayout::PositionFloat32x2ColorUnorm8x4
        );
        assert_ne!(triangle.module_source_hash, indexed.module_source_hash);
        assert_eq!(triangle.fragment_entry_point, indexed.fragment_entry_point);
    }

    #[test]
    fn texture_pack_accepts_only_full_single_sampled_rgba8_images() {
        let image = TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 7,
                height: 3,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let sampled = TextureUsage::empty().with(TextureUsageKind::Sampled);
        assert_eq!(validate_texture_pack_texture_desc(image, sampled), Ok(()));
        assert_eq!(texture_pack_required_size(image), Ok(84));
        assert_eq!(
            validate_texture_pack_texture_desc(
                TextureDesc {
                    mip_levels: 2,
                    ..image
                },
                sampled
            ),
            Err(ComputeCreateError::BindingRecipeMismatch)
        );
        assert_eq!(
            validate_texture_pack_texture_desc(image, TextureUsage::empty()),
            Err(ComputeCreateError::BindingRecipeMismatch)
        );
    }
}
