//! Safe owned-resource creation and lifetime contracts.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionStatus, IndexFormat,
    PhysicalResourceIdentity, ResourceAccessState, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage, TextureUsageKind,
};

use crate::{Backend, Device, NativeCompletion};

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

/// The stage at which an immutable buffer upload reached the native boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BufferUploadStage {
    /// The private staging allocation or host write failed.
    Staging,
    /// Recording the fixed copy operation failed.
    Recording,
    /// The queue rejected the submission before it was accepted.
    SubmitRejected,
    /// Querying or waiting for an accepted completion failed.
    Completion,
}

/// The stage at which an immutable texture upload reached the native boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TextureUploadStage {
    /// Creating or writing private row-padded staging storage failed.
    Staging,
    /// Recording the fixed buffer-to-texture copy failed.
    Recording,
    /// The queue rejected the submission before accepting it.
    SubmitRejected,
    /// Querying or waiting for an accepted completion failed.
    Completion,
}

/// A stable reason why an immutable buffer upload request was rejected before native work.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidBufferUploadReason {
    /// The source byte slice was empty.
    EmptyData,
    /// The source byte length was not a multiple of the portable copy alignment.
    DataLengthNotCopyAligned,
    /// The buffer descriptor size was not exactly the source byte length.
    DescriptorSizeMismatch,
    /// The upload slice accepts only device-local destination buffers.
    MemoryPolicyUnsupported,
    /// The destination declaration did not request copy-destination usage.
    CopyDestinationUsageRequired,
    /// Test-only observation requires the finalized buffer to allow copy-source use.
    CopySourceUsageRequired,
}

/// A stable reason why an immutable RGBA8 texture request was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidTextureUploadReason {
    /// Only a two-dimensional image is accepted.
    DimensionUnsupported,
    /// Only `Rgba8Unorm` is accepted.
    FormatUnsupported,
    /// The closed upload has exactly one mip level.
    MipLevelsUnsupported,
    /// The closed upload has exactly one array layer.
    ArrayLayersUnsupported,
    /// The closed upload is not multisampled.
    SampleCountUnsupported,
    /// The closed upload only owns device-local storage.
    MemoryPolicyUnsupported,
    /// Required copy-destination usage is missing.
    CopyDestinationUsageRequired,
    /// Required sampled usage is missing.
    SampledUsageRequired,
    /// The closed upload declaration requested an operation outside its
    /// production contract. Test-only readback may additionally request
    /// `CopySource`; no other widening is accepted.
    UnexpectedUsage,
    /// The supplied bytes are not exactly tight RGBA8 image bytes.
    DataLengthMismatch,
    /// The declared extent cannot be represented as a tight byte length.
    DataLengthOverflow,
}

/// Why an immutable buffer upload could not be started or observed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BufferUploadError {
    /// The portable upload request was malformed.
    InvalidRequest(InvalidBufferUploadReason),
    /// Creating the destination buffer failed.
    Resource(ResourceCreateError),
    /// An observation request used a buffer from a different native device.
    ForeignDevice,
    /// A private native boundary operation failed.
    Native {
        /// Backend selected by the owning device.
        backend: Backend,
        /// Upload stage that failed.
        stage: BufferUploadStage,
        /// Native diagnostic retained without exposing HAL types.
        reason: String,
    },
}

impl fmt::Display for BufferUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => {
                write!(f, "invalid immutable buffer upload: {reason:?}")
            }
            Self::Resource(error) => write!(f, "immutable buffer upload resource error: {error}"),
            Self::ForeignDevice => f.write_str("immutable buffer upload used a foreign device"),
            Self::Native {
                backend,
                stage,
                reason,
            } => write!(
                f,
                "{backend:?} immutable buffer upload {stage:?} failed: {reason}"
            ),
        }
    }
}

impl std::error::Error for BufferUploadError {}

/// Why an immutable texture upload could not be started or observed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TextureUploadError {
    /// The portable upload request was malformed.
    InvalidRequest(InvalidTextureUploadReason),
    /// Creating the destination texture failed.
    Resource(ResourceCreateError),
    /// An observation request used a texture from a different native device.
    ForeignDevice,
    /// A private native boundary operation failed.
    Native {
        /// Backend selected by the owning device.
        backend: Backend,
        /// Upload stage that failed.
        stage: TextureUploadStage,
        /// Native diagnostic retained without exposing HAL types.
        reason: String,
    },
}

impl fmt::Display for TextureUploadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(reason) => {
                write!(f, "invalid immutable texture upload: {reason:?}")
            }
            Self::Resource(error) => write!(f, "immutable texture upload resource error: {error}"),
            Self::ForeignDevice => f.write_str("immutable texture upload used a foreign device"),
            Self::Native {
                backend,
                stage,
                reason,
            } => write!(
                f,
                "{backend:?} immutable texture upload {stage:?} failed: {reason}"
            ),
        }
    }
}

impl std::error::Error for TextureUploadError {}

/// An immutable buffer upload accepted by the native queue but not yet finalized.
///
/// This value retains the destination and the accepted submission. Dropping it
/// never releases potentially in-flight native storage early: the private
/// completion owns the staging allocation and applies the accepted-unknown
/// quarantine contract when completion cannot be proven.
pub struct PendingBufferUpload {
    buffer: Option<Buffer>,
    completion: NativeCompletion,
    backend: Backend,
}

/// A non-complete immutable upload returned by [`PendingBufferUpload::finalize`].
pub struct IncompleteBufferUpload {
    upload: PendingBufferUpload,
    status: CompletionStatus,
}

/// An immutable buffer whose initial contents are available to later GPU work.
///
/// Its only published incoming/outgoing state is [`ResourceAccessState::CopyDestination`].
#[derive(Clone)]
pub struct UploadedBuffer {
    buffer: Buffer,
}

/// An immutable texture upload accepted by the native queue but not yet finalized.
pub struct PendingTextureUpload {
    texture: Option<Texture>,
    completion: NativeCompletion,
    backend: Backend,
}

/// A non-complete texture upload returned by [`PendingTextureUpload::finalize`].
pub struct IncompleteTextureUpload {
    upload: PendingTextureUpload,
    status: CompletionStatus,
}

/// An immutable RGBA8 texture whose contents are available to later GPU work.
#[derive(Clone)]
pub struct UploadedTexture {
    texture: Texture,
}

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
    /// The selected adapter cannot filter the requested fixed texture format.
    TextureFormatNotFilterable {
        /// The fixed texture format that requires filtering support.
        format: TextureFormat,
    },
    /// The pipeline, attachment, or buffer operation crossed device identities.
    ForeignDevice,
    /// The closed raster uniform recipe received an incompatible object.
    BindingRecipeMismatch,
    /// The uniform buffer is not exactly the fixed 80-byte whole binding.
    InvalidBindingRange,
    /// The buffer's native creation facts do not authorize uniform reads.
    UniformUsageRequired,
    /// The position stream is not the exact closed f32x3 vertex range.
    InvalidPositionStreamRange,
    /// The texture-coordinate stream is not the exact closed f32x2 vertex range.
    InvalidTextureCoordinateStreamRange,
    /// A closed vertex stream was created without vertex usage.
    VertexUsageRequired,
    /// The two closed vertex streams do not describe the requested vertex count.
    VertexStreamCountMismatch,
    /// Native creation of a validated fixed raster binding failed.
    NativeFailure(String),
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
            Self::TextureFormatNotFilterable { format } => {
                write!(f, "raster texture format is not filterable: {format:?}")
            }
            Self::ForeignDevice => f.write_str("raster objects belong to different devices"),
            Self::BindingRecipeMismatch => {
                f.write_str("raster bindings do not match the fixed artifact")
            }
            Self::InvalidBindingRange => f.write_str("invalid fixed raster uniform range"),
            Self::UniformUsageRequired => {
                f.write_str("raster uniform binding requires uniform usage")
            }
            Self::InvalidPositionStreamRange => {
                f.write_str("invalid fixed raster position stream range")
            }
            Self::InvalidTextureCoordinateStreamRange => {
                f.write_str("invalid fixed raster texture-coordinate stream range")
            }
            Self::VertexUsageRequired => {
                f.write_str("fixed raster vertex streams require vertex usage")
            }
            Self::VertexStreamCountMismatch => {
                f.write_str("fixed raster vertex stream count mismatch")
            }
            Self::NativeFailure(reason) => {
                write!(f, "native raster binding creation failed: {reason}")
            }
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

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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

/// The deterministic raster artifacts supported by this milestone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterKernel {
    /// A fixed, non-indexed triangle with no resource bindings.
    Triangle,
    /// An indexed mesh using `float32x2 position + unorm8x4 color` vertices.
    IndexedPositionColor,
    /// An indexed clip-space mesh using `float32x3` positions and a fixed
    /// opaque fragment color.
    IndexedPositionFloat32x3,
    /// An indexed model-identity mesh using `float32x3` positions and exactly
    /// one 80-byte view-projection plus base-color uniform binding.
    IndexedPositionFloat32x3CameraMaterial,
    /// An indexed camera/material mesh with one closed `textureLoad` RGBA8 binding.
    IndexedPositionFloat32x3CameraMaterialTexture,
    /// An indexed camera/material mesh with separate f32x3 position and f32x2
    /// UV streams plus one closed `textureLoad` RGBA8 binding.
    IndexedPositionFloat32x3CameraMaterialTextureUv,
    /// An indexed camera/material mesh with explicit UV streams and a fixed
    /// filtering linear-clamp sampler at explicit mip level zero.
    IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
}

/// The non-configurable vertex layout selected by a fixed raster artifact.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterVertexLayout {
    /// The vertex shader derives its three fixed positions from `vertex_index`.
    None,
    /// One vertex is `float32x2 position` at byte zero followed by
    /// `unorm8x4 color` at byte eight, for a 12-byte stride.
    PositionFloat32x2ColorUnorm8x4,
    /// One vertex is `float32x3 position` at byte zero, for a 12-byte stride.
    PositionFloat32x3,
    /// Slot zero is tightly packed f32x3 position and slot one is tightly
    /// packed f32x2 texture coordinates.
    PositionFloat32x3AndTextureCoordinateFloat32x2,
}

/// Shader-stage visibility recorded by a closed raster binding identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterBindingVisibility {
    /// The binding is visible to both vertex and fragment stages.
    VertexFragment,
    /// The binding is visible only to the fragment stage.
    Fragment,
}

/// Texture sample type recorded by a closed raster binding identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterTextureSampleType {
    /// A normalized floating-point sampled texture read as `f32`.
    Float,
}

/// Sampler type recorded by a closed raster artifact identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerBindingType {
    /// A filtering sampler used with a float sampled texture.
    Filtering,
}

/// Fixed filter choice recorded by a closed raster sampler identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerFilter {
    /// Linear interpolation between neighboring texels.
    Linear,
    /// Nearest mip selection.
    Nearest,
}

/// Fixed coordinate addressing recorded by a closed raster sampler identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RasterSamplerAddressMode {
    /// Clamp each coordinate to the texture edge.
    ClampToEdge,
}

/// Portable identity of one fixed raster artifact.
///
/// It describes the source-level recipe shared by DX12 and Vulkan, rather
/// than their intentionally different native binaries.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
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
    /// The required index element format, or `None` for the non-indexed recipe.
    pub index_format: Option<IndexFormat>,
    /// Number of fixed bind-group entries; zero means no raster bindings.
    pub binding_count: u32,
    /// Minimum byte size of the sole fixed uniform binding, or zero when none.
    pub uniform_binding_size: u64,
    /// Uniform binding number for a fixed uniform recipe.
    pub uniform_binding: Option<u32>,
    /// Shader-stage visibility of the fixed uniform binding.
    pub uniform_visibility: Option<RasterBindingVisibility>,
    /// Version of the fixed binding layout; zero means no binding layout.
    pub binding_recipe_version: u32,
    /// Texture ABI schema version; zero means no texture binding.
    pub texture_binding_recipe_version: u32,
    /// Texture binding number for the closed textured recipe.
    pub texture_binding: Option<u32>,
    /// Texture dimension for the closed textured recipe.
    pub texture_dimension: Option<TextureDimension>,
    /// Texture format for the closed textured recipe.
    pub texture_format: Option<TextureFormat>,
    /// Shader-stage visibility of the closed texture binding.
    pub texture_visibility: Option<RasterBindingVisibility>,
    /// Sample type exposed by the closed texture binding.
    pub texture_sample_type: Option<RasterTextureSampleType>,
    /// The only mip level read by the closed texture recipe.
    pub texture_mip_level: Option<u32>,
    /// Version of the fixed texture-coordinate mapping recipe.
    pub texture_mapping_recipe_version: u32,
    /// Slot-one layout, only present for the explicit-UV recipe.
    pub texture_coordinate_vertex_layout: Option<RasterVertexLayout>,
    /// Slot-one byte stride, only present for the explicit-UV recipe.
    pub texture_coordinate_vertex_stride: Option<u32>,
    /// Shader location consumed from slot one, only present for the explicit-UV recipe.
    pub texture_coordinate_shader_location: Option<u32>,
    /// Sampler binding number for the fixed linear-clamp recipe.
    pub sampler_binding: Option<u32>,
    /// Sampler ABI schema version; zero means this artifact has no sampler.
    pub sampler_binding_recipe_version: u32,
    /// Sampler type for the fixed linear-clamp recipe.
    pub sampler_binding_type: Option<RasterSamplerBindingType>,
    /// Minification filter for the fixed linear-clamp recipe.
    pub sampler_min_filter: Option<RasterSamplerFilter>,
    /// Magnification filter for the fixed linear-clamp recipe.
    pub sampler_mag_filter: Option<RasterSamplerFilter>,
    /// Mipmap filter for the fixed linear-clamp recipe.
    pub sampler_mipmap_filter: Option<RasterSamplerFilter>,
    /// U coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_u: Option<RasterSamplerAddressMode>,
    /// V coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_v: Option<RasterSamplerAddressMode>,
    /// W coordinate address mode for the fixed linear-clamp recipe.
    pub sampler_address_mode_w: Option<RasterSamplerAddressMode>,
    /// Explicit mip level sampled by the fixed linear-clamp recipe.
    pub sampler_explicit_mip_level: Option<u32>,
    /// Version of the fixed vertex/index/target recipe.
    pub recipe_version: u32,
}

impl fmt::Debug for RasterArtifactIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut identity = formatter.debug_struct("RasterArtifactIdentity");
        identity
            .field("module_source_hash", &self.module_source_hash)
            .field("vertex_entry_point", &self.vertex_entry_point)
            .field("fragment_entry_point", &self.fragment_entry_point)
            .field("target_format", &self.target_format)
            .field("vertex_layout", &self.vertex_layout)
            .field("vertex_stride", &self.vertex_stride)
            .field("index_format", &self.index_format);
        // Preserve the exact 0.2.1 U02 artifact representation. Binding facts
        // are an additive 0.2.2 identity component only for artifacts that
        // actually declare a binding recipe.
        if self.binding_count != 0
            || self.uniform_binding_size != 0
            || self.binding_recipe_version != 0
        {
            identity
                .field("binding_count", &self.binding_count)
                .field("uniform_binding_size", &self.uniform_binding_size)
                .field("binding_recipe_version", &self.binding_recipe_version);
        }
        if self.texture_binding_recipe_version != 0 {
            identity
                .field(
                    "texture_binding_recipe_version",
                    &self.texture_binding_recipe_version,
                )
                .field("texture_binding", &self.texture_binding)
                .field("texture_dimension", &self.texture_dimension)
                .field("texture_format", &self.texture_format)
                .field("uniform_binding", &self.uniform_binding)
                .field("uniform_visibility", &self.uniform_visibility)
                .field("texture_visibility", &self.texture_visibility)
                .field("texture_sample_type", &self.texture_sample_type)
                .field("texture_mip_level", &self.texture_mip_level)
                .field(
                    "texture_mapping_recipe_version",
                    &self.texture_mapping_recipe_version,
                );
        }
        if self.texture_coordinate_vertex_layout.is_some() {
            identity
                .field(
                    "texture_coordinate_vertex_layout",
                    &self.texture_coordinate_vertex_layout,
                )
                .field(
                    "texture_coordinate_vertex_stride",
                    &self.texture_coordinate_vertex_stride,
                )
                .field(
                    "texture_coordinate_shader_location",
                    &self.texture_coordinate_shader_location,
                );
        }
        if self.sampler_binding.is_some() {
            identity
                .field(
                    "sampler_binding_recipe_version",
                    &self.sampler_binding_recipe_version,
                )
                .field("sampler_binding", &self.sampler_binding)
                .field("sampler_binding_type", &self.sampler_binding_type)
                .field("sampler_min_filter", &self.sampler_min_filter)
                .field("sampler_mag_filter", &self.sampler_mag_filter)
                .field("sampler_mipmap_filter", &self.sampler_mipmap_filter)
                .field("sampler_address_mode_u", &self.sampler_address_mode_u)
                .field("sampler_address_mode_v", &self.sampler_address_mode_v)
                .field("sampler_address_mode_w", &self.sampler_address_mode_w)
                .field(
                    "sampler_explicit_mip_level",
                    &self.sampler_explicit_mip_level,
                );
        }
        identity
            .field("recipe_version", &self.recipe_version)
            .finish()
    }
}

impl RasterKernel {
    /// Returns the fixed vertex entry point.
    pub const fn vertex_entry_point(self) -> &'static str {
        match self {
            Self::Triangle => "triangle_vertex",
            Self::IndexedPositionColor => "position_color_vertex",
            Self::IndexedPositionFloat32x3 => "position_f32x3_vertex",
            Self::IndexedPositionFloat32x3CameraMaterial => "camera_material_vertex",
            Self::IndexedPositionFloat32x3CameraMaterialTexture => "camera_material_texture_vertex",
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                "camera_material_texture_uv_vertex"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "camera_material_texture_uv_linear_clamp_vertex"
            }
        }
    }

    /// Returns the shared fixed fragment entry point.
    pub const fn fragment_entry_point(self) -> &'static str {
        match self {
            Self::IndexedPositionFloat32x3CameraMaterialTexture
            | Self::IndexedPositionFloat32x3CameraMaterialTextureUv => "texture_color_fragment",
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "linear_clamp_texture_color_fragment"
            }
            _ => "color_fragment",
        }
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
            Self::IndexedPositionFloat32x3 => 12,
            Self::IndexedPositionFloat32x3CameraMaterial => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTexture => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => 12,
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => 12,
        }
    }

    /// Returns the required index element format for this closed recipe.
    pub const fn index_format(self) -> Option<IndexFormat> {
        match self {
            Self::Triangle => None,
            Self::IndexedPositionColor => Some(IndexFormat::Uint16),
            Self::IndexedPositionFloat32x3 => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterial => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTexture => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => Some(IndexFormat::Uint32),
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                Some(IndexFormat::Uint32)
            }
        }
    }

    /// Returns the complete non-configurable vertex layout.
    pub const fn vertex_layout(self) -> RasterVertexLayout {
        match self {
            Self::Triangle => RasterVertexLayout::None,
            Self::IndexedPositionColor => RasterVertexLayout::PositionFloat32x2ColorUnorm8x4,
            Self::IndexedPositionFloat32x3 => RasterVertexLayout::PositionFloat32x3,
            Self::IndexedPositionFloat32x3CameraMaterial => RasterVertexLayout::PositionFloat32x3,
            Self::IndexedPositionFloat32x3CameraMaterialTexture => {
                RasterVertexLayout::PositionFloat32x3
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
            }
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
            index_format: self.index_format(),
            binding_count: if self.has_frame_uniform() {
                if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                ) {
                    3
                } else if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTexture
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                ) {
                    2
                } else {
                    1
                }
            } else {
                0
            },
            uniform_binding_size: if self.has_frame_uniform() { 80 } else { 0 },
            uniform_binding: if self.has_frame_uniform() {
                Some(0)
            } else {
                None
            },
            uniform_visibility: if self.has_frame_uniform() {
                Some(RasterBindingVisibility::VertexFragment)
            } else {
                None
            },
            binding_recipe_version: if self.has_frame_uniform() { 1 } else { 0 },
            texture_binding_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                1
            } else {
                0
            },
            texture_binding: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(1)
            } else {
                None
            },
            texture_dimension: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(TextureDimension::D2)
            } else {
                None
            },
            texture_format: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(TextureFormat::Rgba8Unorm)
            } else {
                None
            },
            texture_visibility: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterBindingVisibility::Fragment)
            } else {
                None
            },
            texture_sample_type: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterTextureSampleType::Float)
            } else {
                None
            },
            texture_mip_level: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
            ) {
                Some(0)
            } else {
                None
            },
            texture_mapping_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTexture
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                if matches!(
                    self,
                    Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                        | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                ) {
                    2
                } else {
                    1
                }
            } else {
                0
            },
            texture_coordinate_vertex_layout: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2)
            } else {
                None
            },
            texture_coordinate_vertex_stride: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(8)
            } else {
                None
            },
            texture_coordinate_shader_location: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(1)
            } else {
                None
            },
            sampler_binding: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(2)
            } else {
                None
            },
            sampler_binding_recipe_version: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                1
            } else {
                0
            },
            sampler_binding_type: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerBindingType::Filtering)
            } else {
                None
            },
            sampler_min_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerFilter::Linear)
            } else {
                None
            },
            sampler_mag_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerFilter::Linear)
            } else {
                None
            },
            sampler_mipmap_filter: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerFilter::Nearest)
            } else {
                None
            },
            sampler_address_mode_u: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_address_mode_v: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_address_mode_w: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(RasterSamplerAddressMode::ClampToEdge)
            } else {
                None
            },
            sampler_explicit_mip_level: if matches!(
                self,
                Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            ) {
                Some(0)
            } else {
                None
            },
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
            Self::IndexedPositionFloat32x3 => {
                "struct VertexInput {\n\
         @location(0) position: vec3<f32>,\n\
         };\n\
         struct VertexOutput {\n\
         @builtin(position) position: vec4<f32>,\n\
         };\n\
         @vertex fn position_f32x3_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(vec4(input.position, 1.0));\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> {\n\
             _ = input;\n\
             return vec4(48.0 / 255.0, 176.0 / 255.0, 112.0 / 255.0, 1.0);\n\
         }"
            }
            Self::IndexedPositionFloat32x3CameraMaterial => {
                "struct FrameUniforms {\n\
         view_projection: mat4x4<f32>,\n\
         base_color: vec4<f32>,\n\
         };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         struct VertexInput { @location(0) position: vec3<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, };\n\
         @vertex fn camera_material_vertex(input: VertexInput) -> VertexOutput {\n\
             return VertexOutput(frame.view_projection * vec4(input.position, 1.0));\n\
         }\n\
         @fragment fn color_fragment(input: VertexOutput) -> @location(0) vec4<f32> {\n\
             _ = input; return frame.base_color;\n\
         }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTexture => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         struct VertexInput { @location(0) position: vec3<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.position.xy * vec2(0.5, -0.5) + vec2(0.5)); }\n\
         @fragment fn texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { let dimensions = textureDimensions(tex, 0); let texel = min(vec2<u32>(floor(clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dimensions))), dimensions - vec2<u32>(1)); return frame.base_color * textureLoad(tex, vec2<i32>(texel), 0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUv => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_uv_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.uv); }\n\
         @fragment fn texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { let dimensions = textureDimensions(tex, 0); let texel = min(vec2<u32>(floor(clamp(input.uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dimensions))), dimensions - vec2<u32>(1)); return frame.base_color * textureLoad(tex, vec2<i32>(texel), 0); }"
            }
            Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                "struct FrameUniforms { view_projection: mat4x4<f32>, base_color: vec4<f32>, };\n\
         @group(0) @binding(0) var<uniform> frame: FrameUniforms;\n\
         @group(0) @binding(1) var tex: texture_2d<f32>;\n\
         @group(0) @binding(2) var linear_clamp_sampler: sampler;\n\
         struct VertexInput { @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, };\n\
         struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) @interpolate(perspective, center) uv: vec2<f32>, };\n\
         @vertex fn camera_material_texture_uv_linear_clamp_vertex(input: VertexInput) -> VertexOutput { return VertexOutput(frame.view_projection * vec4(input.position, 1.0), input.uv); }\n\
         @fragment fn linear_clamp_texture_color_fragment(input: VertexOutput) -> @location(0) vec4<f32> { return frame.base_color * textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0); }"
            }
        }
    }

    const fn has_frame_uniform(self) -> bool {
        matches!(
            self,
            Self::IndexedPositionFloat32x3CameraMaterial
                | Self::IndexedPositionFloat32x3CameraMaterialTexture
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUv
                | Self::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        )
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

struct RasterUniformBindingsShared {
    _native: crate::imp::NativeRasterUniformBindings,
    pipeline: RasterPipeline,
    _buffer: BufferLease,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed group-0/binding-0 80-byte frame uniform binding for the camera and
/// material raster artifact only.
#[derive(Clone)]
pub struct RasterUniformBindings(Arc<RasterUniformBindingsShared>);

/// A cloneable strong lease retaining the uniform bind group, pipeline, and buffer.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "the lease is retained through ResourceLease for terminal native completion"
)]
pub struct RasterUniformBindingsLease(Arc<RasterUniformBindingsShared>);

impl fmt::Debug for RasterUniformBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUniformBindingsLease(..)")
    }
}

struct RasterTextureBindingsShared {
    _native: crate::imp::NativeRasterTextureBindings,
    pipeline: RasterPipeline,
    _uniform: BufferLease,
    _texture: TextureLease,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed group-0 bindings: one 80-byte frame uniform and one whole sampled RGBA8 texture.
#[derive(Clone)]
pub struct RasterTextureBindings(Arc<RasterTextureBindingsShared>);

/// Strong lease retaining the textured raster binding and both resources.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterTextureBindingsLease(Arc<RasterTextureBindingsShared>);

impl fmt::Debug for RasterTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterTextureBindingsLease(..)")
    }
}

struct RasterUvTextureBindingsShared {
    _native: crate::imp::NativeRasterTextureBindings,
    pipeline: RasterPipeline,
    _uniform: BufferLease,
    _texture: TextureLease,
    _positions: BufferLease,
    _texture_coordinates: BufferLease,
    position_identity: PhysicalResourceIdentity,
    texture_coordinate_identity: PhysicalResourceIdentity,
    vertex_count: u32,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed explicit-UV raster binding. It retains every resource and records
/// the two physical vertex-role identities that native recording must match.
#[derive(Clone)]
pub struct RasterUvTextureBindings(Arc<RasterUvTextureBindingsShared>);

/// Strong lease retaining the explicit-UV binding and all its resources.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterUvTextureBindingsLease(Arc<RasterUvTextureBindingsShared>);

impl fmt::Debug for RasterUvTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUvTextureBindingsLease(..)")
    }
}

struct RasterUvLinearClampTextureBindingsShared {
    _native: crate::imp::NativeRasterTextureBindings,
    pipeline: RasterPipeline,
    _uniform: BufferLease,
    _texture: TextureLease,
    _positions: BufferLease,
    _texture_coordinates: BufferLease,
    position_identity: PhysicalResourceIdentity,
    texture_coordinate_identity: PhysicalResourceIdentity,
    vertex_count: u32,
    device: fluxel_rendergraph::DeviceIdentity,
}

/// Closed explicit-UV binding with an internally owned filtering linear-clamp
/// sampler. The sampler is not a graph resource or a configurable public API.
#[derive(Clone)]
pub struct RasterUvLinearClampTextureBindings(Arc<RasterUvLinearClampTextureBindingsShared>);

/// Strong lease retaining the closed linear-clamp binding, its private sampler,
/// pipeline, and every referenced resource through terminal completion.
#[derive(Clone)]
#[allow(
    dead_code,
    reason = "retained through ResourceLease for terminal native completion"
)]
pub struct RasterUvLinearClampTextureBindingsLease(Arc<RasterUvLinearClampTextureBindingsShared>);

impl fmt::Debug for RasterUvLinearClampTextureBindingsLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RasterUvLinearClampTextureBindingsLease(..)")
    }
}

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
    /// Retains the closed camera/material raster uniform binding.
    RasterUniformBindings(RasterUniformBindingsLease),
    /// Retains the closed textured raster binding.
    RasterTextureBindings(RasterTextureBindingsLease),
    /// Retains the closed explicit-UV textured raster binding.
    RasterUvTextureBindings(RasterUvTextureBindingsLease),
    /// Retains the closed explicit-UV linear-clamp sampler binding.
    RasterUvLinearClampTextureBindings(RasterUvLinearClampTextureBindingsLease),
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

impl From<RasterUniformBindingsLease> for ResourceLease {
    fn from(value: RasterUniformBindingsLease) -> Self {
        Self::RasterUniformBindings(value)
    }
}
impl From<RasterTextureBindingsLease> for ResourceLease {
    fn from(value: RasterTextureBindingsLease) -> Self {
        Self::RasterTextureBindings(value)
    }
}
impl From<RasterUvTextureBindingsLease> for ResourceLease {
    fn from(value: RasterUvTextureBindingsLease) -> Self {
        Self::RasterUvTextureBindings(value)
    }
}
impl From<RasterUvLinearClampTextureBindingsLease> for ResourceLease {
    fn from(value: RasterUvLinearClampTextureBindingsLease) -> Self {
        Self::RasterUvLinearClampTextureBindings(value)
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

impl PendingBufferUpload {
    /// Returns the completion state without blocking the calling thread.
    pub fn status(&self) -> Result<CompletionStatus, BufferUploadError> {
        crate::imp::completion_status(&self.completion.0).map_err(|reason| {
            BufferUploadError::Native {
                backend: self.backend,
                stage: BufferUploadStage::Completion,
                reason,
            }
        })
    }

    /// Waits no longer than `timeout` for the accepted upload submission.
    ///
    /// A timeout returns [`CompletionStatus::Pending`]; it does not cancel the
    /// upload or release its staging allocation.
    pub fn wait(
        &self,
        timeout: core::time::Duration,
    ) -> Result<CompletionStatus, BufferUploadError> {
        crate::imp::wait_completion(&self.completion.0, timeout).map_err(|reason| {
            BufferUploadError::Native {
                backend: self.backend,
                stage: BufferUploadStage::Completion,
                reason,
            }
        })
    }

    /// Finalizes this upload only after native completion is proven.
    ///
    /// Pending and failed submissions are returned intact so their completion
    /// and keepalive storage remain owned by the caller.
    pub fn finalize(mut self) -> Result<UploadedBuffer, IncompleteBufferUpload> {
        let status = crate::imp::completion_status(&self.completion.0).unwrap_or(
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost),
        );
        if status == CompletionStatus::Complete {
            Ok(UploadedBuffer {
                buffer: self
                    .buffer
                    .take()
                    .expect("pending upload retains its destination until completion"),
            })
        } else {
            Err(IncompleteBufferUpload {
                upload: self,
                status,
            })
        }
    }
}

impl Drop for PendingBufferUpload {
    fn drop(&mut self) {
        // Dropping an application-level pending operation must neither block
        // the caller nor release storage still referenced by accepted native
        // work. A leaked Arc clone keeps the submission bundle, target lease,
        // and private staging allocation quarantined whenever completion is
        // not proven. Complete bundles take the ordinary immediate cleanup
        // path when this value's original completion field is dropped.
        if !matches!(
            crate::imp::completion_status(&self.completion.0),
            Ok(CompletionStatus::Complete)
        ) {
            let _quarantined = std::mem::ManuallyDrop::new(self.completion.clone());
        }
    }
}

impl IncompleteBufferUpload {
    /// Returns the observed non-complete state.
    #[must_use]
    pub fn status(&self) -> CompletionStatus {
        self.status
    }

    /// Returns the retained upload so it can be polled or waited again.
    #[must_use]
    pub fn upload(&self) -> &PendingBufferUpload {
        &self.upload
    }

    /// Returns ownership of the retained upload.
    #[must_use]
    pub fn into_pending(self) -> PendingBufferUpload {
        self.upload
    }
}

impl UploadedBuffer {
    /// Returns the immutable destination buffer.
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns the state that later graph imports must use as their incoming state.
    #[must_use]
    pub const fn outgoing_state(&self) -> ResourceAccessState {
        ResourceAccessState::CopyDestination
    }

    /// Returns an independent lifetime lease for later graph submission.
    #[must_use]
    pub fn lease(&self) -> BufferLease {
        self.buffer.lease()
    }
}

impl PendingTextureUpload {
    /// Returns the accepted upload's current completion state without blocking.
    pub fn status(&self) -> Result<CompletionStatus, TextureUploadError> {
        crate::imp::completion_status(&self.completion.0).map_err(|reason| {
            TextureUploadError::Native {
                backend: self.backend,
                stage: TextureUploadStage::Completion,
                reason,
            }
        })
    }

    /// Waits at most `timeout` for this accepted upload to complete.
    pub fn wait(
        &self,
        timeout: core::time::Duration,
    ) -> Result<CompletionStatus, TextureUploadError> {
        crate::imp::wait_completion(&self.completion.0, timeout).map_err(|reason| {
            TextureUploadError::Native {
                backend: self.backend,
                stage: TextureUploadStage::Completion,
                reason,
            }
        })
    }

    /// Converts a proven-complete upload into its graph-importable texture.
    pub fn finalize(mut self) -> Result<UploadedTexture, IncompleteTextureUpload> {
        let status = crate::imp::completion_status(&self.completion.0).unwrap_or(
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost),
        );
        if status == CompletionStatus::Complete {
            Ok(UploadedTexture {
                texture: self
                    .texture
                    .take()
                    .expect("pending upload retains its destination until completion"),
            })
        } else {
            Err(IncompleteTextureUpload {
                upload: self,
                status,
            })
        }
    }
}

impl Drop for PendingTextureUpload {
    fn drop(&mut self) {
        if !matches!(
            crate::imp::completion_status(&self.completion.0),
            Ok(CompletionStatus::Complete)
        ) {
            let _quarantined = std::mem::ManuallyDrop::new(self.completion.clone());
        }
    }
}

impl IncompleteTextureUpload {
    /// Returns the observed non-complete state.
    #[must_use]
    pub fn status(&self) -> CompletionStatus {
        self.status
    }
    /// Returns the retained operation for a later poll or wait.
    #[must_use]
    pub fn upload(&self) -> &PendingTextureUpload {
        &self.upload
    }
    /// Returns ownership of the retained upload.
    #[must_use]
    pub fn into_pending(self) -> PendingTextureUpload {
        self.upload
    }
}

impl UploadedTexture {
    /// Returns the immutable destination texture.
    #[must_use]
    pub fn texture(&self) -> &Texture {
        &self.texture
    }
    /// Returns the state that graph imports must use as their incoming state.
    #[must_use]
    pub const fn outgoing_state(&self) -> ResourceAccessState {
        ResourceAccessState::CopyDestination
    }
    /// Acquires an independent lifetime lease for later graph submission.
    #[must_use]
    pub fn lease(&self) -> TextureLease {
        self.texture.lease()
    }
}

impl fmt::Debug for PendingTextureUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingTextureUpload")
            .field("texture", &self.texture)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for IncompleteTextureUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncompleteTextureUpload")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}
impl fmt::Debug for UploadedTexture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadedTexture")
            .field("texture", &self.texture)
            .field("outgoing_state", &self.outgoing_state())
            .finish()
    }
}

impl fmt::Debug for PendingBufferUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingBufferUpload")
            .field("buffer", &self.buffer)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for IncompleteBufferUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncompleteBufferUpload")
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for UploadedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UploadedBuffer")
            .field("buffer", &self.buffer)
            .field("outgoing_state", &self.outgoing_state())
            .finish()
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
    pub(crate) fn same_object(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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

impl RasterUniformBindings {
    /// Returns the sole camera/material pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a lease retaining the binding, pipeline, and uniform buffer.
    pub fn lease(&self) -> RasterUniformBindingsLease {
        RasterUniformBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterUniformBindings {
        &self.0._native
    }
}

impl RasterTextureBindings {
    /// Returns the textured raster pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the owning device identity.
    pub fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
    /// Acquires a strong lifetime lease.
    pub fn lease(&self) -> RasterTextureBindingsLease {
        RasterTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
}

impl RasterUvTextureBindings {
    /// Returns the sole explicit-UV pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the physical buffer generation required at vertex slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the physical buffer generation required at vertex slot one.
    pub fn texture_coordinate_identity(&self) -> PhysicalResourceIdentity {
        self.0.texture_coordinate_identity
    }
    /// Returns the exact number of vertices represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Acquires a terminal-lifetime lease for the complete binding.
    pub fn lease(&self) -> RasterUvTextureBindingsLease {
        RasterUvTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
    }
}

impl RasterUvLinearClampTextureBindings {
    /// Returns the sole explicit-UV linear-clamp pipeline accepted by this binding.
    pub fn pipeline(&self) -> &RasterPipeline {
        &self.0.pipeline
    }
    /// Returns the physical buffer generation required at vertex slot zero.
    pub fn position_identity(&self) -> PhysicalResourceIdentity {
        self.0.position_identity
    }
    /// Returns the physical buffer generation required at vertex slot one.
    pub fn texture_coordinate_identity(&self) -> PhysicalResourceIdentity {
        self.0.texture_coordinate_identity
    }
    /// Returns the exact number of vertices represented by both streams.
    pub fn vertex_count(&self) -> u32 {
        self.0.vertex_count
    }
    /// Acquires a terminal-lifetime lease for the complete binding and its
    /// private native sampler.
    pub fn lease(&self) -> RasterUvLinearClampTextureBindingsLease {
        RasterUvLinearClampTextureBindingsLease(Arc::clone(&self.0))
    }
    pub(crate) fn native(&self) -> &crate::imp::NativeRasterTextureBindings {
        &self.0._native
    }
    pub(crate) fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.0.device
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
    /// Creates a device-local RGBA8 texture and starts one immutable whole-image upload.
    ///
    /// Input bytes are tightly packed RGBA8 rows. The private native boundary
    /// supplies required row padding; callers never provide a native pitch.
    pub fn upload_immutable_texture(
        &self,
        descriptor: TextureDescriptor,
        bytes: &[u8],
    ) -> Result<PendingTextureUpload, TextureUploadError> {
        validate_immutable_texture_upload_descriptor(descriptor, bytes)?;
        let texture = self
            .create_texture(descriptor)
            .map_err(TextureUploadError::Resource)?;
        let completion = crate::imp::upload_immutable_texture(
            &self.inner,
            texture.native(),
            texture.lease().into(),
            descriptor.texture,
            bytes,
        )
        .map_err(|(stage, reason)| TextureUploadError::Native {
            backend: self.hardware.backend,
            stage,
            reason,
        })?;
        Ok(PendingTextureUpload {
            texture: Some(texture),
            completion: NativeCompletion(completion),
            backend: self.hardware.backend,
        })
    }

    /// Creates a device-local buffer and starts one immutable copy upload.
    ///
    /// The descriptor size must exactly match `bytes`, the byte length must be
    /// non-zero and four-byte aligned, and the declared usage must include
    /// copy-destination access. The returned value does not make the buffer
    /// graph-importable until [`PendingBufferUpload::finalize`] reports a
    /// completed submission.
    pub fn upload_immutable_buffer(
        &self,
        descriptor: BufferDescriptor,
        bytes: &[u8],
    ) -> Result<PendingBufferUpload, BufferUploadError> {
        validate_immutable_upload_descriptor(descriptor, bytes)?;
        let buffer = self
            .create_buffer(descriptor)
            .map_err(BufferUploadError::Resource)?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::CopyDestination)
        {
            return Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::CopyDestinationUsageRequired,
            ));
        }
        let completion = crate::imp::upload_immutable_buffer(
            &self.inner,
            buffer.native(),
            buffer.lease().into(),
            bytes,
        )
        .map_err(|(stage, reason)| BufferUploadError::Native {
            backend: self.hardware.backend,
            stage,
            reason,
        })?;
        Ok(PendingBufferUpload {
            buffer: Some(buffer),
            completion: NativeCompletion(completion),
            backend: self.hardware.backend,
        })
    }

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
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        let native = crate::imp::create_compute_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.entry_point(),
        )
        .map_err(map_compute_pipeline_create_error)?;
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
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
        if kernel == RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            && !self.capabilities().rgba8_unorm_filterable
        {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            });
        }
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel,
        )
        .map_err(map_raster_pipeline_create_error)?;
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        let native = crate::imp::create_raster_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.vertex_entry_point(),
            kernel.fragment_entry_point(),
            kernel,
        )
        .map_err(RasterCreateError::NativeObjectCreation)?;
        Ok(RasterPipeline(Arc::new(RasterPipelineShared {
            _native: native,
            kernel,
            device: self.identity,
        })))
    }

    /// Creates the only group-0/binding-0 uniform binding accepted by the
    /// camera/material raster artifact.
    pub fn create_raster_uniform_bindings(
        &self,
        pipeline: &RasterPipeline,
        buffer: &Buffer,
    ) -> Result<RasterUniformBindings, RasterCreateError> {
        validate_raster_uniform_contract(
            pipeline.kernel(),
            buffer.descriptor().buffer.size,
            buffer.allowed_usage(),
        )?;
        if pipeline.device_identity() != self.identity || buffer.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        let native = crate::imp::create_raster_uniform_bindings(
            &self.inner,
            pipeline.native(),
            buffer.native(),
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUniformBindings(Arc::new(
            RasterUniformBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _buffer: buffer.lease(),
                device: self.identity,
            },
        )))
    }

    /// Creates the only uniform-plus-whole-texture binding accepted by the textured raster artifact.
    pub fn create_raster_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
    ) -> Result<RasterTextureBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let native = crate::imp::create_raster_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterTextureBindings(Arc::new(
            RasterTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                device: self.identity,
            },
        )))
    }

    /// Creates the closed explicit-UV textured raster binding. Both vertex
    /// streams are whole, tightly packed, same-device vertex buffers and are
    /// retained with their physical identities for later role validation.
    pub fn create_raster_uv_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterUvTextureBindings, RasterCreateError> {
        if pipeline.kernel() != RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || texture_coordinates.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let uv_size = u64::from(vertex_count)
            .checked_mul(8)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if texture_coordinates.descriptor().buffer.size != uv_size {
            return Err(RasterCreateError::InvalidTextureCoordinateStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !texture_coordinates
                .allowed_usage()
                .contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_uv_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
            positions.identity(),
            position_size,
            texture_coordinates.identity(),
            uv_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUvTextureBindings(Arc::new(
            RasterUvTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                _positions: positions.lease(),
                _texture_coordinates: texture_coordinates.lease(),
                position_identity: positions.identity(),
                texture_coordinate_identity: texture_coordinates.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
    }

    /// Creates the closed explicit-UV linear-clamp sampled-texture binding.
    ///
    /// The sampler is owned privately by the returned opaque binding; callers
    /// cannot configure filter, address, comparison, or LOD behavior.
    #[allow(
        clippy::too_many_arguments,
        reason = "the closed recipe has independent uniform, texture, and two vertex role inputs"
    )]
    pub fn create_raster_uv_linear_clamp_texture_bindings(
        &self,
        pipeline: &RasterPipeline,
        uniform: &Buffer,
        texture: &Texture,
        positions: &Buffer,
        texture_coordinates: &Buffer,
        vertex_count: u32,
    ) -> Result<RasterUvLinearClampTextureBindings, RasterCreateError> {
        if !self.capabilities().rgba8_unorm_filterable {
            return Err(RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            });
        }
        if pipeline.kernel()
            != RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        {
            return Err(RasterCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || uniform.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || positions.device_identity() != self.identity
            || texture_coordinates.device_identity() != self.identity
        {
            return Err(RasterCreateError::ForeignDevice);
        }
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            uniform.descriptor().buffer.size,
            uniform.allowed_usage(),
        )?;
        validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
            .map_err(|_| RasterCreateError::BindingRecipeMismatch)?;
        let position_size = u64::from(vertex_count)
            .checked_mul(12)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        let uv_size = u64::from(vertex_count)
            .checked_mul(8)
            .ok_or(RasterCreateError::VertexStreamCountMismatch)?;
        if vertex_count == 0 || positions.descriptor().buffer.size != position_size {
            return Err(RasterCreateError::InvalidPositionStreamRange);
        }
        if texture_coordinates.descriptor().buffer.size != uv_size {
            return Err(RasterCreateError::InvalidTextureCoordinateStreamRange);
        }
        if !positions.allowed_usage().contains(BufferUsageKind::Vertex)
            || !texture_coordinates
                .allowed_usage()
                .contains(BufferUsageKind::Vertex)
        {
            return Err(RasterCreateError::VertexUsageRequired);
        }
        let native = crate::imp::create_raster_uv_linear_clamp_texture_bindings(
            &self.inner,
            pipeline.native(),
            uniform.native(),
            texture.native(),
            positions.identity(),
            position_size,
            texture_coordinates.identity(),
            uv_size,
        )
        .map_err(RasterCreateError::NativeFailure)?;
        Ok(RasterUvLinearClampTextureBindings(Arc::new(
            RasterUvLinearClampTextureBindingsShared {
                _native: native,
                pipeline: pipeline.clone(),
                _uniform: uniform.lease(),
                _texture: texture.lease(),
                _positions: positions.lease(),
                _texture_coordinates: texture_coordinates.lease(),
                position_identity: positions.identity(),
                texture_coordinate_identity: texture_coordinates.identity(),
                vertex_count,
                device: self.identity,
            },
        )))
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

fn validate_raster_uniform_contract(
    kernel: RasterKernel,
    size: u64,
    usage: BufferUsage,
) -> Result<(), RasterCreateError> {
    if kernel != RasterKernel::IndexedPositionFloat32x3CameraMaterial {
        return Err(RasterCreateError::BindingRecipeMismatch);
    }
    if size != 80 {
        return Err(RasterCreateError::InvalidBindingRange);
    }
    if !usage.contains(BufferUsageKind::Uniform) {
        return Err(RasterCreateError::UniformUsageRequired);
    }
    Ok(())
}

fn validate_immutable_upload_descriptor(
    descriptor: BufferDescriptor,
    bytes: &[u8],
) -> Result<(), BufferUploadError> {
    if bytes.is_empty() {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::EmptyData,
        ));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DataLengthNotCopyAligned,
        ));
    }
    if descriptor.buffer.size != bytes.len() as u64 {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DescriptorSizeMismatch,
        ));
    }
    if descriptor.memory != MemoryPolicy::DeviceOnly {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::MemoryPolicyUnsupported,
        ));
    }
    if !descriptor.usage.contains(BufferUsageKind::CopyDestination) {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::CopyDestinationUsageRequired,
        ));
    }
    Ok(())
}

fn validate_immutable_texture_upload_descriptor(
    descriptor: TextureDescriptor,
    bytes: &[u8],
) -> Result<(), TextureUploadError> {
    let image = descriptor.texture;
    let invalid = |reason| Err(TextureUploadError::InvalidRequest(reason));
    if image.dimension != TextureDimension::D2 {
        return invalid(InvalidTextureUploadReason::DimensionUnsupported);
    }
    if image.format != TextureFormat::Rgba8Unorm {
        return invalid(InvalidTextureUploadReason::FormatUnsupported);
    }
    if image.mip_levels != 1 {
        return invalid(InvalidTextureUploadReason::MipLevelsUnsupported);
    }
    if image.array_layers != 1 {
        return invalid(InvalidTextureUploadReason::ArrayLayersUnsupported);
    }
    if image.sample_count != 1 {
        return invalid(InvalidTextureUploadReason::SampleCountUnsupported);
    }
    if descriptor.memory != MemoryPolicy::DeviceOnly {
        return invalid(InvalidTextureUploadReason::MemoryPolicyUnsupported);
    }
    if !descriptor.usage.contains(TextureUsageKind::CopyDestination) {
        return invalid(InvalidTextureUploadReason::CopyDestinationUsageRequired);
    }
    if !descriptor.usage.contains(TextureUsageKind::Sampled) {
        return invalid(InvalidTextureUploadReason::SampledUsageRequired);
    }
    let production_usage =
        TextureUsage::from_kinds([TextureUsageKind::CopyDestination, TextureUsageKind::Sampled]);
    // The production object deliberately has no readback or general-purpose
    // usage widening.  Test-support may add exactly CopySource so its private
    // oracle can observe the upload; it may never grant storage or attachment
    // access through this creation path.
    #[cfg(any(test, feature = "test-support"))]
    let usage_is_closed = descriptor.usage == production_usage
        || descriptor.usage == production_usage.with(TextureUsageKind::CopySource);
    #[cfg(not(any(test, feature = "test-support")))]
    let usage_is_closed = descriptor.usage == production_usage;
    if !usage_is_closed {
        return invalid(InvalidTextureUploadReason::UnexpectedUsage);
    }
    let expected = u64::from(image.extent.width)
        .checked_mul(u64::from(image.extent.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::DataLengthOverflow,
        ))?;
    if u64::try_from(bytes.len()).ok() != Some(expected) {
        return invalid(InvalidTextureUploadReason::DataLengthMismatch);
    }
    Ok(())
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

    #[test]
    fn immutable_upload_requires_exact_device_only_copy_destination_bytes() {
        let descriptor = BufferDescriptor {
            buffer: BufferDesc { size: 8 },
            usage: BufferUsage::empty().with(BufferUsageKind::CopyDestination),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_immutable_upload_descriptor(descriptor, &[]),
            Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::EmptyData
            ))
        );
        assert_eq!(
            validate_immutable_upload_descriptor(descriptor, &[0; 6]),
            Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::DataLengthNotCopyAligned
            ))
        );
        assert_eq!(
            validate_immutable_upload_descriptor(descriptor, &[0; 4]),
            Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::DescriptorSizeMismatch
            ))
        );
        assert_eq!(
            validate_immutable_upload_descriptor(
                BufferDescriptor {
                    usage: BufferUsage::empty(),
                    ..descriptor
                },
                &[0; 8]
            ),
            Err(BufferUploadError::InvalidRequest(
                InvalidBufferUploadReason::CopyDestinationUsageRequired
            ))
        );
        assert_eq!(
            validate_immutable_upload_descriptor(descriptor, &[0; 8]),
            Ok(())
        );
    }

    #[cfg(all(windows, feature = "dx12"))]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn immutable_upload_dx12_failure_contract() {
        run_immutable_upload_failure_contract(crate::Backend::Dx12);
    }

    #[cfg(all(windows, feature = "vulkan"))]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn immutable_upload_vulkan_failure_contract() {
        run_immutable_upload_failure_contract(crate::Backend::Vulkan);
    }

    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    fn run_immutable_upload_failure_contract(backend: crate::Backend) {
        use core::time::Duration;

        let device = Device::open(
            backend,
            crate::DeviceOptions {
                validation: crate::Validation::Required,
                ..crate::DeviceOptions::default()
            },
        )
        .unwrap();
        crate::imp::clear_validation_diagnostics(&device.inner);
        let descriptor = BufferDescriptor {
            buffer: BufferDesc { size: 8 },
            usage: BufferUsage::from_kinds([BufferUsageKind::CopyDestination]),
            memory: MemoryPolicy::DeviceOnly,
        };

        crate::imp::inject_submit_rejected_once();
        assert!(matches!(
            device.upload_immutable_buffer(descriptor, &[0; 8]),
            Err(BufferUploadError::Native {
                stage: BufferUploadStage::SubmitRejected,
                ..
            })
        ));

        crate::imp::inject_submit_accepted_unknown_once();
        let accepted_unknown = device.upload_immutable_buffer(descriptor, &[1; 8]).unwrap();
        assert_eq!(
            accepted_unknown.status().unwrap(),
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
        );
        let incomplete = accepted_unknown.finalize().unwrap_err();
        assert_eq!(
            incomplete.status(),
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
        );
        drop(incomplete);

        let pending = device.upload_immutable_buffer(descriptor, &[2; 8]).unwrap();
        crate::imp::inject_completion_pending_once();
        assert_eq!(pending.status().unwrap(), CompletionStatus::Pending);
        assert_eq!(
            pending.wait(Duration::from_secs(10)).unwrap(),
            CompletionStatus::Complete
        );
        assert!(pending.finalize().is_ok());
        let diagnostics = crate::imp::validation_diagnostics(&device.inner);
        assert!(diagnostics.is_empty(), "{backend:?}: {diagnostics:?}");
        eprintln!(
            "artifact case=U01-failure backend={backend:?} submit_rejected=true accepted_unknown=DeviceLost observation=Pending completion=Complete diagnostics={diagnostics:?}"
        );
    }

    #[cfg(all(windows, feature = "dx12"))]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn immutable_texture_upload_dx12_failure_contract() {
        run_immutable_texture_upload_failure_contract(crate::Backend::Dx12);
    }

    #[cfg(all(windows, feature = "vulkan"))]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn immutable_texture_upload_vulkan_failure_contract() {
        run_immutable_texture_upload_failure_contract(crate::Backend::Vulkan);
    }

    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    fn run_immutable_texture_upload_failure_contract(backend: crate::Backend) {
        use core::time::Duration;

        let device = Device::open(
            backend,
            crate::DeviceOptions {
                validation: crate::Validation::Required,
                ..crate::DeviceOptions::default()
            },
        )
        .unwrap();
        crate::imp::clear_validation_diagnostics(&device.inner);
        let descriptor = TextureDescriptor {
            texture: TextureDesc {
                dimension: TextureDimension::D2,
                extent: Extent3d {
                    width: 3,
                    height: 2,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                sample_count: 1,
                format: TextureFormat::Rgba8Unorm,
            },
            usage: TextureUsage::from_kinds([
                TextureUsageKind::CopyDestination,
                TextureUsageKind::Sampled,
                TextureUsageKind::CopySource,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        };

        crate::imp::inject_submit_rejected_once();
        assert!(matches!(
            device.upload_immutable_texture(descriptor, &[0; 24]),
            Err(TextureUploadError::Native {
                stage: TextureUploadStage::SubmitRejected,
                ..
            })
        ));

        crate::imp::inject_submit_accepted_unknown_once();
        let unknown = device
            .upload_immutable_texture(descriptor, &[1; 24])
            .unwrap();
        assert_eq!(
            unknown.status().unwrap(),
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
        );
        let incomplete = unknown.finalize().unwrap_err();
        assert_eq!(
            incomplete.status(),
            CompletionStatus::Failed(fluxel_rendergraph::CompletionFailure::DeviceLost)
        );
        // An unproven accepted submission never publishes a texture. Dropping
        // this retained operation takes the same quarantine path as a caller
        // that abandons it before observing completion.
        drop(incomplete);

        let pending = device
            .upload_immutable_texture(descriptor, &[2; 24])
            .unwrap();
        crate::imp::inject_completion_pending_once();
        assert_eq!(pending.status().unwrap(), CompletionStatus::Pending);
        assert_eq!(
            pending.wait(Duration::from_secs(10)).unwrap(),
            CompletionStatus::Complete
        );
        let uploaded = pending.finalize().expect("complete upload is published");
        assert_eq!(
            uploaded.outgoing_state(),
            ResourceAccessState::CopyDestination
        );
        let diagnostics = crate::imp::validation_diagnostics(&device.inner);
        assert!(diagnostics.is_empty(), "{backend:?}: {diagnostics:?}");
        eprintln!(
            "artifact case=D2-failure backend={backend:?} submit_rejected=true accepted_unknown=DeviceLost observation=Pending completion=Complete diagnostics={diagnostics:?}"
        );
    }

    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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

    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
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
        let indexed_f32x3 = RasterKernel::IndexedPositionFloat32x3.portable_identity();
        assert_eq!(triangle.target_format, TextureFormat::Rgba8Unorm);
        assert_eq!(triangle.vertex_stride, 0);
        assert_eq!(triangle.vertex_layout, RasterVertexLayout::None);
        assert_eq!(triangle.index_format, None);
        assert_eq!(indexed.vertex_stride, 12);
        assert_eq!(
            indexed.vertex_layout,
            RasterVertexLayout::PositionFloat32x2ColorUnorm8x4
        );
        assert_ne!(triangle.module_source_hash, indexed.module_source_hash);
        assert_eq!(triangle.fragment_entry_point, indexed.fragment_entry_point);
        assert_eq!(indexed_f32x3.vertex_stride, 12);
        assert_eq!(indexed_f32x3.index_format, Some(IndexFormat::Uint32));
        assert_eq!(
            indexed_f32x3.vertex_layout,
            RasterVertexLayout::PositionFloat32x3
        );
        assert_ne!(indexed.module_source_hash, indexed_f32x3.module_source_hash);
        // Pre-0.2.2 artifacts retain the no-binding identity facts exactly.
        assert_eq!(triangle.binding_count, 0);
        assert_eq!(triangle.uniform_binding_size, 0);
        assert_eq!(triangle.binding_recipe_version, 0);
        assert_eq!(indexed_f32x3.binding_count, 0);
        assert_eq!(
            format!("{indexed_f32x3:?}"),
            "RasterArtifactIdentity { module_source_hash: 16318329126220124131, vertex_entry_point: \"position_f32x3_vertex\", fragment_entry_point: \"color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), recipe_version: 1 }"
        );
        let camera = RasterKernel::IndexedPositionFloat32x3CameraMaterial.portable_identity();
        assert_eq!(camera.binding_count, 1);
        assert_eq!(camera.uniform_binding_size, 80);
        assert_eq!(camera.binding_recipe_version, 1);
        assert_eq!(
            format!("{camera:?}"),
            "RasterArtifactIdentity { module_source_hash: 8563197035213614468, vertex_entry_point: \"camera_material_vertex\", fragment_entry_point: \"color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), binding_count: 1, uniform_binding_size: 80, binding_recipe_version: 1, recipe_version: 1 }"
        );
        assert_ne!(camera.module_source_hash, indexed_f32x3.module_source_hash);
    }

    #[test]
    fn camera_material_uniform_contract_is_closed_and_fail_closed() {
        let uniform =
            BufferUsage::from_kinds([BufferUsageKind::Uniform, BufferUsageKind::CopyDestination]);
        assert_eq!(
            validate_raster_uniform_contract(
                RasterKernel::IndexedPositionFloat32x3CameraMaterial,
                80,
                uniform,
            ),
            Ok(())
        );
        assert_eq!(
            validate_raster_uniform_contract(RasterKernel::IndexedPositionFloat32x3, 80, uniform),
            Err(RasterCreateError::BindingRecipeMismatch)
        );
        assert_eq!(
            validate_raster_uniform_contract(
                RasterKernel::IndexedPositionFloat32x3CameraMaterial,
                64,
                uniform,
            ),
            Err(RasterCreateError::InvalidBindingRange)
        );
        assert_eq!(
            validate_raster_uniform_contract(
                RasterKernel::IndexedPositionFloat32x3CameraMaterial,
                80,
                BufferUsage::from_kinds([BufferUsageKind::CopyDestination]),
            ),
            Err(RasterCreateError::UniformUsageRequired)
        );
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

    #[test]
    fn immutable_texture_upload_is_closed_and_requires_exact_tight_rgba8() {
        let image = TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 3,
                height: 2,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        };
        let descriptor = TextureDescriptor {
            texture: image,
            usage: TextureUsage::from_kinds([
                TextureUsageKind::CopyDestination,
                TextureUsageKind::Sampled,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        };
        assert_eq!(
            validate_immutable_texture_upload_descriptor(descriptor, &[0; 24]),
            Ok(())
        );
        assert_eq!(
            validate_immutable_texture_upload_descriptor(descriptor, &[0; 23]),
            Err(TextureUploadError::InvalidRequest(
                InvalidTextureUploadReason::DataLengthMismatch
            ))
        );
        assert_eq!(
            validate_immutable_texture_upload_descriptor(
                TextureDescriptor {
                    usage: TextureUsage::empty().with(TextureUsageKind::Sampled),
                    ..descriptor
                },
                &[0; 24]
            ),
            Err(TextureUploadError::InvalidRequest(
                InvalidTextureUploadReason::CopyDestinationUsageRequired
            ))
        );
        assert_eq!(
            validate_immutable_texture_upload_descriptor(
                TextureDescriptor {
                    usage: descriptor.usage.with(TextureUsageKind::StorageWrite),
                    ..descriptor
                },
                &[0; 24]
            ),
            Err(TextureUploadError::InvalidRequest(
                InvalidTextureUploadReason::UnexpectedUsage
            ))
        );
        assert_eq!(
            validate_immutable_texture_upload_descriptor(
                TextureDescriptor {
                    usage: descriptor.usage.with(TextureUsageKind::ColorAttachment),
                    ..descriptor
                },
                &[0; 24]
            ),
            Err(TextureUploadError::InvalidRequest(
                InvalidTextureUploadReason::UnexpectedUsage
            ))
        );
        // Readback is a test-only widening, not a production upload feature.
        assert_eq!(
            validate_immutable_texture_upload_descriptor(
                TextureDescriptor {
                    usage: descriptor.usage.with(TextureUsageKind::CopySource),
                    ..descriptor
                },
                &[0; 24]
            ),
            Ok(())
        );
        assert_eq!(
            validate_immutable_texture_upload_descriptor(
                TextureDescriptor {
                    texture: TextureDesc {
                        mip_levels: 2,
                        ..image
                    },
                    ..descriptor
                },
                &[0; 24]
            ),
            Err(TextureUploadError::InvalidRequest(
                InvalidTextureUploadReason::MipLevelsUnsupported
            ))
        );
    }

    #[test]
    fn textured_raster_identity_is_explicit_without_mutating_legacy_debug() {
        let legacy = format!(
            "{:?}",
            RasterKernel::IndexedPositionFloat32x3.portable_identity()
        );
        assert!(!legacy.contains("texture_binding"));
        let identity =
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture.portable_identity();
        assert_eq!(identity.binding_count, 2);
        assert_eq!(identity.uniform_binding_size, 80);
        assert_eq!(identity.uniform_binding, Some(0));
        assert_eq!(
            identity.uniform_visibility,
            Some(RasterBindingVisibility::VertexFragment)
        );
        assert_eq!(identity.texture_binding, Some(1));
        assert_eq!(identity.texture_dimension, Some(TextureDimension::D2));
        assert_eq!(identity.texture_format, Some(TextureFormat::Rgba8Unorm));
        assert_eq!(
            identity.texture_visibility,
            Some(RasterBindingVisibility::Fragment)
        );
        assert_eq!(
            identity.texture_sample_type,
            Some(RasterTextureSampleType::Float)
        );
        assert_eq!(identity.texture_mip_level, Some(0));
        assert_eq!(identity.texture_mapping_recipe_version, 1);
        assert_eq!(
            format!("{identity:?}"),
            "RasterArtifactIdentity { module_source_hash: 8313758755579089945, vertex_entry_point: \"camera_material_texture_vertex\", fragment_entry_point: \"texture_color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), binding_count: 2, uniform_binding_size: 80, binding_recipe_version: 1, texture_binding_recipe_version: 1, texture_binding: Some(1), texture_dimension: Some(D2), texture_format: Some(Rgba8Unorm), uniform_binding: Some(0), uniform_visibility: Some(VertexFragment), texture_visibility: Some(Fragment), texture_sample_type: Some(Float), texture_mip_level: Some(0), texture_mapping_recipe_version: 1, recipe_version: 1 }"
        );
        let uv = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv.portable_identity();
        assert_eq!(uv.texture_mapping_recipe_version, 2);
        assert_eq!(
            uv.vertex_layout,
            RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
        );
        assert_eq!(
            uv.texture_coordinate_vertex_layout,
            Some(RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2)
        );
        assert_eq!(uv.texture_coordinate_vertex_stride, Some(8));
        assert_eq!(uv.texture_coordinate_shader_location, Some(1));
        assert_eq!(
            format!("{uv:?}"),
            "RasterArtifactIdentity { module_source_hash: 13201511842499328436, vertex_entry_point: \"camera_material_texture_uv_vertex\", fragment_entry_point: \"texture_color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3AndTextureCoordinateFloat32x2, vertex_stride: 12, index_format: Some(Uint32), binding_count: 2, uniform_binding_size: 80, binding_recipe_version: 1, texture_binding_recipe_version: 1, texture_binding: Some(1), texture_dimension: Some(D2), texture_format: Some(Rgba8Unorm), uniform_binding: Some(0), uniform_visibility: Some(VertexFragment), texture_visibility: Some(Fragment), texture_sample_type: Some(Float), texture_mip_level: Some(0), texture_mapping_recipe_version: 2, texture_coordinate_vertex_layout: Some(PositionFloat32x3AndTextureCoordinateFloat32x2), texture_coordinate_vertex_stride: Some(8), texture_coordinate_shader_location: Some(1), recipe_version: 1 }"
        );
        // Conditional UV fields must not perturb the exact legacy artifact
        // representation consumed by the earlier conformance evidence.
        assert!(!format!("{identity:?}").contains("texture_coordinate_vertex"));
    }

    #[test]
    fn linear_clamp_identity_is_additive_and_records_the_closed_sampler() {
        let old_uv =
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv.portable_identity();
        assert!(!format!("{old_uv:?}").contains("sampler_binding"));

        let identity = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            .portable_identity();
        assert_eq!(identity.binding_count, 3);
        assert_eq!(identity.texture_binding, Some(1));
        assert_eq!(identity.sampler_binding, Some(2));
        assert_eq!(identity.sampler_binding_recipe_version, 1);
        assert_eq!(
            identity.sampler_binding_type,
            Some(RasterSamplerBindingType::Filtering)
        );
        assert_eq!(
            identity.sampler_min_filter,
            Some(RasterSamplerFilter::Linear)
        );
        assert_eq!(
            identity.sampler_mag_filter,
            Some(RasterSamplerFilter::Linear)
        );
        assert_eq!(
            identity.sampler_mipmap_filter,
            Some(RasterSamplerFilter::Nearest)
        );
        assert_eq!(
            identity.sampler_address_mode_u,
            Some(RasterSamplerAddressMode::ClampToEdge)
        );
        assert_eq!(
            identity.sampler_address_mode_v,
            Some(RasterSamplerAddressMode::ClampToEdge)
        );
        assert_eq!(
            identity.sampler_address_mode_w,
            Some(RasterSamplerAddressMode::ClampToEdge)
        );
        assert_eq!(identity.sampler_explicit_mip_level, Some(0));
        assert!(
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                .wgsl_source()
                .contains("textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0)")
        );
        assert!(
            !RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                .wgsl_source()
                .contains("textureSample(tex,")
        );
    }

    #[test]
    fn not_filterable_error_is_structured() {
        assert_eq!(
            RasterCreateError::TextureFormatNotFilterable {
                format: TextureFormat::Rgba8Unorm,
            }
            .to_string(),
            "raster texture format is not filterable: Rgba8Unorm"
        );
    }
}
