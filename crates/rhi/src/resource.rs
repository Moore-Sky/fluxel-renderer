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
        }
    }

    /// Returns the fixed workgroup shape baked into this artifact.
    pub const fn workgroup_size(self) -> [u32; 3] {
        [64, 1, 1]
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
        1
    }

    pub(crate) const fn wgsl_source(self) -> &'static str {
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
}
