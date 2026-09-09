//! Backend-independent semantic vocabulary for RenderGraph compilation.
//!
//! This module describes capabilities, resource descriptors, and externally visible
//! resource states. It intentionally contains no device, command recording, submission,
//! synchronization-token, or native-handle API.

#![deny(missing_docs)]

/// An opaque logical queue identifier that is stable for one device lifetime.
///
/// The numeric value is not a native queue-family or hardware-engine index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QueueId(u32);

impl QueueId {
    /// Creates a logical queue identifier for a capability description.
    pub fn new(value: u32) -> Self {
        Self(value)
    }
}

/// The command families that a logical queue can legally execute.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct QueueCapabilities {
    /// Whether the queue can execute raster work.
    pub raster: bool,
    /// Whether the queue can execute compute work.
    pub compute: bool,
    /// Whether the queue can execute copy work.
    pub copy: bool,
    /// Whether the queue can perform presentation work.
    pub present: bool,
}

impl QueueCapabilities {
    /// Creates the baseline command-family capability set.
    pub fn new(raster: bool, compute: bool, copy: bool, present: bool) -> Self {
        Self {
            raster,
            compute,
            copy,
            present,
        }
    }
}

/// A logical queue and the command families it supports.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct QueueDescriptor {
    /// The device-local logical queue identifier.
    pub id: QueueId,
    /// The command families that are legal on this queue.
    pub capabilities: QueueCapabilities,
}

impl QueueDescriptor {
    /// Creates one logical queue descriptor.
    pub fn new(id: QueueId, capabilities: QueueCapabilities) -> Self {
        Self { id, capabilities }
    }
}

/// Recording-related device capabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct RecordingCapabilities {
    /// The backend's command-recording model.
    pub model: RecordingModel,
    /// Whether independent recording jobs can own separate encoders concurrently.
    pub parallel_independent_encoders: bool,
}

impl RecordingCapabilities {
    /// Creates the recording capability facts required by the current graph.
    pub fn new(model: RecordingModel, parallel_independent_encoders: bool) -> Self {
        Self {
            model,
            parallel_independent_encoders,
        }
    }
}

/// The backend's command-recording model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecordingModel {
    /// The backend supports recording command buffers before submission.
    DeferredCommandBuffers,
    /// The backend records through an ordered immediate context.
    ImmediateContext,
}

/// The backend's resource-transition model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TransitionCapabilities {
    /// The backend manages legal resource transitions after graph validation.
    BackendManaged,
    /// The graph may lower semantic transitions to explicit backend operations.
    GraphManagedExplicit,
}

/// The backend's GPU submission-synchronization model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SynchronizationCapabilities {
    /// One logical queue provides ordering without cross-queue GPU dependencies.
    SingleQueueOrdering,
    /// Consumer queues can wait for producer submissions without blocking the CPU.
    MultiQueueGpuDependencies,
}

/// The backend's timestamp-query capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TimestampCapabilities {
    /// Timestamp queries are unavailable.
    Unsupported,
    /// Timestamp queries can be placed at pass or merged-pass boundaries.
    PassBoundaries,
}

/// The backend's transient-resource reuse capabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TransientResourceCapabilities {
    /// Whether compatible whole resource objects can be reused across frames.
    pub cross_frame_object_pooling: bool,
    /// Whether compatible resource objects can be reused during one frame.
    pub in_frame_object_reuse: bool,
    /// Whether multiple resource objects can alias the same memory allocation.
    pub aliased_memory: bool,
}

impl TransientResourceCapabilities {
    /// Creates the transient-resource facts required by the current graph.
    pub fn new(
        cross_frame_object_pooling: bool,
        in_frame_object_reuse: bool,
        aliased_memory: bool,
    ) -> Self {
        Self {
            cross_frame_object_pooling,
            in_frame_object_reuse,
            aliased_memory,
        }
    }
}

/// Immutable capabilities used by the graph compiler to choose legal lowering.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct DeviceCapabilities {
    /// The logical queues available to the device.
    pub queues: Vec<QueueDescriptor>,
    /// The command-recording capabilities.
    pub recording: RecordingCapabilities,
    /// The resource-transition capabilities.
    pub transitions: TransitionCapabilities,
    /// The GPU submission-synchronization capabilities.
    pub synchronization: SynchronizationCapabilities,
    /// The timestamp-query capabilities.
    pub timestamps: TimestampCapabilities,
    /// The transient-resource reuse capabilities.
    pub transient_resources: TransientResourceCapabilities,
    /// Limits used by the current graph examples.
    pub limits: DeviceLimits,
    /// Buffer operation support used by semantic access validation.
    pub buffers: BufferCapabilities,
    /// Per-format semantic support used by graph validation.
    pub texture_formats: Vec<TextureFormatCapabilities>,
    /// Presentation-target operations available to this device/surface pair.
    pub surface: Option<SurfaceCapabilities>,
}

/// A normalized, comparable snapshot of device capabilities.
///
/// This is crate-private because it is an implementation detail of compilation
/// caching rather than a backend capability contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapabilityFingerprint {
    capabilities: DeviceCapabilities,
}

impl DeviceCapabilities {
    /// Starts an extensible, fail-closed capability description.
    pub fn builder() -> DeviceCapabilitiesBuilder {
        DeviceCapabilitiesBuilder::default()
    }

    /// Returns a canonical snapshot suitable for comparing capability inputs.
    pub(crate) fn fingerprint(&self) -> CapabilityFingerprint {
        let mut capabilities = self.clone();

        capabilities.queues.sort_by(|_left, _right| {
            _left
                .id
                .cmp(&_right.id)
                .then_with(|| _left.capabilities.raster.cmp(&_right.capabilities.raster))
                .then_with(|| _left.capabilities.compute.cmp(&_right.capabilities.compute))
                .then_with(|| _left.capabilities.copy.cmp(&_right.capabilities.copy))
                .then_with(|| _left.capabilities.present.cmp(&_right.capabilities.present))
        });
        for _format in &mut capabilities.texture_formats {
            _format.attachment_sample_counts.sort_unstable();
        }
        capabilities
            .texture_formats
            .sort_by(compare_texture_format_capabilities);
        if let Some(_surface) = &mut capabilities.surface {
            _surface
                .formats
                .sort_by_key(|_format| texture_format_rank(*_format));
        }

        CapabilityFingerprint { capabilities }
    }
}

fn compare_texture_format_capabilities(
    _left: &TextureFormatCapabilities,
    _right: &TextureFormatCapabilities,
) -> std::cmp::Ordering {
    texture_format_rank(_left.format)
        .cmp(&texture_format_rank(_right.format))
        .then_with(|| _left.sampled.cmp(&_right.sampled))
        .then_with(|| _left.filterable.cmp(&_right.filterable))
        .then_with(|| _left.storage_read.cmp(&_right.storage_read))
        .then_with(|| _left.storage_write.cmp(&_right.storage_write))
        .then_with(|| _left.color_attachment.cmp(&_right.color_attachment))
        .then_with(|| {
            _left
                .depth_stencil_attachment
                .cmp(&_right.depth_stencil_attachment)
        })
        .then_with(|| {
            _left
                .attachment_sample_counts
                .cmp(&_right.attachment_sample_counts)
        })
        .then_with(|| _left.copy_source.cmp(&_right.copy_source))
        .then_with(|| _left.copy_destination.cmp(&_right.copy_destination))
}

fn texture_format_rank(_format: TextureFormat) -> u8 {
    match _format {
        TextureFormat::Rgba8Unorm => 0,
        TextureFormat::Bgra8Unorm => 1,
        TextureFormat::Rgba16Float => 2,
        TextureFormat::Depth32Float => 3,
    }
}

/// Builder for an extensible device capability description.
pub struct DeviceCapabilitiesBuilder {
    capabilities: DeviceCapabilities,
}

impl Default for DeviceCapabilitiesBuilder {
    fn default() -> Self {
        Self {
            capabilities: DeviceCapabilities {
                queues: Vec::new(),
                recording: RecordingCapabilities::new(RecordingModel::ImmediateContext, false),
                transitions: TransitionCapabilities::BackendManaged,
                synchronization: SynchronizationCapabilities::SingleQueueOrdering,
                timestamps: TimestampCapabilities::Unsupported,
                transient_resources: TransientResourceCapabilities::new(false, false, false),
                limits: DeviceLimits::new(0, 1),
                buffers: BufferCapabilities::new(false, false, false),
                texture_formats: Vec::new(),
                surface: None,
            },
        }
    }
}

impl DeviceCapabilitiesBuilder {
    /// Adds one logical queue.
    pub fn queue(mut self, queue: QueueDescriptor) -> Self {
        self.capabilities.queues.push(queue);
        self
    }

    /// Sets command-recording facts.
    pub fn recording(mut self, recording: RecordingCapabilities) -> Self {
        self.capabilities.recording = recording;
        self
    }

    /// Sets resource-transition facts.
    pub fn transitions(mut self, transitions: TransitionCapabilities) -> Self {
        self.capabilities.transitions = transitions;
        self
    }

    /// Sets GPU submission-synchronization facts.
    pub fn synchronization(mut self, synchronization: SynchronizationCapabilities) -> Self {
        self.capabilities.synchronization = synchronization;
        self
    }

    /// Sets timestamp-query facts.
    pub fn timestamps(mut self, timestamps: TimestampCapabilities) -> Self {
        self.capabilities.timestamps = timestamps;
        self
    }

    /// Sets transient-resource facts.
    pub fn transient_resources(mut self, capabilities: TransientResourceCapabilities) -> Self {
        self.capabilities.transient_resources = capabilities;
        self
    }

    /// Sets the limits currently consumed by graph validation.
    pub fn limits(mut self, limits: DeviceLimits) -> Self {
        self.capabilities.limits = limits;
        self
    }

    /// Sets buffer semantic-operation facts.
    pub fn buffers(mut self, buffers: BufferCapabilities) -> Self {
        self.capabilities.buffers = buffers;
        self
    }

    /// Adds one texture-format capability entry.
    pub fn texture_format(mut self, format: TextureFormatCapabilities) -> Self {
        self.capabilities.texture_formats.push(format);
        self
    }

    /// Sets optional presentation-surface facts.
    pub fn surface(mut self, surface: SurfaceCapabilities) -> Self {
        self.capabilities.surface = Some(surface);
        self
    }

    /// Finishes the capability description.
    pub fn build(self) -> DeviceCapabilities {
        self.capabilities
    }
}

/// Device limits required by the current graph API fixtures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct DeviceLimits {
    /// Maximum number of simultaneous color attachments.
    pub max_color_attachments: u32,
    /// Required alignment of dynamic uniform-buffer offsets.
    pub min_uniform_buffer_offset_alignment: u64,
}

impl DeviceLimits {
    /// Creates the limits currently consumed by graph validation.
    pub fn new(max_color_attachments: u32, min_uniform_buffer_offset_alignment: u64) -> Self {
        Self {
            max_color_attachments,
            min_uniform_buffer_offset_alignment,
        }
    }
}

/// Buffer semantic operations consumed by graph capability validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct BufferCapabilities {
    /// Whether read-only shader storage-buffer access is supported.
    pub storage_read: bool,
    /// Whether shader storage-buffer writes are supported.
    pub storage_write: bool,
    /// Whether indirect-command buffer reads are supported.
    pub indirect_read: bool,
}

impl BufferCapabilities {
    /// Creates the optional buffer-operation capability set.
    pub fn new(storage_read: bool, storage_write: bool, indirect_read: bool) -> Self {
        Self {
            storage_read,
            storage_write,
            indirect_read,
        }
    }
}

/// Semantic operations supported for one texture format.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct TextureFormatCapabilities {
    /// Format described by this entry.
    pub format: TextureFormat,
    /// Whether shaders may sample the format.
    pub sampled: bool,
    /// Whether sampled reads may use filtering.
    pub filterable: bool,
    /// Whether storage reads are supported.
    pub storage_read: bool,
    /// Whether storage writes are supported.
    pub storage_write: bool,
    /// Whether the format may be a color attachment.
    pub color_attachment: bool,
    /// Whether the format may be a depth-stencil attachment.
    pub depth_stencil_attachment: bool,
    /// Legal attachment sample counts for the format.
    pub attachment_sample_counts: Vec<u32>,
    /// Whether the format may be copied from.
    pub copy_source: bool,
    /// Whether the format may be copied to.
    pub copy_destination: bool,
}

impl TextureFormatCapabilities {
    /// Starts an extensible, fail-closed description for one texture format.
    pub fn builder(format: TextureFormat) -> TextureFormatCapabilitiesBuilder {
        TextureFormatCapabilitiesBuilder {
            capabilities: TextureFormatCapabilities {
                format,
                sampled: false,
                filterable: false,
                storage_read: false,
                storage_write: false,
                color_attachment: false,
                depth_stencil_attachment: false,
                attachment_sample_counts: Vec::new(),
                copy_source: false,
                copy_destination: false,
            },
        }
    }
}

/// Builder for one texture-format capability entry.
pub struct TextureFormatCapabilitiesBuilder {
    capabilities: TextureFormatCapabilities,
}

impl TextureFormatCapabilitiesBuilder {
    /// Sets sampled and filtering support.
    pub fn sampled(mut self, sampled: bool, filterable: bool) -> Self {
        self.capabilities.sampled = sampled;
        self.capabilities.filterable = filterable;
        self
    }

    /// Sets storage read and write support.
    pub fn storage(mut self, read: bool, write: bool) -> Self {
        self.capabilities.storage_read = read;
        self.capabilities.storage_write = write;
        self
    }

    /// Sets color/depth attachment support and legal sample counts.
    pub fn attachments(
        mut self,
        color: bool,
        depth_stencil: bool,
        sample_counts: Vec<u32>,
    ) -> Self {
        self.capabilities.color_attachment = color;
        self.capabilities.depth_stencil_attachment = depth_stencil;
        self.capabilities.attachment_sample_counts = sample_counts;
        self
    }

    /// Sets copy source and destination support.
    pub fn copies(mut self, source: bool, destination: bool) -> Self {
        self.capabilities.copy_source = source;
        self.capabilities.copy_destination = destination;
        self
    }

    /// Finishes this format entry.
    pub fn build(self) -> TextureFormatCapabilities {
        self.capabilities
    }
}

/// Presentation operations exposed by one device/surface pair.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SurfaceCapabilities {
    /// Formats that may be acquired for presentation.
    pub formats: Vec<TextureFormat>,
    /// Whether raster passes may write the acquired image as a color attachment.
    pub color_attachment: bool,
    /// Whether legal texture copies may target the acquired image.
    pub copy_destination: bool,
}

impl SurfaceCapabilities {
    /// Creates the surface facts required by the current graph.
    pub fn new(
        formats: Vec<TextureFormat>,
        color_attachment: bool,
        copy_destination: bool,
    ) -> Self {
        Self {
            formats,
            color_attachment,
            copy_destination,
        }
    }
}

/// Three-dimensional texture extent in texels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Extent3d {
    /// The width in texels.
    pub width: u32,
    /// The height in texels.
    pub height: u32,
    /// The depth in texels.
    pub depth: u32,
}

/// The dimensionality of texture storage.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TextureDimension {
    /// One-dimensional texture storage.
    D1,
    /// Two-dimensional texture storage.
    D2,
    /// Three-dimensional texture storage.
    D3,
}

/// A portable texture pixel format.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum TextureFormat {
    /// An unsigned-normalized RGBA format with eight bits per component.
    Rgba8Unorm,
    /// An unsigned-normalized BGRA format with eight bits per component.
    Bgra8Unorm,
    /// A floating-point RGBA format with sixteen bits per component.
    Rgba16Float,
    /// A floating-point depth format with thirty-two bits.
    Depth32Float,
}

/// Format of indices read by an indexed draw.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndexFormat {
    /// Unsigned 16-bit indices.
    Uint16,
    /// Unsigned 32-bit indices.
    Uint32,
}

/// A logical texture description used for graph resource compatibility.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TextureDesc {
    /// The dimensionality of the texture storage.
    pub dimension: TextureDimension,
    /// The texture extent in texels.
    pub extent: Extent3d,
    /// The number of mip levels.
    pub mip_levels: u32,
    /// The number of array layers.
    pub array_layers: u32,
    /// The number of samples per texel.
    pub sample_count: u32,
    /// The pixel format.
    pub format: TextureFormat,
}

/// A logical buffer description used for graph resource compatibility.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BufferDesc {
    /// The buffer size in bytes.
    pub size: u64,
}

/// A backend-independent resource access state at an import or export boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ResourceAccessState {
    /// No prior access state is known or required.
    Undefined,
    /// Read access through a color attachment.
    ColorAttachmentRead,
    /// Write access through a color attachment.
    ColorAttachmentWrite,
    /// Combined read and write access through a color attachment.
    ColorAttachmentReadWrite,
    /// Read access through a depth-stencil attachment.
    DepthStencilRead,
    /// Write access through a depth-stencil attachment.
    DepthStencilWrite,
    /// Combined read and write access through a depth-stencil attachment.
    DepthStencilReadWrite,
    /// Read access through a sampled shader binding.
    ShaderSampledRead,
    /// Read access through a shader storage binding.
    ShaderStorageRead,
    /// Write access through a shader storage binding.
    ShaderStorageWrite,
    /// Combined read and write access through a shader storage binding.
    ShaderStorageReadWrite,
    /// Read access through a uniform binding.
    UniformRead,
    /// Read access through a vertex binding.
    VertexRead,
    /// Read access through an index binding.
    IndexRead,
    /// Read access through an indirect-command binding.
    IndirectRead,
    /// Source access for a copy operation.
    CopySource,
    /// Destination access for a copy operation.
    CopyDestination,
    /// Final state required before presentation.
    Present,
}

/// The external owner responsible for an imported resource's lifetime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ExternalOwnership {
    /// The caller owns the resource and retains responsibility for its lifetime.
    Caller,
    /// The presentation surface owns the acquired resource.
    Surface,
}

#[cfg(test)]
mod tests {
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
}
