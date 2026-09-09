//! Opt-in GPU upload coordination for immutable indexed geometry.

use core::fmt;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus, Extent3d,
    TextureDesc, TextureDimension, TextureFormat, TextureUsage, TextureUsageKind,
};
use fluxel_rhi::{
    BufferDescriptor, BufferUploadError, Device, MemoryPolicy, PendingBufferUpload,
    PendingTextureUpload, TextureDescriptor, TextureUploadError, UploadedBuffer, UploadedTexture,
};

use crate::{BasicMaterial, Geometry};

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// An immutable, GPU-ready indexed-mesh generation.
///
/// This type deliberately exposes no native buffers, raw bytes, or resource
/// states to applications. The renderer retains those facts for later graph
/// lowering. Cloning the snapshot retains its complete native generation.
#[derive(Clone)]
pub struct IndexedMeshSnapshot {
    generation: u64,
    position_count: u32,
    index_count: u32,
    positions: UploadedBuffer,
    indices: UploadedBuffer,
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    position_metadata: Arc<[[f32; 3]]>,
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    index_metadata: Arc<[u32]>,
    use_gate: Arc<SnapshotUseGate>,
}

/// Narrow debug view which keeps renderer-internal GPU resources, CPU copies,
/// and synchronization state outside the public observation surface.
struct IndexedMeshSnapshotDebug {
    generation: u64,
    position_count: u32,
    index_count: u32,
}

impl fmt::Debug for IndexedMeshSnapshotDebug {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedMeshSnapshot")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .finish()
    }
}

impl fmt::Debug for IndexedMeshSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        IndexedMeshSnapshotDebug {
            generation: self.generation,
            position_count: self.position_count,
            index_count: self.index_count,
        }
        .fmt(formatter)
    }
}

impl IndexedMeshSnapshot {
    /// Returns the opaque identity of this immutable generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the number of `f32x3` vertex positions in this generation.
    #[must_use]
    pub const fn position_count(&self) -> u32 {
        self.position_count
    }

    /// Returns the number of `u32` indices in this generation.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Returns the uploaded position resource for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next renderer-lowering slice consumes this private ready resource"
    )]
    pub(crate) fn positions(&self) -> &UploadedBuffer {
        &self.positions
    }

    /// Returns the uploaded index resource for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next renderer-lowering slice consumes this private ready resource"
    )]
    pub(crate) fn indices(&self) -> &UploadedBuffer {
        &self.indices
    }

    /// Returns the original positions for renderer-internal CPU-side lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    pub(crate) fn position_metadata(&self) -> &[[f32; 3]] {
        &self.position_metadata
    }

    /// Returns the original indices for renderer-internal CPU-side lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    pub(crate) fn index_metadata(&self) -> &[u32] {
        &self.index_metadata
    }

    /// Reserves this immutable generation for one renderer submission.
    ///
    /// A clone is another owner of the same native buffers, not another
    /// independently state-tracked generation.  The fixed renderer therefore
    /// serializes use until it has observed a terminal completion.
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

/// Why a ready snapshot cannot start another renderer draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SnapshotUseError {
    /// An accepted draw for this generation has not reached a terminal state.
    InFlight,
    /// A prior accepted draw did not prove that the fixed outgoing states held.
    Poisoned,
}

/// One private reservation for a snapshot generation.
///
/// Dropping an accepted operation without proving completion poisons the gate;
/// a pre-submit caller explicitly releases this reservation instead.
#[derive(Debug)]
pub(crate) struct SnapshotDrawReservation {
    gate: Arc<SnapshotUseGate>,
    active: bool,
}

impl SnapshotDrawReservation {
    /// Releases a reservation whose graph execution was rejected before submit.
    pub(crate) fn release_before_submit(mut self) {
        self.gate.release();
        self.active = false;
    }

    /// Releases a reservation after a complete submission restored known state.
    pub(crate) fn release_complete(&mut self) {
        if self.active {
            self.gate.release();
            self.active = false;
        }
    }

    /// Makes this generation permanently unusable after an uncertain outcome.
    pub(crate) fn poison(&mut self) {
        if self.active {
            self.gate.poison();
            self.active = false;
        }
    }
}

impl Drop for SnapshotDrawReservation {
    fn drop(&mut self) {
        if self.active {
            self.gate.poison();
        }
    }
}

#[derive(Debug)]
struct SnapshotUseGate {
    state: Mutex<SnapshotUseState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotUseState {
    Ready,
    InFlight,
    Poisoned,
}

impl SnapshotUseGate {
    fn new() -> Self {
        Self {
            state: Mutex::new(SnapshotUseState::Ready),
        }
    }

    fn reserve(self: &Arc<Self>) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *state {
            SnapshotUseState::Ready => {
                *state = SnapshotUseState::InFlight;
                Ok(SnapshotDrawReservation {
                    gate: Arc::clone(self),
                    active: true,
                })
            }
            SnapshotUseState::InFlight => Err(SnapshotUseError::InFlight),
            SnapshotUseState::Poisoned => Err(SnapshotUseError::Poisoned),
        }
    }

    fn release(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert_eq!(*state, SnapshotUseState::InFlight);
        *state = SnapshotUseState::Ready;
    }

    fn poison(&self) {
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SnapshotUseState::Poisoned;
    }
}

/// A terminal or observed failure of one indexed-mesh upload generation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadFailure {
    /// The position-buffer submission reached a terminal GPU failure.
    Positions(CompletionFailure),
    /// The index-buffer submission reached a terminal GPU failure.
    Indices(CompletionFailure),
    /// A position submission could no longer be observed safely.
    PositionCompletion(BufferUploadError),
    /// An index submission could no longer be observed safely.
    IndexCompletion(BufferUploadError),
    /// Positions were accepted, but the index submission could not be started.
    ///
    /// The returned upload still owns and retires the already accepted position
    /// submission. This failure is therefore intentionally not a start error.
    IndexStartAfterPositionsAccepted(BufferUploadError),
}

/// Why an indexed-mesh upload could not be started before any GPU work was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadStartError {
    /// Indexed snapshots require at least one vertex position.
    EmptyPositions,
    /// Indexed snapshots require at least one index.
    NonIndexedGeometry,
    /// The vertex count cannot be represented by this slice's public metadata.
    PositionCountTooLarge,
    /// The index count cannot be represented by this slice's public metadata.
    IndexCountTooLarge,
    /// The position upload was rejected before the native queue accepted work.
    PositionUpload(BufferUploadError),
}

impl fmt::Display for IndexedMeshUploadStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => formatter.write_str("indexed mesh positions are empty"),
            Self::NonIndexedGeometry => formatter.write_str("indexed mesh geometry has no indices"),
            Self::PositionCountTooLarge => {
                formatter.write_str("indexed mesh position count exceeds u32")
            }
            Self::IndexCountTooLarge => formatter.write_str("indexed mesh index count exceeds u32"),
            Self::PositionUpload(error) => {
                write!(formatter, "position upload did not start: {error}")
            }
        }
    }
}

impl std::error::Error for IndexedMeshUploadStartError {}

/// The non-blocking observable state of an [`IndexedMeshUpload`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadStatus {
    /// At least one accepted upload is still incomplete.
    Pending,
    /// Both accepted uploads completed and an immutable snapshot is available.
    Ready,
    /// This generation cannot become ready.
    ///
    /// The upload object may still retain a sibling accepted submission until
    /// that submission becomes terminal. Calling [`IndexedMeshUpload::poll`]
    /// remains useful and never turns this status back into `Ready`.
    Failed(IndexedMeshUploadFailure),
}

/// One opt-in, non-blocking indexed-mesh upload operation.
///
/// The operation is intentionally not cloneable: it is the unique coordinator
/// that advances both accepted submissions. It never waits for the CPU; call
/// [`Self::poll`] from an application's normal progress loop.
pub struct IndexedMeshUpload {
    generation: u64,
    position_count: u32,
    index_count: u32,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    positions: UploadSlot,
    indices: UploadSlot,
    failure: Option<IndexedMeshUploadFailure>,
    snapshot: Option<IndexedMeshSnapshot>,
}

enum UploadSlot {
    Absent,
    Pending(PendingBufferUpload),
    Ready(UploadedBuffer),
}

impl IndexedMeshUpload {
    /// Starts the two immutable uploads required for `geometry`.
    ///
    /// Position and index payloads are serialized exactly as tightly packed
    /// little-endian IEEE-754 `f32x3` and little-endian `u32` values. If the
    /// position request fails, no queue work has been accepted and this returns
    /// an error. Once positions have been accepted, every outcome returns an
    /// owning operation, including an index-start failure, so accepted storage
    /// remains alive until it is safe to retire.
    pub fn begin(
        device: &Device,
        geometry: &Geometry,
    ) -> Result<Self, IndexedMeshUploadStartError> {
        let payload = IndexedMeshPayload::from_geometry(geometry)?;
        let position_metadata: Arc<[[f32; 3]]> = Arc::from(geometry.positions());
        let index_metadata: Arc<[u32]> = Arc::from(geometry.indices());
        let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let positions = device
            .upload_immutable_buffer(
                buffer_descriptor(payload.positions.len(), BufferUsageKind::Vertex),
                &payload.positions,
            )
            .map_err(IndexedMeshUploadStartError::PositionUpload)?;

        match device.upload_immutable_buffer(
            buffer_descriptor(payload.indices.len(), BufferUsageKind::Index),
            &payload.indices,
        ) {
            Ok(indices) => Ok(Self {
                generation,
                position_count: payload.position_count,
                index_count: payload.index_count,
                position_metadata,
                index_metadata,
                positions: UploadSlot::Pending(positions),
                indices: UploadSlot::Pending(indices),
                failure: None,
                snapshot: None,
            }),
            Err(error) => Ok(Self {
                generation,
                position_count: payload.position_count,
                index_count: payload.index_count,
                position_metadata,
                index_metadata,
                positions: UploadSlot::Pending(positions),
                indices: UploadSlot::Absent,
                failure: Some(IndexedMeshUploadFailure::IndexStartAfterPositionsAccepted(
                    error,
                )),
                snapshot: None,
            }),
        }
    }

    /// Polls both accepted uploads without blocking the calling thread.
    ///
    /// A terminal failure is monotonic. Even after reporting it, this method
    /// continues polling any accepted sibling so its keepalive storage can be
    /// retained through a terminal completion.
    pub fn poll(&mut self) -> IndexedMeshUploadStatus {
        poll_slot(&mut self.positions, true, &mut self.failure);
        poll_slot(&mut self.indices, false, &mut self.failure);

        if let Some(failure) = &self.failure {
            return IndexedMeshUploadStatus::Failed(failure.clone());
        }
        let publication = SnapshotPublication::new(
            upload_slot_state(&self.positions),
            upload_slot_state(&self.indices),
            self.failure.is_some(),
        );
        if self.snapshot.is_none() && publication.can_publish() {
            let positions = take_ready(&mut self.positions);
            let indices = take_ready(&mut self.indices);
            self.snapshot = Some(IndexedMeshSnapshot {
                generation: self.generation,
                position_count: self.position_count,
                index_count: self.index_count,
                positions,
                indices,
                position_metadata: Arc::clone(&self.position_metadata),
                index_metadata: Arc::clone(&self.index_metadata),
                use_gate: Arc::new(SnapshotUseGate::new()),
            });
        }
        if self.snapshot.is_some() {
            IndexedMeshUploadStatus::Ready
        } else {
            IndexedMeshUploadStatus::Pending
        }
    }

    /// Returns a strong, immutable ready snapshot after [`Self::poll`] reports `Ready`.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<IndexedMeshSnapshot> {
        self.snapshot.clone()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetainedUploadState {
    Missing,
    Pending,
    Ready,
}

/// Pure publication policy kept separate from the RHI completion adapter.
///
/// The small state-only coordinator makes the atomic renderer contract testable
/// without a native device: a partial pair never publishes, and an observed
/// failure remains terminal even while another accepted upload is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SnapshotPublication {
    positions: RetainedUploadState,
    indices: RetainedUploadState,
    failed: bool,
}

impl SnapshotPublication {
    const fn new(
        positions: RetainedUploadState,
        indices: RetainedUploadState,
        failed: bool,
    ) -> Self {
        Self {
            positions,
            indices,
            failed,
        }
    }

    const fn can_publish(self) -> bool {
        !self.failed
            && matches!(self.positions, RetainedUploadState::Ready)
            && matches!(self.indices, RetainedUploadState::Ready)
    }
}

impl fmt::Debug for IndexedMeshUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedMeshUpload")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

fn buffer_descriptor(byte_len: usize, terminal_usage: BufferUsageKind) -> BufferDescriptor {
    let usage = BufferUsage::from_kinds([BufferUsageKind::CopyDestination, terminal_usage]);
    #[cfg(test)]
    let usage = usage.with(BufferUsageKind::CopySource);
    BufferDescriptor {
        buffer: BufferDesc {
            size: u64::try_from(byte_len).expect("validated renderer payload length fits u64"),
        },
        // The conformance-only CopySource addition permits the doc-hidden
        // readback oracle. Production snapshots authorize only their upload
        // and future vertex/index consumption states.
        usage,
        memory: MemoryPolicy::DeviceOnly,
    }
}

fn poll_slot(
    slot: &mut UploadSlot,
    positions: bool,
    failure: &mut Option<IndexedMeshUploadFailure>,
) {
    let UploadSlot::Pending(upload) = slot else {
        return;
    };
    match upload.status() {
        Ok(CompletionStatus::Pending) => {}
        Ok(CompletionStatus::Complete) => {
            let UploadSlot::Pending(upload) = core::mem::replace(slot, UploadSlot::Absent) else {
                unreachable!("slot was pending while completing it");
            };
            match upload.finalize() {
                Ok(buffer) => *slot = UploadSlot::Ready(buffer),
                Err(incomplete) => {
                    let observed = incomplete.status();
                    *slot = UploadSlot::Pending(incomplete.into_pending());
                    record_status_failure(observed, positions, failure);
                }
            }
        }
        Ok(status) => record_status_failure(status, positions, failure),
        Err(error) => {
            failure.get_or_insert_with(|| {
                if positions {
                    IndexedMeshUploadFailure::PositionCompletion(error)
                } else {
                    IndexedMeshUploadFailure::IndexCompletion(error)
                }
            });
        }
    }
}

fn record_status_failure(
    status: CompletionStatus,
    positions: bool,
    failure: &mut Option<IndexedMeshUploadFailure>,
) {
    if let CompletionStatus::Failed(reason) = status {
        failure.get_or_insert(if positions {
            IndexedMeshUploadFailure::Positions(reason)
        } else {
            IndexedMeshUploadFailure::Indices(reason)
        });
    }
}

fn take_ready(slot: &mut UploadSlot) -> UploadedBuffer {
    match core::mem::replace(slot, UploadSlot::Absent) {
        UploadSlot::Ready(buffer) => buffer,
        _ => unreachable!("snapshot construction requires ready uploads"),
    }
}

fn upload_slot_state(slot: &UploadSlot) -> RetainedUploadState {
    match slot {
        UploadSlot::Absent => RetainedUploadState::Missing,
        UploadSlot::Pending(_) => RetainedUploadState::Pending,
        UploadSlot::Ready(_) => RetainedUploadState::Ready,
    }
}

#[derive(Debug)]
struct IndexedMeshPayload {
    positions: Vec<u8>,
    indices: Vec<u8>,
    position_count: u32,
    index_count: u32,
}

impl IndexedMeshPayload {
    fn from_geometry(geometry: &Geometry) -> Result<Self, IndexedMeshUploadStartError> {
        if geometry.positions().is_empty() {
            return Err(IndexedMeshUploadStartError::EmptyPositions);
        }
        if !geometry.is_indexed() {
            return Err(IndexedMeshUploadStartError::NonIndexedGeometry);
        }
        let position_count = u32::try_from(geometry.positions().len())
            .map_err(|_| IndexedMeshUploadStartError::PositionCountTooLarge)?;
        let index_count = u32::try_from(geometry.indices().len())
            .map_err(|_| IndexedMeshUploadStartError::IndexCountTooLarge)?;
        let position_bytes = geometry
            .positions()
            .len()
            .checked_mul(12)
            .ok_or(IndexedMeshUploadStartError::PositionCountTooLarge)?;
        let index_bytes = geometry
            .indices()
            .len()
            .checked_mul(4)
            .ok_or(IndexedMeshUploadStartError::IndexCountTooLarge)?;
        let mut positions = Vec::with_capacity(position_bytes);
        for position in geometry.positions() {
            for component in position {
                positions.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        let mut indices = Vec::with_capacity(index_bytes);
        for index in geometry.indices() {
            indices.extend_from_slice(&index.to_le_bytes());
        }
        Ok(Self {
            positions,
            indices,
            position_count,
            index_count,
        })
    }
}

/// A tightly packed, immutable RGBA8 image for the fixed base-color path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rgba8Image {
    extent: [u32; 2],
    pixels: Vec<u8>,
}

impl Rgba8Image {
    /// Creates a full-image, tightly packed RGBA8 payload.
    pub fn new(extent: [u32; 2], pixels: Vec<u8>) -> Result<Self, Rgba8ImageError> {
        let expected_len = rgba8_byte_len(extent)?;
        if pixels.len() != expected_len {
            return Err(Rgba8ImageError::IncorrectByteLength {
                expected: expected_len,
                actual: pixels.len(),
            });
        }
        Ok(Self { extent, pixels })
    }

    /// Returns the two-dimensional texel extent.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the tightly packed RGBA8 texels in row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Why an [`Rgba8Image`] could not be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Rgba8ImageError {
    /// The image width is zero.
    ZeroWidth,
    /// The image height is zero.
    ZeroHeight,
    /// The requested tightly packed byte length cannot be represented locally.
    ByteLengthOverflow,
    /// The supplied payload is not exactly one full tightly packed image.
    IncorrectByteLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
}

impl fmt::Display for Rgba8ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("RGBA8 image width is zero"),
            Self::ZeroHeight => formatter.write_str("RGBA8 image height is zero"),
            Self::ByteLengthOverflow => formatter.write_str("RGBA8 image byte length overflows"),
            Self::IncorrectByteLength { expected, actual } => {
                write!(
                    formatter,
                    "RGBA8 image has {actual} bytes; expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Rgba8ImageError {}

fn rgba8_byte_len(extent: [u32; 2]) -> Result<usize, Rgba8ImageError> {
    let [width, height] = extent;
    if width == 0 {
        return Err(Rgba8ImageError::ZeroWidth);
    }
    if height == 0 {
        return Err(Rgba8ImageError::ZeroHeight);
    }
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Rgba8ImageError::ByteLengthOverflow)?;
    usize::try_from(bytes).map_err(|_| Rgba8ImageError::ByteLengthOverflow)
}

/// A completed immutable base-color texture generation.
///
/// The native texture, its outgoing state, and its lease remain opaque. A
/// clone retains the same generation and shares one use gate.
#[derive(Clone)]
pub struct BaseColorTextureSnapshot {
    generation: u64,
    extent: [u32; 2],
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    texture: UploadedTexture,
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    use_gate: Arc<SnapshotUseGate>,
}

/// Narrow debug view which deliberately excludes renderer-internal native
/// resource, state, descriptor, usage, identity, and synchronization facts.
struct BaseColorTextureSnapshotDebug {
    generation: u64,
    extent: [u32; 2],
}

impl fmt::Debug for BaseColorTextureSnapshotDebug {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BaseColorTextureSnapshot")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .finish()
    }
}

impl fmt::Debug for BaseColorTextureSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        BaseColorTextureSnapshotDebug {
            generation: self.generation,
            extent: self.extent,
        }
        .fmt(formatter)
    }
}

impl BaseColorTextureSnapshot {
    /// Returns the opaque identity of this immutable generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the uploaded texture extent in texels.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the uploaded texture for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    pub(crate) fn texture(&self) -> &UploadedTexture {
        &self.texture
    }

    /// Reserves this texture generation for one renderer submission.
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

/// Why a base-color texture upload could not start before queue acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadStartError {
    /// The native immutable texture request was rejected before acceptance.
    Upload(TextureUploadError),
}

impl fmt::Display for BaseColorTextureUploadStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upload(error) => write!(
                formatter,
                "base-color texture upload did not start: {error}"
            ),
        }
    }
}

impl std::error::Error for BaseColorTextureUploadStartError {}

/// A terminal or observed failure of an accepted base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadFailure {
    /// The texture submission reached a terminal GPU failure.
    Completion(CompletionFailure),
    /// Completion could no longer be observed safely.
    Observation(TextureUploadError),
    /// The backend reported a completion state outside this closed upload contract.
    UnknownCompletionStatus,
}

/// The non-blocking observable state of a base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadStatus {
    /// The accepted submission has not completed.
    Pending,
    /// Completion proved the texture is ready for immutable use.
    Ready,
    /// This accepted generation cannot become ready.
    Failed(BaseColorTextureUploadFailure),
}

/// One owning, non-blocking immutable base-color texture upload.
///
/// Before `begin` returns successfully the source image is merely borrowed and
/// can be retried. After native acceptance this operation retains all state
/// required to observe or safely retire the submission.
pub struct BaseColorTextureUpload {
    generation: u64,
    extent: [u32; 2],
    pending: Option<PendingTextureUpload>,
    failure: Option<BaseColorTextureUploadFailure>,
    snapshot: Option<BaseColorTextureSnapshot>,
}

impl BaseColorTextureUpload {
    /// Starts one immutable RGBA8 texture upload.
    pub fn begin(
        device: &Device,
        image: &Rgba8Image,
    ) -> Result<Self, BaseColorTextureUploadStartError> {
        let pending = device
            .upload_immutable_texture(base_color_texture_descriptor(image.extent), image.pixels())
            .map_err(BaseColorTextureUploadStartError::Upload)?;
        Ok(Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            extent: image.extent,
            pending: Some(pending),
            failure: None,
            snapshot: None,
        })
    }

    /// Polls the accepted upload without waiting for the CPU.
    pub fn poll(&mut self) -> BaseColorTextureUploadStatus {
        if let Some(pending) = self.pending.as_ref() {
            match pending.status() {
                Ok(CompletionStatus::Pending) => {}
                Ok(CompletionStatus::Complete) => {
                    let pending = self.pending.take().expect("pending upload was observed");
                    match pending.finalize() {
                        Ok(texture) => {
                            self.snapshot = Some(BaseColorTextureSnapshot {
                                generation: self.generation,
                                extent: self.extent,
                                texture,
                                use_gate: Arc::new(SnapshotUseGate::new()),
                            });
                        }
                        Err(incomplete) => {
                            let status = incomplete.status();
                            self.pending = Some(incomplete.into_pending());
                            self.record_status(status);
                        }
                    }
                }
                Ok(status) => self.record_status(status),
                Err(error) => {
                    self.failure
                        .get_or_insert(BaseColorTextureUploadFailure::Observation(error));
                }
            }
        }
        if let Some(failure) = &self.failure {
            BaseColorTextureUploadStatus::Failed(failure.clone())
        } else if self.snapshot.is_some() {
            BaseColorTextureUploadStatus::Ready
        } else {
            BaseColorTextureUploadStatus::Pending
        }
    }

    fn record_status(&mut self, status: CompletionStatus) {
        match status {
            CompletionStatus::Pending | CompletionStatus::Complete => {}
            CompletionStatus::Failed(failure) => {
                self.failure
                    .get_or_insert(BaseColorTextureUploadFailure::Completion(failure));
            }
            _ => {
                self.failure
                    .get_or_insert(BaseColorTextureUploadFailure::UnknownCompletionStatus);
            }
        }
    }

    /// Returns a strong immutable snapshot after [`Self::poll`] reports ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<BaseColorTextureSnapshot> {
        self.snapshot.clone()
    }
}

impl fmt::Debug for BaseColorTextureUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BaseColorTextureUpload")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

fn base_color_texture_descriptor(extent: [u32; 2]) -> TextureDescriptor {
    let usage =
        TextureUsage::from_kinds([TextureUsageKind::CopyDestination, TextureUsageKind::Sampled]);
    // This widening exists only in renderer unit/conformance builds, where the
    // RHI's doc-hidden readback oracle must observe the completed upload.  The
    // production descriptor remains exactly CopyDestination + Sampled.
    #[cfg(test)]
    let usage = usage.with(TextureUsageKind::CopySource);
    TextureDescriptor {
        texture: TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
        usage,
        memory: MemoryPolicy::DeviceOnly,
    }
}

/// A basic material coupled to one immutable base-color texture generation.
#[derive(Clone, Debug)]
pub struct TexturedBasicMaterial {
    material: BasicMaterial,
    base_color_texture: BaseColorTextureSnapshot,
}

impl TexturedBasicMaterial {
    /// Couples a basic material to one ready immutable base-color texture.
    #[must_use]
    pub fn new(material: BasicMaterial, base_color_texture: BaseColorTextureSnapshot) -> Self {
        Self {
            material,
            base_color_texture,
        }
    }

    /// Returns the scalar basic-material properties.
    #[must_use]
    pub const fn material(&self) -> &BasicMaterial {
        &self.material
    }

    /// Returns the immutable base-color texture generation.
    #[must_use]
    pub const fn base_color_texture(&self) -> &BaseColorTextureSnapshot {
        &self.base_color_texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_positions_as_exact_little_endian_bits_and_indices_in_order() {
        let geometry = Geometry::from_positions(vec![[f32::from_bits(0x8000_0000), 1.0, f32::NAN]])
            .with_indices(vec![0])
            .unwrap();
        let payload = IndexedMeshPayload::from_geometry(&geometry).unwrap();
        assert_eq!(
            payload.positions,
            [
                0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0xc0, 0x7f,
            ]
        );
        assert_eq!(payload.indices, [0, 0, 0, 0]);
    }

    #[test]
    fn payload_rejects_empty_or_non_indexed_geometry_before_upload() {
        assert_eq!(
            IndexedMeshPayload::from_geometry(&Geometry::from_positions(Vec::new())).unwrap_err(),
            IndexedMeshUploadStartError::EmptyPositions
        );
        assert_eq!(
            IndexedMeshPayload::from_geometry(&Geometry::from_positions(vec![[0.0; 3]]))
                .unwrap_err(),
            IndexedMeshUploadStartError::NonIndexedGeometry
        );
    }

    #[test]
    fn publication_policy_never_exposes_a_partially_ready_or_failed_pair() {
        let pending = RetainedUploadState::Pending;
        let ready = RetainedUploadState::Ready;
        assert!(!SnapshotPublication::new(pending, pending, false).can_publish());
        assert!(!SnapshotPublication::new(ready, pending, false).can_publish());
        assert!(!SnapshotPublication::new(pending, ready, false).can_publish());
        assert!(SnapshotPublication::new(ready, ready, false).can_publish());

        // A failure is monotonic from the renderer's point of view. The
        // sibling can still be retained and later reach Ready, but that must
        // never repair this generation into a published snapshot.
        assert!(!SnapshotPublication::new(ready, ready, true).can_publish());
    }

    #[test]
    fn publication_policy_represents_partial_acceptance_without_ready_output() {
        assert_eq!(
            SnapshotPublication::new(
                RetainedUploadState::Pending,
                RetainedUploadState::Missing,
                true,
            ),
            SnapshotPublication {
                positions: RetainedUploadState::Pending,
                indices: RetainedUploadState::Missing,
                failed: true,
            }
        );
    }

    #[test]
    fn generation_use_gate_serializes_clones_and_poison_is_monotonic() {
        let gate = Arc::new(SnapshotUseGate::new());
        let reservation = gate.reserve().unwrap();
        assert_eq!(gate.reserve().unwrap_err(), SnapshotUseError::InFlight);
        reservation.release_before_submit();

        let mut accepted = gate.reserve().unwrap();
        accepted.release_complete();
        let poisoned = gate.reserve().unwrap();
        drop(poisoned);
        assert_eq!(gate.reserve().unwrap_err(), SnapshotUseError::Poisoned);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows DX12 device with required validation"]
    fn u01_indexed_mesh_upload_dx12() {
        run_u01(fluxel_rhi::Backend::Dx12);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires a Windows Vulkan device with required validation"]
    fn u01_indexed_mesh_upload_vulkan() {
        run_u01(fluxel_rhi::Backend::Vulkan);
    }

    #[cfg(windows)]
    fn run_u01(backend: fluxel_rhi::Backend) {
        use std::{
            sync::{Mutex, OnceLock},
            thread,
            time::{Duration, Instant},
        };

        use fluxel_rendergraph::ResourceAccessState;
        use fluxel_rhi::{DeviceOptions, Validation, test_support};

        static HARDWARE_SERIAL: OnceLock<Mutex<()>> = OnceLock::new();
        let guard = HARDWARE_SERIAL
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("hardware fixture mutex must not be poisoned");
        let device = Device::open(
            backend,
            DeviceOptions {
                validation: Validation::Required,
                ..DeviceOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("U01 {backend:?} device open failed: {error}"));
        test_support::clear_validation_diagnostics(&device);

        let geometry = Geometry::from_positions(vec![
            [-1.0, -0.5, 0.0],
            [0.0, 1.0, -0.0],
            [0.75, -0.25, f32::from_bits(0x7fc0_0011)],
        ])
        .with_indices(vec![2, 1, 0, 2, 0, 1])
        .unwrap();
        let expected = IndexedMeshPayload::from_geometry(&geometry).unwrap();
        let mut upload = IndexedMeshUpload::begin(&device, &geometry)
            .unwrap_or_else(|error| panic!("U01 {backend:?} upload start failed: {error}"));
        let deadline = Instant::now() + Duration::from_secs(10);
        let snapshot = loop {
            match upload.poll() {
                IndexedMeshUploadStatus::Ready => {
                    break upload
                        .ready_snapshot()
                        .expect("Ready upload must publish its complete snapshot");
                }
                IndexedMeshUploadStatus::Failed(error) => {
                    panic!("U01 {backend:?} upload failed: {error:?}")
                }
                IndexedMeshUploadStatus::Pending if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                IndexedMeshUploadStatus::Pending => panic!("U01 {backend:?} timed out"),
            }
        };
        let actual_positions =
            test_support::readback_uploaded_buffer(&device, snapshot.positions()).unwrap_or_else(
                |error| panic!("U01 {backend:?} position readback failed: {error}"),
            );
        let actual_indices = test_support::readback_uploaded_buffer(&device, snapshot.indices())
            .unwrap_or_else(|error| panic!("U01 {backend:?} index readback failed: {error}"));
        assert_eq!(
            actual_positions, expected.positions,
            "U01 {backend:?} positions"
        );
        assert_eq!(actual_indices, expected.indices, "U01 {backend:?} indices");
        assert_eq!(
            snapshot.positions().outgoing_state(),
            ResourceAccessState::CopyDestination
        );
        assert_eq!(
            snapshot.indices().outgoing_state(),
            ResourceAccessState::CopyDestination
        );
        let diagnostics = test_support::validation_diagnostics(&device);
        assert!(diagnostics.is_empty(), "U01 {backend:?}: {diagnostics:?}");

        let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
        eprintln!(
            "artifact case=U01 backend={backend:?} commit={commit} os={}; hardware={:?}; input=indexed f32x3 positions={} u32 indices={}; position_descriptor={:?}; index_descriptor={:?}; expected_positions={:?}; actual_positions={actual_positions:?}; expected_indices={:?}; actual_indices={actual_indices:?}; first_difference_positions={:?}; first_difference_indices={:?}; outgoing_position={:?}; outgoing_index={:?}; completion=Complete; diagnostics={diagnostics:?}",
            std::env::consts::OS,
            device.hardware(),
            snapshot.position_count(),
            snapshot.index_count(),
            snapshot.positions().buffer().descriptor(),
            snapshot.indices().buffer().descriptor(),
            expected.positions,
            expected.indices,
            first_difference(&expected.positions, &actual_positions),
            first_difference(&expected.indices, &actual_indices),
            snapshot.positions().outgoing_state(),
            snapshot.indices().outgoing_state(),
        );
        drop(guard);
    }

    #[cfg(windows)]
    fn first_difference(expected: &[u8], actual: &[u8]) -> Option<usize> {
        expected
            .iter()
            .zip(actual)
            .position(|(expected, actual)| expected != actual)
            .or_else(|| {
                (expected.len() != actual.len()).then_some(expected.len().min(actual.len()))
            })
    }

    #[test]
    fn rgba8_image_requires_nonzero_extent_and_exact_tight_length() {
        assert_eq!(
            Rgba8Image::new([0, 1], vec![]).unwrap_err(),
            Rgba8ImageError::ZeroWidth
        );
        assert_eq!(
            Rgba8Image::new([1, 0], vec![]).unwrap_err(),
            Rgba8ImageError::ZeroHeight
        );
        assert_eq!(
            Rgba8Image::new([2, 3], vec![0; 23]).unwrap_err(),
            Rgba8ImageError::IncorrectByteLength {
                expected: 24,
                actual: 23,
            }
        );
        let image = Rgba8Image::new([2, 3], vec![7; 24]).unwrap();
        assert_eq!(image.extent(), [2, 3]);
        assert_eq!(image.pixels(), vec![7; 24]);
    }

    #[test]
    fn rgba8_image_checked_length_handles_largest_constructible_dimensions() {
        assert_eq!(
            rgba8_byte_len([u32::MAX, u32::MAX]),
            Err(Rgba8ImageError::ByteLengthOverflow)
        );
    }

    #[test]
    fn base_color_texture_snapshot_debug_is_exact_safe_metadata_only() {
        let debug = format!(
            "{:?}",
            BaseColorTextureSnapshotDebug {
                generation: 17,
                extent: [3, 2],
            }
        );
        assert_eq!(
            debug,
            "BaseColorTextureSnapshot { generation: 17, extent: [3, 2] }"
        );
        for forbidden in [
            "CopyDestination",
            "UploadedTexture",
            "usage",
            "Ready",
            "descriptor",
            "identity",
            "gate",
            "state",
        ] {
            assert!(
                !debug.contains(forbidden),
                "snapshot debug must not expose {forbidden}: {debug}"
            );
        }
    }

    #[test]
    fn indexed_mesh_snapshot_debug_is_exact_safe_metadata_only() {
        let debug = format!(
            "{:?}",
            IndexedMeshSnapshotDebug {
                generation: 23,
                position_count: 3,
                index_count: 3,
            }
        );
        assert_eq!(
            debug,
            "IndexedMeshSnapshot { generation: 23, position_count: 3, index_count: 3 }"
        );
        for forbidden in [
            "UploadedBuffer",
            "positions:",
            "indices:",
            "metadata",
            "usage",
            "descriptor",
            "identity",
            "gate",
            "state",
        ] {
            assert!(
                !debug.contains(forbidden),
                "snapshot debug must not expose {forbidden}: {debug}"
            );
        }
    }

    #[test]
    fn mesh_snapshot_retains_original_cpu_metadata() {
        let geometry = Geometry::from_positions(vec![[1.0, 2.0, 3.0]])
            .with_indices(vec![0])
            .unwrap();
        let payload = IndexedMeshPayload::from_geometry(&geometry).unwrap();
        assert_eq!(geometry.positions(), &[[1.0, 2.0, 3.0]]);
        assert_eq!(geometry.indices(), &[0]);
        assert_eq!(payload.position_count, 1);
        assert_eq!(payload.index_count, 1);
    }
}
