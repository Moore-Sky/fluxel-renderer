//! Closed CPU preparation shared by native and browser retained-scene paths.
//!
//! A short-lived [`slot_graph`] expresses the real fan-out from one immutable
//! scene input into per-draw validation and P*V*M/material preparation, followed
//! by insertion-ordered assembly. It owns no GPU work or scheduling policy.

use core::fmt;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use slot_graph::{Graph, InputSpec, Local, NodeError, NodeOutputs, OutputSpec, RunInputs, Schema};

use fluxel_rendergraph::{
    BufferCapabilities, BufferDesc, CompileError, CompiledGraph, DeviceCapabilities, DeviceLimits,
    ExternalOwnership, ImportBufferContract, InitialContents, PresentContract, QueueCapabilities,
    QueueDescriptor, QueueId, RecordingCapabilities, RecordingModel, RenderGraph,
    ResourceAccessState, SurfaceCapabilities, SurfaceTextureContract, SynchronizationCapabilities,
    TextureDesc, TextureDimension, TextureFormat, TextureFormatCapabilities, TimestampCapabilities,
    TransientResourceCapabilities, TransitionCapabilities,
};

use crate::{
    BasicMaterial, Camera, DrawList, Geometry, ModelTransform,
    frame_uniform::{FrameUniform, FrameUniformError},
};

/// A deterministic, fully validated retained basic scene.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedBasicScene {
    draws: Vec<PreparedBasicDraw>,
}

/// One insertion-ordered unlit indexed draw in the closed native/browser ABI.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedBasicDraw {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    pvm_and_color: [f32; 20],
    insertion_index: usize,
    uniform: FrameUniform,
}

/// The two canvas formats accepted by the closed retained presentation recipe.
///
/// This is deliberately not a render-graph format escape hatch: both variants
/// carry the same opaque-alpha, color-attachment-and-present-only contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentableFormat {
    /// Eight-bit linear RGBA, used by the WebGL2 drawing buffer.
    Rgba8Unorm,
    /// Eight-bit linear BGRA, commonly selected by a WebGPU canvas.
    Bgra8Unorm,
}

/// Closed presentation facts for the retained fixed-unlit recipe.
///
/// A profile has no caller-configurable capabilities. It always requires an
/// opaque-alpha presentable image, color-attachment rendering, final present,
/// one raster-capable queue, and the fixed buffer/copy limits used below.
/// Backends report a [`PresentableFormat`] and the binding selects this value;
/// arbitrary [`DeviceCapabilities`] never cross this renderer boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentationProfile {
    format: PresentableFormat,
}

impl PresentationProfile {
    /// Creates the one fixed profile for a backend-reported canvas format.
    #[must_use]
    pub const fn for_format(format: PresentableFormat) -> Self {
        Self { format }
    }

    /// Returns the format selected by the backend's closed canvas-format map.
    #[must_use]
    pub const fn format(self) -> PresentableFormat {
        self.format
    }
}

/// A compiled portable presentation declaration for one prepared basic scene.
pub struct PreparedBasicGraph {
    compiled: CompiledGraph<()>,
    draw_count: usize,
    extent: [u32; 2],
    format: PresentableFormat,
}

/// Why the closed WebGL presentation graph could not be compiled.
#[derive(Debug)]
#[non_exhaustive]
pub enum PreparedBasicGraphError {
    /// Width and height must both be nonzero.
    InvalidExtent,
    /// A CPU stream cannot be represented by the graph's `u64` byte size.
    SizeOverflow,
    /// The WebGL indexed recipe cannot represent this index count.
    IndexCountOverflow,
    /// Portable capability or graph validation rejected the declaration.
    Compile(CompileError),
}

/// Why a scene could not enter the shared closed preparation seam.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PreparedBasicSceneError {
    /// No draw exists to prepare.
    Empty,
    /// The private CPU dependency graph could not be compiled or completed.
    PreparationGraph,
    /// One insertion-ordered draw failed validation.
    Draw {
        /// Stable failing insertion index.
        index: usize,
        /// Closed reason for rejection.
        reason: PreparedBasicDrawError,
    },
}

/// Why one basic draw is not representable by this fixed recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PreparedBasicDrawError {
    /// The closed recipe needs a non-empty triangle-list index stream.
    InvalidIndexCount,
    /// Camera or material values cannot form the fixed uniform ABI.
    InvalidCameraMaterial,
    /// Finite inputs overflowed while forming P*V*M.
    ModelTransformProductNonFinite,
    /// A position contained NaN or infinity.
    NonFinitePosition,
    /// Matrix application produced a non-finite clip coordinate.
    NonFiniteClipPosition,
    /// A clip position had non-positive homogeneous W.
    ClipWNonPositive,
    /// A position fell outside the portable clip volume.
    ClipOutOfBounds,
}

#[derive(Clone)]
struct OwnedScene {
    camera: Camera,
    draws: Vec<OwnedDraw>,
}

#[derive(Clone)]
struct OwnedDraw {
    geometry: Geometry,
    material: BasicMaterial,
    transform: ModelTransform,
}

#[derive(Debug)]
struct PreparationTaskFailure;

impl PreparedBasicScene {
    /// Runs the renderer-owned preparation DAG and owns its validated result.
    pub fn prepare(list: &DrawList<'_>) -> Result<Self, PreparedBasicSceneError> {
        if list.is_empty() {
            return Err(PreparedBasicSceneError::Empty);
        }
        let scene = OwnedScene::from_draw_list(list);
        let errors = Arc::new(Mutex::new(None));
        let mut graph = Graph::<Local>::new();

        let source_schema = Schema::builder()
            .input(InputSpec::required_one::<OwnedScene>("scene"))
            .output(OutputSpec::new::<OwnedScene>("scene"))
            .build()
            .bind();
        let source_input = source_schema
            .input::<OwnedScene>("scene")
            .map_err(|_| graph_error())?;
        let source_output = source_schema
            .output::<OwnedScene>("scene")
            .map_err(|_| graph_error())?;
        let source = graph
            .add_sync("scene-source", source_schema, move |_, inputs| {
                let scene = inputs.required_key(source_input)?;
                let mut outputs = NodeOutputs::new();
                outputs.insert_shared_key(source_output, scene);
                Ok(outputs)
            })
            .map_err(|_| graph_error())?;
        let external_scene = graph
            .expose_input::<OwnedScene>(
                graph
                    .input::<OwnedScene>(source, "scene")
                    .map_err(|_| graph_error())?,
            )
            .map_err(|_| graph_error())?;
        let source_scene = graph
            .output::<OwnedScene>(source, "scene")
            .map_err(|_| graph_error())?;

        let draw_schema = Schema::builder()
            .input(InputSpec::required_one::<OwnedScene>("scene"))
            .output(OutputSpec::new::<PreparedBasicDraw>("draw"))
            .build()
            .bind();
        let draw_scene = draw_schema
            .input::<OwnedScene>("scene")
            .map_err(|_| graph_error())?;
        let draw_output = draw_schema
            .output::<PreparedBasicDraw>("draw")
            .map_err(|_| graph_error())?;
        let mut prepared_nodes = Vec::with_capacity(scene.draws.len());
        for index in 0..scene.draws.len() {
            let errors = Arc::clone(&errors);
            let schema = draw_schema.clone();
            let node = graph
                .add_sync(format!("prepare-draw-{index}"), schema, move |_, inputs| {
                    let scene = inputs.required_key(draw_scene)?;
                    match prepare_one(&scene, index) {
                        Ok(draw) => {
                            let mut outputs = NodeOutputs::new();
                            outputs.insert_key(draw_output, draw);
                            Ok(outputs)
                        }
                        Err(error) => {
                            if let Ok(mut stored) = errors.lock() {
                                record_first_error(&mut stored, error);
                            }
                            Err(NodeError::user(PreparationTaskFailure))
                        }
                    }
                })
                .map_err(|_| graph_error())?;
            graph
                .connect(
                    source_scene,
                    graph
                        .input::<OwnedScene>(node, "scene")
                        .map_err(|_| graph_error())?,
                )
                .map_err(|_| graph_error())?;
            prepared_nodes.push(node);
        }

        let assembly_schema = Schema::builder()
            .input(InputSpec::required_many::<PreparedBasicDraw>("draws"))
            .output(OutputSpec::new::<PreparedBasicScene>("scene"))
            .build()
            .bind();
        let assembly_input = assembly_schema
            .input::<PreparedBasicDraw>("draws")
            .map_err(|_| graph_error())?;
        let assembly_output = assembly_schema
            .output::<PreparedBasicScene>("scene")
            .map_err(|_| graph_error())?;
        let assembly = graph
            .add_sync("assemble-scene", assembly_schema, move |_, inputs| {
                let mut draws = inputs
                    .many_key(assembly_input)?
                    .into_iter()
                    .map(|draw| (*draw).clone())
                    .collect::<Vec<_>>();
                draws.sort_by_key(PreparedBasicDraw::insertion_index);
                let mut outputs = NodeOutputs::new();
                outputs.insert_key(assembly_output, PreparedBasicScene { draws });
                Ok(outputs)
            })
            .map_err(|_| graph_error())?;
        graph
            .collect_into(
                prepared_nodes,
                graph
                    .input::<PreparedBasicDraw>(assembly, "draws")
                    .map_err(|_| graph_error())?,
            )
            .map_err(|_| graph_error())?;
        graph
            .set_active(assembly, true)
            .map_err(|_| graph_error())?;
        let output = graph
            .output::<PreparedBasicScene>(assembly, "scene")
            .map_err(|_| graph_error())?;
        let version = graph.compile().map_err(|_| graph_error())?;
        let mut inputs = RunInputs::new();
        inputs
            .insert(external_scene, scene)
            .map_err(|_| graph_error())?;
        let mut report = match poll_ready(version.execute(inputs)) {
            Ok(Ok(report)) => report,
            Ok(Err(_)) => return Err(take_error(&errors).unwrap_or_else(graph_error)),
            Err(()) => return Err(graph_error()),
        };
        report
            .take_output(output)
            .map(|scene| (*scene).clone())
            .map_err(|_| graph_error())
    }

    /// Returns the stable prepared draw order.
    pub fn draws(&self) -> &[PreparedBasicDraw] {
        &self.draws
    }

    /// Compiles the fixed imported-presentable declaration for the WebGL2
    /// RGBA8 drawing-buffer profile.
    ///
    /// This compatibility entry point keeps the published WebGL caller on its
    /// fixed format. New backend bindings must select their closed profile via
    /// [`Self::compile_presentable_graph_for_profile`].
    pub fn compile_presentable_graph(
        &self,
        extent: [u32; 2],
    ) -> Result<PreparedBasicGraph, PreparedBasicGraphError> {
        self.compile_presentable_graph_for_profile(
            PresentationProfile::for_format(PresentableFormat::Rgba8Unorm),
            extent,
        )
    }

    /// Compiles the fixed imported-presentable declaration for `profile`.
    ///
    /// The selected format changes only the imported presentable descriptor and
    /// its closed compiler facts; pass, draw, usage, clear, store, and present
    /// topology remain identical.
    pub fn compile_presentable_graph_for_profile(
        &self,
        profile: PresentationProfile,
        extent: [u32; 2],
    ) -> Result<PreparedBasicGraph, PreparedBasicGraphError> {
        if extent[0] == 0 || extent[1] == 0 {
            return Err(PreparedBasicGraphError::InvalidExtent);
        }
        let mut graph = RenderGraph::new();
        let mut inputs = Vec::with_capacity(self.draws.len());
        for (index, draw) in self.draws.iter().enumerate() {
            let positions = graph.import_buffer_slot(
                format!("web-draw-{index}-positions"),
                imported_buffer(byte_size(draw.positions.len(), 12)?),
            );
            let indices = graph.import_buffer_slot(
                format!("web-draw-{index}-indices"),
                imported_buffer(byte_size(draw.indices.len(), 4)?),
            );
            let uniform = graph.import_buffer_slot(
                format!("web-draw-{index}-uniform"),
                imported_buffer(crate::frame_uniform::FRAME_UNIFORM_BYTES as u64),
            );
            inputs.push((positions, indices, uniform));
        }
        let descriptor = TextureDesc {
            dimension: TextureDimension::D2,
            extent: fluxel_rendergraph::Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: profile.texture_format(),
        };
        let surface = graph.import_surface_texture_slot(
            "web-presentable-target",
            SurfaceTextureContract { descriptor },
        );
        let draws = inputs
            .iter()
            .zip(&self.draws)
            .map(|((positions, indices, uniform), draw)| {
                Ok(crate::fixed_frame::graph_shared::FixedGraphDraw {
                    positions: &positions.version,
                    indices: &indices.version,
                    uniform: &uniform.version,
                    index_count: u32::try_from(draw.indices.len())
                        .map_err(|_| PreparedBasicGraphError::IndexCountOverflow)?,
                })
            })
            .collect::<Result<Vec<_>, PreparedBasicGraphError>>()?;
        let pass = crate::fixed_frame::graph_shared::declare_fixed_unlit_pass(
            &mut graph,
            surface.version,
            &draws,
            extent,
        );
        graph.present(pass.output, PresentContract::new());
        let compiled = graph
            .compile(&profile.capabilities())
            .map_err(PreparedBasicGraphError::Compile)?
            .graph;
        Ok(PreparedBasicGraph {
            compiled,
            draw_count: self.draws.len(),
            extent,
            format: profile.format(),
        })
    }
}

impl PresentableFormat {
    const fn texture_format(self) -> TextureFormat {
        match self {
            Self::Rgba8Unorm => TextureFormat::Rgba8Unorm,
            Self::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        }
    }
}

impl PresentationProfile {
    fn texture_format(self) -> TextureFormat {
        self.format.texture_format()
    }

    fn capabilities(self) -> DeviceCapabilities {
        let presentable = TextureFormatCapabilities::builder(self.texture_format())
            .sampled(true, true)
            .storage(false, false)
            .attachments(true, false, vec![1])
            .copies(true, true)
            .build();
        DeviceCapabilities::builder()
            .queue(QueueDescriptor::new(
                QueueId::new(0),
                QueueCapabilities::new(true, false, true, true),
            ))
            .recording(RecordingCapabilities::new(
                RecordingModel::ImmediateContext,
                false,
            ))
            .transitions(TransitionCapabilities::BackendManaged)
            .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
            .timestamps(TimestampCapabilities::Unsupported)
            .transient_resources(TransientResourceCapabilities::new(true, false, false))
            .limits(DeviceLimits::new(4, 256))
            .buffers(BufferCapabilities::new(false, false, false))
            .texture_format(presentable)
            .surface(SurfaceCapabilities::new(
                vec![self.texture_format()],
                true,
                false,
            ))
            .build()
    }
}

impl PreparedBasicGraph {
    /// Returns the immutable portable graph validated for WebGL2 capabilities.
    pub const fn compiled(&self) -> &CompiledGraph<()> {
        &self.compiled
    }
    /// Returns the declared insertion-ordered draw count.
    pub const fn draw_count(&self) -> usize {
        self.draw_count
    }
    /// Returns the declared drawing-buffer pixel extent.
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }
    /// Returns the closed format used by the imported presentable image.
    pub const fn format(&self) -> PresentableFormat {
        self.format
    }
}

impl PreparedBasicDraw {
    /// Returns owned position data for the closed indexed recipe.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Returns owned triangle-list indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    /// Returns the exact 80-byte ABI decoded as twenty f32 values.
    pub const fn pvm_and_color(&self) -> [f32; 20] {
        self.pvm_and_color
    }
    /// Returns the renderer insertion order.
    pub const fn insertion_index(&self) -> usize {
        self.insertion_index
    }
    pub(crate) const fn uniform(&self) -> &FrameUniform {
        &self.uniform
    }
}

impl OwnedScene {
    fn from_draw_list(list: &DrawList<'_>) -> Self {
        Self {
            camera: list.camera().clone(),
            draws: list
                .iter()
                .map(|item| OwnedDraw {
                    geometry: item.mesh().geometry().clone(),
                    material: item.mesh().material().clone(),
                    transform: item.transform(),
                })
                .collect(),
        }
    }
}

fn prepare_one(
    scene: &OwnedScene,
    index: usize,
) -> Result<PreparedBasicDraw, PreparedBasicSceneError> {
    let draw = scene.draws.get(index).ok_or_else(graph_error)?;
    let reason = |reason| PreparedBasicSceneError::Draw { index, reason };
    if draw.geometry.indices().is_empty() || !draw.geometry.indices().len().is_multiple_of(3) {
        return Err(reason(PreparedBasicDrawError::InvalidIndexCount));
    }
    let uniform =
        FrameUniform::new_with_model_transform(&scene.camera, &draw.material, draw.transform)
            .map_err(|error| {
                reason(match error {
                    FrameUniformError::ModelTransformProductNonFinite => {
                        PreparedBasicDrawError::ModelTransformProductNonFinite
                    }
                    _ => PreparedBasicDrawError::InvalidCameraMaterial,
                })
            })?;
    crate::fixed_frame::renderer::validate_clip(
        draw.geometry.positions(),
        draw.geometry.indices(),
        uniform.view_projection(),
    )
    .map_err(|error| reason(map_clip_error(error)))?;
    let mut pvm_and_color = [0.0; 20];
    for (index, chunk) in uniform.bytes().chunks_exact(4).enumerate() {
        pvm_and_color[index] = f32::from_bits(u32::from_le_bytes(
            chunk.try_into().expect("exact f32 chunk"),
        ));
    }
    Ok(PreparedBasicDraw {
        positions: draw.geometry.positions().to_vec(),
        indices: draw.geometry.indices().to_vec(),
        pvm_and_color,
        insertion_index: index,
        uniform,
    })
}

fn map_clip_error(error: crate::fixed_frame::DrawStartError) -> PreparedBasicDrawError {
    use crate::fixed_frame::DrawStartError;
    match error {
        DrawStartError::NonFinitePosition => PreparedBasicDrawError::NonFinitePosition,
        DrawStartError::NonFiniteClipPosition => PreparedBasicDrawError::NonFiniteClipPosition,
        DrawStartError::ClipWNonPositive => PreparedBasicDrawError::ClipWNonPositive,
        DrawStartError::ClipOutOfBounds => PreparedBasicDrawError::ClipOutOfBounds,
        _ => PreparedBasicDrawError::InvalidIndexCount,
    }
}

fn record_first_error(slot: &mut Option<PreparedBasicSceneError>, error: PreparedBasicSceneError) {
    let replace = match (&*slot, &error) {
        (None, _) => true,
        (
            Some(PreparedBasicSceneError::Draw {
                index: previous, ..
            }),
            PreparedBasicSceneError::Draw { index, .. },
        ) => index < previous,
        _ => false,
    };
    if replace {
        *slot = Some(error);
    }
}

fn take_error(errors: &Mutex<Option<PreparedBasicSceneError>>) -> Option<PreparedBasicSceneError> {
    errors.lock().ok().and_then(|mut errors| errors.take())
}

const fn graph_error() -> PreparedBasicSceneError {
    PreparedBasicSceneError::PreparationGraph
}

fn byte_size(count: usize, stride: u64) -> Result<u64, PreparedBasicGraphError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(stride))
        .ok_or(PreparedBasicGraphError::SizeOverflow)
}

const fn imported_buffer(size: u64) -> ImportBufferContract {
    ImportBufferContract {
        descriptor: BufferDesc { size },
        initial_state: ResourceAccessState::CopyDestination,
        ownership: ExternalOwnership::Caller,
        initial_contents: InitialContents::Defined,
    }
}

fn poll_ready<F: Future>(future: F) -> Result<F::Output, ()> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(value) => Ok(value),
        Poll::Pending => Err(()),
    }
}

impl fmt::Display for PreparedBasicSceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "prepared basic scene rejected: {self:?}")
    }
}
impl std::error::Error for PreparedBasicSceneError {}

impl fmt::Display for PreparedBasicGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "prepared basic graph rejected: {self:?}")
    }
}
impl std::error::Error for PreparedBasicGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for PreparationTaskFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("renderer CPU preparation rejected a draw")
    }
}
impl std::error::Error for PreparationTaskFailure {}

#[cfg(test)]
mod tests {
    use fluxel_rendergraph::{LoadOp, PassKind, ResourceAccessState, StoreOp};

    use super::*;
    use crate::Mesh;

    fn triangle(color: [f32; 4]) -> Mesh {
        Mesh::new(
            Geometry::from_positions(vec![[-0.2, -0.2, 0.5], [0.2, -0.2, 0.5], [0.0, 0.2, 0.5]])
                .with_indices(vec![0, 1, 2])
                .expect("fixed indices"),
            BasicMaterial::new(color).expect("fixed color"),
        )
    }

    #[test]
    fn shared_preparation_preserves_order_and_exact_uniform_values() {
        let red = triangle([1.0, 0.0, 0.0, 1.0]);
        let green = triangle([0.0, 1.0, 0.0, 1.0]);
        let camera = Camera::default();
        let mut list = DrawList::new(&camera);
        list.push(&red);
        list.push(&green);

        let prepared = PreparedBasicScene::prepare(&list).expect("scene prepares");
        assert_eq!(prepared.draws().len(), 2);
        assert_eq!(prepared.draws()[0].insertion_index(), 0);
        assert_eq!(prepared.draws()[1].insertion_index(), 1);
        assert_eq!(
            &prepared.draws()[0].pvm_and_color()[16..],
            &[1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            &prepared.draws()[1].pvm_and_color()[16..],
            &[0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn browser_graph_is_one_black_clear_store_and_present_contract() {
        let mesh = triangle([1.0, 0.0, 0.0, 1.0]);
        let camera = Camera::default();
        let mut list = DrawList::new(&camera);
        list.push(&mesh);
        let prepared = PreparedBasicScene::prepare(&list).expect("scene prepares");
        let graph = prepared
            .compile_presentable_graph([800, 600])
            .expect("graph compiles");
        let plan = graph.compiled().execution_plan();

        assert_eq!(graph.draw_count(), 1);
        assert_eq!(graph.extent(), [800, 600]);
        assert_eq!(plan.passes().len(), 1);
        assert_eq!(plan.passes()[0].kind, PassKind::Raster);
        let raster = plan.passes()[0].raster.as_ref().expect("raster plan");
        assert_eq!(raster.colors.len(), 1);
        assert_eq!(
            raster.colors[0].descriptor.operations.load,
            LoadOp::Clear([0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(raster.colors[0].descriptor.operations.store, StoreOp::Store);
        assert!(
            plan.final_transitions()
                .iter()
                .any(|transition| transition.after == ResourceAccessState::Present)
        );
    }

    #[test]
    fn closed_presentable_profiles_change_only_the_imported_format() {
        let mesh = triangle([1.0, 0.0, 0.0, 1.0]);
        let camera = Camera::default();
        let mut list = DrawList::new(&camera);
        list.push(&mesh);
        let prepared = PreparedBasicScene::prepare(&list).expect("scene prepares");

        let rgba = prepared
            .compile_presentable_graph_for_profile(
                PresentationProfile::for_format(PresentableFormat::Rgba8Unorm),
                [800, 600],
            )
            .expect("rgba graph compiles");
        let bgra = prepared
            .compile_presentable_graph_for_profile(
                PresentationProfile::for_format(PresentableFormat::Bgra8Unorm),
                [800, 600],
            )
            .expect("bgra graph compiles");

        assert_eq!(rgba.format(), PresentableFormat::Rgba8Unorm);
        assert_eq!(bgra.format(), PresentableFormat::Bgra8Unorm);
        assert_eq!(rgba.draw_count(), bgra.draw_count());
        assert_eq!(rgba.extent(), bgra.extent());
        let topology = |graph: &PreparedBasicGraph| {
            let plan = graph.compiled().execution_plan();
            (
                plan.passes()
                    .iter()
                    .map(|pass| {
                        (
                            pass.kind,
                            pass.transitions
                                .iter()
                                .map(|transition| (transition.before, transition.after))
                                .collect::<Vec<_>>(),
                            pass.raster
                                .as_ref()
                                .map(|raster| raster.colors.len())
                                .unwrap_or(0),
                        )
                    })
                    .collect::<Vec<_>>(),
                plan.final_transitions()
                    .iter()
                    .map(|transition| (transition.before, transition.after))
                    .collect::<Vec<_>>(),
                plan.resource_requirements()
                    .iter()
                    .map(|requirement| requirement.usage)
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(
            topology(&rgba),
            topology(&bgra),
            "format selection must not change fixed-pass topology"
        );
    }

    #[test]
    fn closed_presentable_profiles_reject_zero_sized_targets() {
        let mesh = triangle([1.0, 0.0, 0.0, 1.0]);
        let camera = Camera::default();
        let mut list = DrawList::new(&camera);
        list.push(&mesh);
        let prepared = PreparedBasicScene::prepare(&list).expect("scene prepares");

        for profile in [
            PresentationProfile::for_format(PresentableFormat::Rgba8Unorm),
            PresentationProfile::for_format(PresentableFormat::Bgra8Unorm),
        ] {
            assert!(matches!(
                prepared.compile_presentable_graph_for_profile(profile, [0, 600]),
                Err(PreparedBasicGraphError::InvalidExtent)
            ));
        }
    }
}
