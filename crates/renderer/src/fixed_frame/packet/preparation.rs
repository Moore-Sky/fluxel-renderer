//! Renderer-private CPU preparation DAG for multi-draw packet lowering.
//!
//! This adapter owns a short-lived [`slot_graph`] declaration for one packet:
//! one externally supplied, shared scene source fans out into per-draw
//! validation/uniform preparation and those results fan back in, in insertion
//! order, to packet assembly.  It deliberately has no executor, thread pool,
//! GPU resource, or synchronization responsibility.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use fluxel_rendergraph::DeviceIdentity;
use slot_graph::{Graph, InputSpec, Local, NodeError, NodeOutputs, OutputSpec, RunInputs, Schema};

use crate::{
    BasicMaterial, Camera, DrawList, Geometry, IndexedMeshSnapshot, ModelTransform,
    frame_uniform::FrameUniform,
};

use super::build::map_clip_error;
use super::{PacketDraw, RenderPacketBuildError, RenderPacketDrawBuildError};

/// Executes the current slice's CPU-only preparation dependencies synchronously.
///
/// The `Local` graph is intentionally created per lowering operation: this
/// stage has no cache or scheduling policy yet.  Its graph identities remain
/// entirely private; callers only receive the existing packet build errors.
pub(super) fn prepare_draws(
    device: DeviceIdentity,
    list: &DrawList<'_>,
    snapshots: &[IndexedMeshSnapshot],
) -> Result<Vec<PacketDraw>, RenderPacketBuildError> {
    let scene = OwnedScene::from_draw_list(list, snapshots);
    let errors = Arc::new(Mutex::new(None));
    let mut graph = Graph::<Local>::new();

    let source_schema = Schema::builder()
        .input(InputSpec::required_one::<OwnedScene>("scene"))
        .output(OutputSpec::new::<OwnedScene>("scene"))
        .build()
        .bind();
    let source_input = source_schema
        .input::<OwnedScene>("scene")
        .map_err(|_| preparation_graph())?;
    let source_output = source_schema
        .output::<OwnedScene>("scene")
        .map_err(|_| preparation_graph())?;
    let source = graph
        .add_sync("scene-source", source_schema, move |_, inputs| {
            let scene = inputs.required_key(source_input)?;
            let mut outputs = NodeOutputs::new();
            outputs.insert_shared_key(source_output, scene);
            Ok(outputs)
        })
        .map_err(|_| preparation_graph())?;
    let external_scene = graph
        .expose_input::<OwnedScene>(
            graph
                .input::<OwnedScene>(source, "scene")
                .map_err(|_| preparation_graph())?,
        )
        .map_err(|_| preparation_graph())?;
    let source_scene = graph
        .output::<OwnedScene>(source, "scene")
        .map_err(|_| preparation_graph())?;

    let draw_schema = Schema::builder()
        .input(InputSpec::required_one::<OwnedScene>("scene"))
        .output(OutputSpec::new::<PacketDraw>("draw"))
        .build()
        .bind();
    let draw_scene = draw_schema
        .input::<OwnedScene>("scene")
        .map_err(|_| preparation_graph())?;
    let draw_output = draw_schema
        .output::<PacketDraw>("draw")
        .map_err(|_| preparation_graph())?;
    let mut prepared_nodes = Vec::with_capacity(scene.draws.len());
    for index in 0..scene.draws.len() {
        let errors = Arc::clone(&errors);
        let draw_schema = draw_schema.clone();
        let node = graph
            .add_sync(
                format!("prepare-draw-{index}"),
                draw_schema,
                move |_, inputs| {
                    let scene = inputs.required_key(draw_scene)?;
                    match prepare_one(device, &scene, index) {
                        Ok(draw) => {
                            let mut outputs = NodeOutputs::new();
                            outputs.insert_key(draw_output, draw);
                            Ok(outputs)
                        }
                        Err(error) => {
                            // The public renderer error retains the existing
                            // draw index/reason contract; slot-graph's NodeError
                            // only stops this dependency branch.
                            if let Ok(mut stored) = errors.lock() {
                                record_first_error(&mut stored, error);
                            }
                            Err(NodeError::user(PreparationTaskFailure))
                        }
                    }
                },
            )
            .map_err(|_| preparation_graph())?;
        graph
            .connect(
                source_scene,
                graph
                    .input::<OwnedScene>(node, "scene")
                    .map_err(|_| preparation_graph())?,
            )
            .map_err(|_| preparation_graph())?;
        prepared_nodes.push(node);
    }

    let assembly_schema = Schema::builder()
        .input(InputSpec::required_many::<PacketDraw>("draws"))
        .output(OutputSpec::new::<PreparedDraws>("packet"))
        .build()
        .bind();
    let assembly_input = assembly_schema
        .input::<PacketDraw>("draws")
        .map_err(|_| preparation_graph())?;
    let assembly_output = assembly_schema
        .output::<PreparedDraws>("packet")
        .map_err(|_| preparation_graph())?;
    let assembly = graph
        .add_sync("assemble-packet", assembly_schema, move |_, inputs| {
            let draws = inputs.many_key(assembly_input)?;
            let mut outputs = NodeOutputs::new();
            outputs.insert_key(
                assembly_output,
                PreparedDraws(draws.into_iter().map(|draw| (*draw).clone()).collect()),
            );
            Ok(outputs)
        })
        .map_err(|_| preparation_graph())?;
    graph
        .collect_into(
            prepared_nodes,
            graph
                .input::<PacketDraw>(assembly, "draws")
                .map_err(|_| preparation_graph())?,
        )
        .map_err(|_| preparation_graph())?;
    graph
        .set_active(assembly, true)
        .map_err(|_| preparation_graph())?;
    let output = graph
        .output::<PreparedDraws>(assembly, "packet")
        .map_err(|_| preparation_graph())?;
    let version = graph.compile().map_err(|_| preparation_graph())?;
    let mut inputs = RunInputs::new();
    inputs
        .insert(external_scene, scene)
        .map_err(|_| preparation_graph())?;

    let mut report = match poll_ready(version.execute(inputs)) {
        Ok(Ok(report)) => report,
        Ok(Err(_)) => return Err(take_preparation_error(&errors).unwrap_or_else(preparation_graph)),
        Err(()) => return Err(preparation_graph()),
    };
    let packet = report
        .take_output(output)
        .map_err(|_| preparation_graph())?;
    Ok(packet.0.clone())
}

#[derive(Clone)]
struct OwnedScene {
    camera: Camera,
    draws: Vec<OwnedDraw>,
}

impl OwnedScene {
    fn from_draw_list(list: &DrawList<'_>, snapshots: &[IndexedMeshSnapshot]) -> Self {
        Self {
            camera: list.camera().clone(),
            draws: list
                .iter()
                .zip(snapshots)
                .map(|(item, snapshot)| OwnedDraw {
                    geometry: item.mesh().geometry().clone(),
                    material: item.mesh().material().clone(),
                    transform: item.transform(),
                    snapshot: snapshot.clone(),
                })
                .collect(),
        }
    }
}

#[derive(Clone)]
struct OwnedDraw {
    geometry: Geometry,
    material: BasicMaterial,
    transform: ModelTransform,
    snapshot: IndexedMeshSnapshot,
}

#[derive(Clone)]
struct PreparedDraws(Vec<PacketDraw>);

#[derive(Debug)]
struct PreparationTaskFailure;

impl std::fmt::Display for PreparationTaskFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("renderer CPU preparation rejected a draw")
    }
}

impl std::error::Error for PreparationTaskFailure {}

fn prepare_one(
    device: DeviceIdentity,
    scene: &OwnedScene,
    index: usize,
) -> Result<PacketDraw, RenderPacketBuildError> {
    let draw = scene.draws.get(index).ok_or_else(preparation_graph)?;
    let reason = |reason| RenderPacketBuildError::Draw { index, reason };
    if draw.snapshot.positions().buffer().device_identity() != device
        || draw.snapshot.indices().buffer().device_identity() != device
    {
        return Err(reason(RenderPacketDrawBuildError::ForeignSnapshotDevice));
    }
    if !positions_equal_by_bits(draw.geometry.positions(), draw.snapshot.position_metadata()) {
        return Err(reason(RenderPacketDrawBuildError::PositionMetadataMismatch));
    }
    if draw.geometry.indices() != draw.snapshot.index_metadata() {
        return Err(reason(RenderPacketDrawBuildError::IndexMetadataMismatch));
    }
    if draw.snapshot.index_count() == 0 || !draw.snapshot.index_count().is_multiple_of(3) {
        return Err(reason(RenderPacketDrawBuildError::InvalidIndexCount));
    }
    let uniform =
        FrameUniform::new_with_model_transform(&scene.camera, &draw.material, draw.transform)
            .map_err(|error| match error {
                crate::frame_uniform::FrameUniformError::ModelTransformProductNonFinite => {
                    reason(RenderPacketDrawBuildError::ModelTransformProductNonFinite)
                }
                _ => reason(RenderPacketDrawBuildError::InvalidCameraMaterial),
            })?;
    super::super::renderer::validate_clip(
        draw.snapshot.position_metadata(),
        draw.snapshot.index_metadata(),
        uniform.view_projection(),
    )
    .map_err(|error| reason(map_clip_error(error)))?;
    Ok(PacketDraw {
        snapshot: draw.snapshot.clone(),
        uniform,
    })
}

fn positions_equal_by_bits(left: &[[f32; 3]], right: &[[f32; 3]]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.iter()
                .zip(right)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        })
}

fn take_preparation_error(
    errors: &Mutex<Option<RenderPacketBuildError>>,
) -> Option<RenderPacketBuildError> {
    errors.lock().ok().and_then(|mut errors| errors.take())
}

fn record_first_error(slot: &mut Option<RenderPacketBuildError>, error: RenderPacketBuildError) {
    // Independent failed branches may all run before the graph reports its
    // aggregate failure. Preserve the same first-insertion-index diagnostic
    // that the pre-DAG sequential lowering loop exposed.
    let replace = match (&*slot, &error) {
        (None, _) => true,
        (
            Some(RenderPacketBuildError::Draw {
                index: previous, ..
            }),
            RenderPacketBuildError::Draw { index, .. },
        ) => index < previous,
        _ => false,
    };
    if replace {
        *slot = Some(error);
    }
}

fn preparation_graph() -> RenderPacketBuildError {
    RenderPacketBuildError::PreparationGraph
}

/// This graph is all synchronous today. A `Pending` result would mean a
/// future accidental async node entered a renderer path that promises no
/// executor or scheduling policy, so fail closed instead of spinning.
fn poll_ready<F: Future>(future: F) -> Result<F::Output, ()> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(value) => Ok(value),
        Poll::Pending => Err(()),
    }
}

#[cfg(test)]
mod tests;
