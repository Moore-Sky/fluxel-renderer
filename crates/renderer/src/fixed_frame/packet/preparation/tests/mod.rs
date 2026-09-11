//! CPU-only contract tests for the renderer preparation DAG and its ordering.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use slot_graph::{Graph, InputSpec, Local, NodeError, NodeOutputs, OutputSpec, RunInputs, Schema};

#[derive(Clone)]
struct ProbeScene(Vec<i32>);

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProbeDraw {
    index: usize,
    value: i32,
}

#[derive(Clone)]
struct ProbePacket(Vec<ProbeDraw>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProbeTrace {
    source_fan_out: usize,
    assembly_fan_in: usize,
    assembly_runs: usize,
}

fn run_probe(
    scene: ProbeScene,
    invalid: Option<usize>,
) -> Result<(Vec<ProbeDraw>, ProbeTrace), usize> {
    let assembly_runs = Arc::new(Mutex::new(0_usize));
    let mut graph = Graph::<Local>::new();

    let source_schema = Schema::builder()
        .input(InputSpec::required_one::<ProbeScene>("scene"))
        .output(OutputSpec::new::<ProbeScene>("scene"))
        .build()
        .bind();
    let source_input = source_schema.input::<ProbeScene>("scene").unwrap();
    let source_output = source_schema.output::<ProbeScene>("scene").unwrap();
    let source = graph
        .add_sync("source", source_schema, move |_, inputs| {
            let scene = inputs.required_key(source_input)?;
            let mut outputs = NodeOutputs::new();
            outputs.insert_shared_key(source_output, scene);
            Ok(outputs)
        })
        .unwrap();
    let external = graph
        .expose_input(graph.input::<ProbeScene>(source, "scene").unwrap())
        .unwrap();
    let source_output = graph.output::<ProbeScene>(source, "scene").unwrap();

    let draw_schema = Schema::builder()
        .input(InputSpec::required_one::<ProbeScene>("scene"))
        .output(OutputSpec::new::<ProbeDraw>("draw"))
        .build()
        .bind();
    let draw_input = draw_schema.input::<ProbeScene>("scene").unwrap();
    let draw_output = draw_schema.output::<ProbeDraw>("draw").unwrap();
    let mut nodes = Vec::new();
    for index in 0..scene.0.len() {
        let schema = draw_schema.clone();
        let node = graph
            .add_sync(format!("draw-{index}"), schema, move |_, inputs| {
                if invalid == Some(index) {
                    return Err(NodeError::user(ProbeFailure));
                }
                let scene = inputs.required_key(draw_input)?;
                let mut outputs = NodeOutputs::new();
                outputs.insert_key(
                    draw_output,
                    ProbeDraw {
                        index,
                        value: scene.0[index],
                    },
                );
                Ok(outputs)
            })
            .unwrap();
        graph
            .connect(
                source_output,
                graph.input::<ProbeScene>(node, "scene").unwrap(),
            )
            .unwrap();
        nodes.push(node);
    }

    let assembly_schema = Schema::builder()
        .input(InputSpec::required_many::<ProbeDraw>("draws"))
        .output(OutputSpec::new::<ProbePacket>("packet"))
        .build()
        .bind();
    let assembly_input = assembly_schema.input::<ProbeDraw>("draws").unwrap();
    let assembly_output = assembly_schema.output::<ProbePacket>("packet").unwrap();
    let assembly_runs_for_task = Arc::clone(&assembly_runs);
    let assembly = graph
        .add_sync("assembly", assembly_schema, move |_, inputs| {
            *assembly_runs_for_task.lock().unwrap() += 1;
            let mut outputs = NodeOutputs::new();
            outputs.insert_key(
                assembly_output,
                ProbePacket(
                    inputs
                        .many_key(assembly_input)?
                        .into_iter()
                        .map(|draw| (*draw).clone())
                        .collect(),
                ),
            );
            Ok(outputs)
        })
        .unwrap();
    let assembly_edges = graph
        .collect_into(
            nodes.clone(),
            graph.input::<ProbeDraw>(assembly, "draws").unwrap(),
        )
        .unwrap();
    graph.set_active(assembly, true).unwrap();
    let packet = graph.output::<ProbePacket>(assembly, "packet").unwrap();
    let version = graph.compile().unwrap();
    let mut inputs = RunInputs::new();
    inputs.insert(external, scene).unwrap();
    let mut report = poll_ready(version.execute(inputs))
        .map_err(|_| *assembly_runs.lock().unwrap())?
        .map_err(|_| *assembly_runs.lock().unwrap())?;
    let packet = report
        .take_output(packet)
        .map_err(|_| *assembly_runs.lock().unwrap())?;
    Ok((
        packet.0.clone(),
        ProbeTrace {
            source_fan_out: nodes.len(),
            assembly_fan_in: assembly_edges.len(),
            assembly_runs: *assembly_runs.lock().unwrap(),
        },
    ))
}

#[derive(Debug)]
struct ProbeFailure;

impl std::fmt::Display for ProbeFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("invalid draw")
    }
}

impl std::error::Error for ProbeFailure {}

fn poll_ready<F: Future>(future: F) -> Result<F::Output, ()> {
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    let mut future = std::pin::pin!(future);
    match Future::poll(Pin::as_mut(&mut future), &mut context) {
        Poll::Ready(value) => Ok(value),
        Poll::Pending => Err(()),
    }
}

#[test]
fn preparation_fans_out_then_assembles_in_insertion_order() {
    let (draws, trace) = run_probe(ProbeScene(vec![7, 3, 9]), None).unwrap();
    assert_eq!(
        draws,
        vec![
            ProbeDraw { index: 0, value: 7 },
            ProbeDraw { index: 1, value: 3 },
            ProbeDraw { index: 2, value: 9 },
        ]
    );
    assert_eq!(trace.source_fan_out, 3);
    assert_eq!(trace.assembly_fan_in, 3);
    assert_eq!(trace.assembly_runs, 1);
}

#[test]
fn changing_one_scene_object_only_changes_its_prepared_draw() {
    let (before, _) = run_probe(ProbeScene(vec![10, 20, 30]), None).unwrap();
    let (after, _) = run_probe(ProbeScene(vec![10, 99, 30]), None).unwrap();
    assert_eq!(before[0], after[0]);
    assert_ne!(before[1], after[1]);
    assert_eq!(before[2], after[2]);
}

#[test]
fn invalid_draw_prevents_assembly() {
    // The graph reports the failed preparation branch; without a successful
    // fan-in value, its dependent assembly task is never executed.
    assert_eq!(run_probe(ProbeScene(vec![1, 2, 3]), Some(1)), Err(0));
}
