//! Orchestrates validation, dependency discovery, liveness, and immutable plan lowering.
//!
//! Reference validation must precede graph traversal. RAW producer edges plus explicit
//! order/root edges define the reverse liveness closure; only after that closure is
//! stable may WAR edges be added, otherwise a dead reader could become observable.

use super::*;

pub(crate) fn compile_graph<F>(
    graph: &RenderGraph<F>,
    caps: &DeviceCapabilities,
) -> CompileResult<F> {
    let resources: HashMap<_, _> = graph.resources.iter().map(|r| (r.id, r)).collect();
    let pass_ids: HashSet<_> = graph.passes.iter().map(|p| p.id).collect();
    validate_references(graph, &resources, &pass_ids)?;
    let mut writers = HashMap::new();
    let mut readers: HashMap<VersionKey, Vec<PassId>> = HashMap::new();
    for pass in &graph.passes {
        for access in &pass.accesses {
            if access.read_required || access.output_version.is_none() {
                readers
                    .entry((access.resource, access.input_version))
                    .or_default()
                    .push(pass.id);
            }
            if let Some(v) = access.output_version {
                if let Some(old) = writers.insert((access.resource, v), pass.id) {
                    return Err(err(
                        CompileErrorKind::InvalidResourceBranch,
                        vec![old, pass.id],
                        Some(access.resource),
                        "multiple writers produced the same successor version",
                        None,
                    ));
                }
            }
        }
    }
    let mut deps = Vec::new();
    let mut seen = HashSet::new();
    for pass in &graph.passes {
        for access in &pass.accesses {
            let key = (access.resource, access.input_version);
            if let Some(&producer) = writers.get(&key) {
                push_dep(&mut deps, &mut seen, producer, pass.id, access.resource);
            } else if access.input_version != 0 {
                return Err(err(
                    CompileErrorKind::StaleOrForeignVersion,
                    vec![pass.id],
                    Some(access.resource),
                    "input version has no producer",
                    None,
                ));
            }
        }
    }
    let mut retained = HashSet::new();
    for root in &graph.roots {
        match root {
            RootDecl::Texture(_, r, v, _)
            | RootDecl::Buffer(_, r, v, _)
            | RootDecl::Present(_, r, v, _) => {
                if let Some(&p) = writers.get(&(*r, *v)) {
                    retained.insert(p);
                }
            }
            RootDecl::SideEffect(p, _) => {
                retained.insert(*p);
            }
        }
    }
    let mut pred: HashMap<PassId, Vec<PassId>> = HashMap::new();
    for edge in &deps {
        pred.entry(edge.consumer).or_default().push(edge.producer);
    }
    for order in &graph.orders {
        pred.entry(order.after).or_default().push(order.before);
    }
    let mut pending: Vec<_> = retained.iter().copied().collect();
    while let Some(p) = pending.pop() {
        if let Some(items) = pred.get(&p) {
            for &before in items {
                if retained.insert(before) {
                    pending.push(before);
                }
            }
        }
    }
    // WAR orders matter only between passes that are independently live. A
    // dead observer must not become observable merely because a later live
    // pass consumes the version it would have read.
    for pass in &graph.passes {
        if !retained.contains(&pass.id) {
            continue;
        }
        for access in &pass.accesses {
            if access.output_version.is_none() {
                continue;
            }
            if let Some(items) = readers.get(&(access.resource, access.input_version)) {
                for &reader in items {
                    if retained.contains(&reader) && reader != pass.id {
                        push_dep(&mut deps, &mut seen, reader, pass.id, access.resource);
                    }
                }
            }
        }
    }
    let deps: Vec<_> = deps
        .into_iter()
        .filter(|e| retained.contains(&e.producer) && retained.contains(&e.consumer))
        .collect();
    let orders: Vec<_> = graph
        .orders
        .iter()
        .filter(|e| retained.contains(&e.before) && retained.contains(&e.after))
        .map(|e| ExplicitOrder {
            before: e.before,
            after: e.after,
            reason: e.reason.clone(),
        })
        .collect();
    let execution_order = topo(&graph.passes, &retained, &deps, &orders)?;
    validate_initialization(graph, &resources, &retained, &writers)?;
    validate_capabilities(graph, &resources, &retained, caps)?;
    validate_roots(graph, &resources, &writers, caps)?;
    let passes: Vec<PassDecl<F>> = graph
        .passes
        .iter()
        .filter(|p| retained.contains(&p.id))
        .cloned()
        .collect();
    let culled_passes = graph
        .passes
        .iter()
        .filter(|p| !retained.contains(&p.id))
        .map(|p| p.id)
        .collect();
    let retained_side_effects = graph
        .roots
        .iter()
        .filter_map(|root| match root {
            RootDecl::SideEffect(pass, reason) if retained.contains(pass) => {
                Some(RetainedSideEffect {
                    pass: *pass,
                    reason: reason.clone(),
                })
            }
            _ => None,
        })
        .collect();
    let execution_plan = build_execution_plan(
        &passes,
        &execution_order,
        &graph.resources,
        &graph.roots,
        caps,
    );
    Ok(CompileOutput {
        graph: CompiledGraph {
            identity: NEXT_COMPILED_ID.fetch_add(1, Ordering::Relaxed),
            passes,
            execution_order,
            resources: graph.resources.clone(),
            roots: graph.roots.clone(),
            _dependencies: deps.clone(),
            _explicit_orders: orders.clone(),
            capability_fingerprint: caps.fingerprint(),
            execution_plan,
        },
        report: CompileReport {
            culled_passes,
            inferred_dependencies: deps,
            explicit_orders: orders,
            retained_side_effects,
            capability_fallbacks: Vec::new(),
        },
    })
}
