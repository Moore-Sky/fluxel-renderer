//! Derives resource dependencies and deterministic topological execution order.

use super::*;

pub(super) fn push_dep(
    out: &mut Vec<PassDependency>,
    seen: &mut HashSet<(PassId, PassId, ResourceId)>,
    a: PassId,
    b: PassId,
    r: ResourceId,
) {
    if a != b && seen.insert((a, b, r)) {
        out.push(PassDependency {
            producer: a,
            consumer: b,
            resource: r,
        });
    }
}

pub(super) fn topo<F>(
    passes: &[PassDecl<F>],
    retained: &HashSet<PassId>,
    deps: &[PassDependency],
    orders: &[ExplicitOrder],
) -> Result<Vec<PassId>, CompileError> {
    let mut degree: HashMap<_, _> = retained.iter().map(|&p| (p, 0usize)).collect();
    let mut out: HashMap<PassId, Vec<PassId>> = HashMap::new();
    let mut seen = HashSet::new();
    for (a, b) in deps
        .iter()
        .map(|e| (e.producer, e.consumer))
        .chain(orders.iter().map(|e| (e.before, e.after)))
    {
        if seen.insert((a, b)) {
            *degree.get_mut(&b).unwrap() += 1;
            out.entry(a).or_default().push(b);
        }
    }
    let index: HashMap<_, _> = passes.iter().enumerate().map(|(i, p)| (p.id, i)).collect();
    let mut ready: Vec<_> = degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(&p, _)| p)
        .collect();
    ready.sort_by_key(|p| index[p]);
    let mut q: VecDeque<_> = ready.into();
    let mut result = Vec::new();
    while let Some(p) = q.pop_front() {
        result.push(p);
        if let Some(items) = out.get(&p) {
            let mut new = Vec::new();
            for &n in items {
                let d = degree.get_mut(&n).unwrap();
                *d -= 1;
                if *d == 0 {
                    new.push(n)
                }
            }
            new.sort_by_key(|p| index[p]);
            q.extend(new)
        }
    }
    if result.len() != retained.len() {
        let mut cycle: Vec<_> = degree
            .into_iter()
            .filter(|(_, d)| *d > 0)
            .map(|(p, _)| p)
            .collect();
        cycle.sort_by_key(|pass| index[pass]);
        Err(err(
            CompileErrorKind::DependencyCycle,
            cycle,
            None,
            "dependency cycle",
            None,
        ))
    } else {
        Ok(result)
    }
}
