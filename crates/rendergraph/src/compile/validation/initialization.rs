//! Tracks defined content across resource versions and declared ranges.

use super::super::*;

#[derive(Clone)]
enum Validity {
    Buffer(Vec<(u64, u64)>),
    Texture(HashSet<(u32, u32, u8)>),
}
impl Validity {
    fn empty(r: &ResourceDecl) -> Self {
        match r.kind {
            ResourceKind::Buffer(_) => Self::Buffer(Vec::new()),
            ResourceKind::Texture(_) => Self::Texture(HashSet::new()),
        }
    }
    fn full(r: &ResourceDecl) -> Self {
        let mut s = Self::empty(r);
        let range = match r.kind {
            ResourceKind::Buffer(_) => DeclRange::Buffer(BufferRange::Whole),
            ResourceKind::Texture(_) => DeclRange::Texture(TextureRange::Whole),
        };
        s.add(r, range);
        s
    }
    fn contains(&self, r: &ResourceDecl, range: DeclRange) -> bool {
        let mut need = Self::empty(r);
        need.add(r, range);
        match (self, need) {
            (Self::Buffer(h), Self::Buffer(n)) => n
                .iter()
                .all(|&(a, b)| h.iter().any(|&(s, e)| s <= a && e >= b)),
            (Self::Texture(h), Self::Texture(n)) => n.is_subset(h),
            _ => false,
        }
    }
    fn add(&mut self, r: &ResourceDecl, range: DeclRange) {
        match self {
            Self::Buffer(v) => {
                v.push(buffer_bounds(r, range));
                v.sort_unstable();
                let mut m: Vec<(u64, u64)> = Vec::new();
                for &(s, e) in v.iter() {
                    if let Some(last) = m.last_mut() {
                        if s <= last.1 {
                            last.1 = last.1.max(e);
                            continue;
                        }
                    }
                    m.push((s, e));
                }
                *v = m
            }
            Self::Texture(s) => s.extend(texture_keys(r, range)),
        }
    }
    fn remove(&mut self, r: &ResourceDecl, range: DeclRange) {
        match self {
            Self::Buffer(v) => {
                let (s, e) = buffer_bounds(r, range);
                let mut n = Vec::new();
                for &(a, b) in v.iter() {
                    if b <= s || a >= e {
                        n.push((a, b))
                    } else {
                        if a < s {
                            n.push((a, s))
                        }
                        if b > e {
                            n.push((e, b))
                        }
                    }
                }
                *v = n
            }
            Self::Texture(s) => {
                for k in texture_keys(r, range) {
                    s.remove(&k);
                }
            }
        }
    }
}
pub(in crate::compile) fn buffer_bounds(r: &ResourceDecl, range: DeclRange) -> (u64, u64) {
    let ResourceKind::Buffer(d) = r.kind else {
        unreachable!()
    };
    match range {
        DeclRange::Buffer(BufferRange::Whole) => (0, d.size),
        DeclRange::Buffer(BufferRange::Bytes { offset, size }) => (offset, offset + size),
        _ => unreachable!(),
    }
}
pub(in crate::compile) fn texture_keys(r: &ResourceDecl, range: DeclRange) -> Vec<(u32, u32, u8)> {
    let ResourceKind::Texture(d) = r.kind else {
        unreachable!()
    };
    let (bm, mc, ba, ac, a) = match range {
        DeclRange::Texture(TextureRange::Whole) => {
            (0, d.mip_levels, 0, d.array_layers, TextureAspect::All)
        }
        DeclRange::Texture(TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        }) => (
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        ),
        _ => unreachable!(),
    };
    let aspects: &[u8] = match a {
        TextureAspect::All if d.format == TextureFormat::Depth32Float => &[1],
        TextureAspect::All | TextureAspect::Color => &[0],
        TextureAspect::Depth => &[1],
        TextureAspect::Stencil => &[2],
    };
    let mut out = Vec::new();
    for m in bm..bm + mc {
        for l in ba..ba + ac {
            for &a in aspects {
                out.push((m, l, a));
            }
        }
    }
    out
}

pub(in crate::compile) fn validate_initialization<F>(
    graph: &RenderGraph<F>,
    resources: &HashMap<ResourceId, &ResourceDecl>,
    retained: &HashSet<PassId>,
    writers: &HashMap<VersionKey, PassId>,
) -> Result<(), CompileError> {
    let mut states = HashMap::new();
    for r in resources.values() {
        let defined = match r.origin {
            ResourceOrigin::TextureImport(_, c) => c.initial_contents == InitialContents::Defined,
            ResourceOrigin::BufferImport(_, c) => c.initial_contents == InitialContents::Defined,
            _ => false,
        };
        states.insert(
            (r.id, 0),
            if defined {
                Validity::full(r)
            } else {
                Validity::empty(r)
            },
        );
    }
    let mut writes: Vec<_> = graph
        .passes
        .iter()
        .filter(|p| retained.contains(&p.id))
        .flat_map(|p| p.accesses.iter().filter(|a| a.output_version.is_some()))
        .collect();
    writes.sort_by_key(|a| (a.resource.0, a.output_version.unwrap()));
    for a in writes {
        let pass = writers[&(a.resource, a.output_version.unwrap())];
        let r = resources[&a.resource];
        let Some(mut state) = states.get(&(a.resource, a.input_version)).cloned() else {
            return Err(err(
                CompileErrorKind::StaleOrForeignVersion,
                vec![pass],
                Some(a.resource),
                "predecessor has no content state",
                None,
            ));
        };
        if a.read_required && !state.contains(r, a.range) {
            return Err(err(
                CompileErrorKind::ReadBeforeInitialization,
                vec![pass],
                Some(a.resource),
                "read requires initialized contents",
                None,
            ));
        }
        if a.invalidate_before {
            state.remove(r, a.range)
        }
        if a.coverage == WriteCoverage::Full {
            state.add(r, a.range)
        }
        if a.discard_after {
            state.remove(r, a.range)
        }
        states.insert((a.resource, a.output_version.unwrap()), state);
    }
    for p in graph.passes.iter().filter(|p| retained.contains(&p.id)) {
        for a in p.accesses.iter().filter(|a| a.output_version.is_none()) {
            let r = resources[&a.resource];
            if !matches!(states.get(&(a.resource, a.input_version)), Some(s) if s.contains(r, a.range))
            {
                return Err(err(
                    CompileErrorKind::ReadBeforeInitialization,
                    vec![p.id],
                    Some(a.resource),
                    "read requires initialized contents",
                    None,
                ));
            }
        }
    }
    for root in &graph.roots {
        if let RootDecl::Texture(_, r, v, _)
        | RootDecl::Buffer(_, r, v, _)
        | RootDecl::Present(_, r, v, _) = *root
        {
            let d = resources[&r];
            let range = match d.kind {
                ResourceKind::Texture(_) => DeclRange::Texture(TextureRange::Whole),
                ResourceKind::Buffer(_) => DeclRange::Buffer(BufferRange::Whole),
            };
            if !matches!(states.get(&(r, v)), Some(s) if s.contains(d, range)) {
                return Err(err(
                    CompileErrorKind::InvalidExportOrPresent,
                    Vec::new(),
                    Some(r),
                    "root contents are not fully initialized",
                    None,
                ));
            }
        }
    }
    Ok(())
}
