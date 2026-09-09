//! Immutable compile outputs and inspectable compiler decisions.

use super::*;

/// A resource-derived execution dependency between two logical passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PassDependency {
    /// Producer pass.
    pub producer: PassId,
    /// Consumer pass.
    pub consumer: PassId,
    /// Resource causing the dependency.
    pub resource: ResourceId,
}

/// A user-declared ordering edge with no resource access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplicitOrder {
    /// First pass.
    pub before: PassId,
    /// Second pass.
    pub after: PassId,
    /// Author supplied reason.
    pub reason: String,
}

/// One safely degraded execution strategy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapabilityFallback {
    /// Affected pass, when pass-specific.
    pub pass: Option<PassId>,
    /// Selected strategy.
    pub description: String,
}

/// A pass retained as a graph root for a declared non-resource side effect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedSideEffect {
    /// Retained pass.
    pub pass: PassId,
    /// Observable reason the pass must not be culled.
    pub reason: crate::SideEffectReason,
}

/// Inspectable compiler decisions.
#[derive(Clone, Debug)]
pub struct CompileReport {
    /// Passes removed because no root depends on them.
    pub culled_passes: Vec<PassId>,
    /// Dependencies inferred from resource versions.
    pub inferred_dependencies: Vec<PassDependency>,
    /// Retained explicit ordering edges.
    pub explicit_orders: Vec<ExplicitOrder>,
    /// Non-resource side-effect roots retained during culling.
    pub retained_side_effects: Vec<RetainedSideEffect>,
    /// Safe capability degradations.
    pub capability_fallbacks: Vec<CapabilityFallback>,
}

/// An immutable, reusable compile-only graph plan.
pub struct CompiledGraph<F = ()> {
    pub(crate) identity: u64,
    pub(crate) passes: Vec<PassDecl<F>>,
    pub(super) execution_order: Vec<PassId>,
    pub(crate) resources: Vec<ResourceDecl>,
    pub(crate) roots: Vec<RootDecl>,
    pub(super) _dependencies: Vec<PassDependency>,
    pub(super) _explicit_orders: Vec<ExplicitOrder>,
    pub(crate) capability_fingerprint: crate::rhi::CapabilityFingerprint,
    pub(super) execution_plan: ExecutionPlan,
}

impl<F> std::fmt::Debug for CompiledGraph<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("pass_count", &self.passes.len())
            .field("execution_order", &self.execution_order)
            .finish()
    }
}
impl<F> CompiledGraph<F> {
    /// Returns the deterministic topological order of retained passes.
    pub fn execution_order(&self) -> &[PassId] {
        &self.execution_order
    }

    /// Returns the reusable single-queue execution plan.
    pub fn execution_plan(&self) -> &ExecutionPlan {
        &self.execution_plan
    }
}

/// Successful compilation and diagnostics.
#[derive(Debug)]
pub struct CompileOutput<F = ()> {
    /// Immutable snapshot.
    pub graph: CompiledGraph<F>,
    /// Compiler report.
    pub report: CompileReport,
}
/// Graph compilation result.
pub type CompileResult<F = ()> = Result<CompileOutput<F>, CompileError>;
pub(super) type VersionKey = (ResourceId, u32);
