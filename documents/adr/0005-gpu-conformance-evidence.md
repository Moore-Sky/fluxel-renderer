# ADR-0005: Require GPU conformance evidence for native correctness claims

**Status:** Accepted

## Context

Compilation, mocks, and CPU protocol tests cannot prove that DX12 and Vulkan
commands, barriers, shaders, and readback agree with portable semantics.

## Decision

Claim native correctness only when the same compiled ExecutionPlan runs on
DX12 and Vulkan with Required validation and an independent CPU oracle.
Artifacts identify the exact commit, backend, hardware/driver, inputs,
expected/actual result, completion, outgoing state, and diagnostics.

## Alternatives

- Treat CI compilation, TestRhi, or one backend demo as conformance.
- Let readback assume a convenient state instead of consuming an export's
  reported outgoing state.

## Consequences

Hardware fixtures are distinct from ordinary tests and are rerun on the final
commit before release. CPU and compile evidence remain useful but are labeled
as such; they do not substitute for native oracle evidence.

## Evidence

0.1.2 C01–C03, 0.1.3 K01–K02, and 0.1.4 R01/R02/X01 established the pattern.
0.2.0–0.2.7 extended it through U01–U08 and exact-SHA release reruns.

See [RHI design](../design-rhi.md) for the current test-evidence boundary.
