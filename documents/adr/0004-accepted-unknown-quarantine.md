# ADR-0004: Quarantine accepted-unknown GPU work

**Status:** Accepted

## Context

A native submit error may occur after work has been accepted. Treating it as a
known rejection can release commands, staging memory, resources, or leases
while the GPU may still reference them.

## Decision

Distinguish known pre-submit rejection from accepted-unknown work and terminal
completion failure. Accepted-unknown work retains/quarantines all referenced
objects until safe retirement; it never publishes guessed outgoing state or a
ready snapshot.

## Alternatives

- Collapse all submit errors into one `Result` failure.
- Use queue-idle or a blocking drop as cleanup.

## Consequences

Pending operations are owning, non-blocking state machines. Renderer snapshot
reservations release before accepted work, but terminal failure or early drop
after acceptance poisons an affected generation when its state is unknown.

## Evidence

0.1.2 formalized structured completion and quarantine. 0.2.0 upload and
0.2.1–0.2.7 snapshot fault fixtures exercised partial acceptance, failure,
drop, and subsequent reuse behavior.

See [RHI design](../design-rhi.md) and [Renderer design](../design-renderer.md)
for the current lifecycle contract.
