# ADR-0001: Use a versioned common evidence layer

## Status

Accepted

## Context

Five independently selectable speaking-axis modules need the same source audio, transcript, timings, and morphology. Re-running extraction per module is costly and risks inconsistent facts, while a shared scoring layer would violate module independence.

## Decision

Create a thin, immutable, versioned evidence layer. It extracts and stores only objective facts and provenance. Modules consume the evidence but do not call each other. AI Judges evaluate the collected facts downstream.

## Consequences

- Positive: consistent source evidence, optional modules, reusable cached preprocessing, auditable provenance.
- Negative: schema evolution and capability availability need explicit management.
- Mitigation: additive schemas, capability markers, opt-in cache, and compatibility adapters.
