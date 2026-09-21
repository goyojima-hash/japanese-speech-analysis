# ADR 0003: Use a deterministic Coherence fact baseline before a model provider

## Status

Accepted for design; implementation pending.

## Context

J-GRADE needs a Coherence module while preserving already shipped Fluency, Range, and Accuracy facts. Shared Evidence v1 has a Japanese transcript, Sudachi tokens with character offsets, and pause spans, but it has no validated sentence punctuation, dependency graph, coreference graph, or token-to-time alignment.

## Decision

Create a deterministic, fact-only Coherence collector over Evidence v1. It reports explicit connectives, conservatively derived candidate units, cross-unit lexical repetitions, and copied pause spans. It does not use a new model or mutate common evidence.

KWJA is deferred behind an optional observation-provider protocol. Its output will be model-derived evidence with provenance, never a replacement for shared Evidence or a level decision.

## Consequences

- Positive: minimal regression surface; deterministic outputs; zero new runtime model dependency; easy teacher review.
- Negative: no implicit relation, dependency, or coreference observation in V1; units are candidates rather than linguistically guaranteed sentences.
- Guardrail: neither deterministic nor future model observations may contain a quality score or learner-error judgement.
