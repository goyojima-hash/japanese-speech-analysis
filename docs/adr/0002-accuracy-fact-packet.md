# ADR-0002: Accuracy is a fact packet, not a correctness classifier

## Status

Accepted for implementation planning; model/provider selection remains deferred.

## Context

The Accuracy axis needs evidence about recognized speech and morphology. The system requirement is that every axis module remain independently selectable and produce facts only; an AI Judge performs evaluation later. Raw CTC confidence is often mistaken for learner accuracy, and a reference transcript may be unavailable or inappropriate for open speaking tasks.

## Decision

Add optional ASR alignment evidence to the shared bundle and implement Accuracy as an independent `AccuracyFactPacket` collector. Store compact uncalibrated CTC-derived observations, shared morphology facts, and optional explicit-reference differences. Do not store raw logits or emit correct/incorrect labels, scores, CEFR, corrections, or LLM output.

## Consequences

- Positive: independent module selection, auditability, and no hidden evaluation before the Judge.
- Cost: the first model provider requires explicit calibration work before confidence can inform stronger claims.
- Safeguard: every ASR-derived observation carries provenance, derivation, and calibration status; missing capabilities remain explicit.
