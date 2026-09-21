# Shared Fact-Module Runner Requirements

## Goal

Use one fact-module execution path after `EvidenceBundle` creation. The API and interactive console must both invoke the same runner and receive the same packet shapes for Range, Accuracy, and Coherence.

## Requirements

- The runner accepts already-built `EvidenceBundle` and never runs speech extraction or tokenisation.
- Module selection, validation, Range evidence consumption, and packet assembly exist only in the runner.
- When `selected_modules` is omitted, the runner selects every implemented fact module: Fluency, Range, Accuracy, and Coherence.
- The API and interactive console use that same all-implemented-modules default. A caller can still explicitly select a subset for module-only use.
- Terminal progress is ordered as shared Evidence, Fluency, Range, Accuracy, Coherence, Judge, and deliberation. Each module packet is displayed immediately after that collector completes.
- Both callers construct Judge input from the runner result; selected packet keys are copied automatically without caller-specific module conditionals.
- Unselected packets are absent from both objective data and Judge input.
- No change to `EvidenceBundle`, Fluency, Range, Accuracy, or Coherence extraction algorithms.

## Acceptance criteria

1. API and console call `run_fact_modules`, not individual Range/Accuracy/Coherence collectors.
2. A single `LinguisticEvidence` instance is read by every selected text module.
3. Console output contains `=== Coherence客観データ ===` and its Judge input contains `coherence_data`.
4. Default API and console runs report all four implemented modules and include Range, Accuracy, and Coherence packets in both objective data and Judge input.
