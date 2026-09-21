# Shared Fact-Module Runner Bug Hunter Report

## Reproduced regression

The API could select `coherence`, but the interactive console directly called Range and Accuracy and never selected Coherence. The console therefore omitted both the Coherence display and `coherence_data` in Judge input.

## 5 Whys

1. Why was Coherence absent in the console? The console had its own module orchestration.
2. Why did it have its own orchestration? API and console assembled fact packets independently.
3. Why was that risky? Adding a module required two hand-maintained changes.
4. Why did the omission escape? Tests asserted individual console packets but not a common execution contract.
5. Why can it not recur now? Both callers use `run_fact_modules`; tests verify selected packet propagation and console Coherence output.

## Fishbone controls

| Cause area | Control |
| --- | --- |
| Architecture | One runner owns module validation and collection |
| Integration | `FactModuleRun` copies selected packets to objective and Judge data |
| Compatibility | API default selection remains unchanged |
| Console | Explicit all-implemented-module profile uses the same runner |
| Testing | RED runner and interactive Coherence tests; full regression suite |

## Timeline

1. Reproduced missing runner and missing console Coherence data with RED tests.
2. Added the shared runner and refactored API/console callers.
3. Added Coherence console rendering from optional packet data.
4. Verified focused tests, complete suite, compilation, and whitespace checks.

## Recurrence prevention

No API or console code directly collects Range, Accuracy, or Coherence. Any new fact module is registered and collected once in `fact_modules.py`; both entry points receive it through the same packet mechanism.
