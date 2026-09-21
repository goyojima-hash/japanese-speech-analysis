# Interactive Console Evidence Output Report

## Problem

The interactive console constructed the shared `EvidenceBundle`, but did not show its factual output. It also retained the older Fluency/Range-only orchestration, so it neither invoked the implemented Accuracy module nor passed `accuracy_data` to the Judge. The API selection route was correct; the CLI route was incomplete.

## Reproduction evidence

1. Start `./open_jgrade_console_terminal.command` and select an audio file.
2. Observe the console move from `[1/5] Fluency` to `[2/5] Range` and then directly to `[3/5] Judge`.
3. Observe that neither `=== 共通Evidence層客観データ ===` nor `=== Accuracy客観データ ===` is printed.
4. Run the new focused console test. Before the repair it fails with `KeyError: 'accuracy_data'` in the captured Judge input.

## Timeline

- Evidence v1 was integrated as a shared fact layer.
- Accuracy was added and made selectable on the API route.
- A real Terminal run exposed that the interactive route still used its previous two-module sequence.
- A focused interactive regression test reproduced the missing `accuracy_data` before production code changed.

## 5 Whys

1. Why was Accuracy absent from the console? `run_interactive` never called `AccuracyModule.collect`.
2. Why did it never call it? The console retained the old Fluency/Range orchestration after the API integration.
3. Why was shared evidence invisible? No console printer had been added for the already-built `EvidenceBundle`.
4. Why did this escape tests? The prior interactive test asserted only Range input and phase labels.
5. Why is that insufficient? A module is not delivered to console users unless its factual packet is both produced and visible/forwarded.

## Fishbone

- **Code:** API and interactive orchestration duplicated the selected-fact assembly.
- **Integration:** Accuracy was wired only in the API route.
- **Presentation:** shared facts had no concise CLI representation.
- **Tests:** the interactive contract omitted Accuracy and common-layer assertions.
- **Process:** a real Terminal smoke test occurred after, rather than within, the console integration checks.

## Repair and recurrence prevention

- Print one concise, fact-only common Evidence summary immediately after the bundle is built.
- Run Accuracy independently from the same bundle after Range; print its fact packet and send it to the Judge input.
- Renumber the visible console phases so Judge and deliberation follow the three fact modules.
- Keep the new focused interactive regression test: it asserts common/Accuracy headers and the `accuracy_data` Judge boundary.
- Re-run focused, complete, compile, and diff checks before restarting the console.

