# Fact-Module Default Alignment Regression Report

## Reproduced behavior

Calling `evaluate_speech_level(..., selected_modules=None)` selected only Fluency and Range, while the interactive terminal selected every implemented fact module. Consequently, an ordinary API evaluation omitted `accuracy_data` and `coherence_data` from both its objective data and its Judge input.

## Expected behavior

When no module subset is explicitly supplied, both entry points select the same four implemented fact modules: Fluency, Range, Accuracy, and Coherence. Explicit subsets remain available for module-only use.

## 5 Whys

1. Why did the API omit two implemented packets? Its default constant contained only `fluency` and `range`.
2. Why did the terminal differ? It passed the supported-module set explicitly, while the API relied on the default.
3. Why did that difference remain after the shared runner was introduced? The runner unified collection but intentionally retained an old response-compatibility policy.
4. Why was this now a defect? The product policy changed to require all implemented fact modules from both entry points by default.
5. Why will it not recur? `DEFAULT_FACT_MODULES` now aliases the one supported-module set and tests assert default API packets and Judge propagation.

## Fishbone controls

| Area | Control |
| --- | --- |
| Policy | One default selection: `SUPPORTED_FACT_MODULES` |
| Entry points | Both invoke `run_fact_modules` without divergent defaults |
| Optionality | Explicit `selected_modules` subsets remain validated and supported |
| Contract | API test asserts Accuracy and Coherence in response and Judge input |
| Documentation | Requirements, design, task plan, and current flow state the shared default |

## Timeline

1. Reproduced the differing default selections with focused runner and API tests.
2. Wrote failing assertions for all-default module selection and packet propagation.
3. Changed only the shared default selection constant; extraction algorithms and packet shapes were not changed.
4. Updated the shared-runner specification and flow diagram.
5. Verified focused tests, full suite, compilation, whitespace, and diagram rendering.

## Verification

- Focused: `python -m unittest tests.test_fact_modules tests.test_jgrade_api -v` — 11 passed.
- Full: `python -m unittest discover -s tests -v` — 84 passed.
- Static: `python -m compileall -q jgrade_eval` and `git diff --check` — passed.

## Recurrence prevention

Do not give API and terminal different implicit module defaults. Add a module once to `SUPPORTED_FACT_MODULES`; omitted selection then includes it in both paths. Keep explicit subsets only for callers that deliberately request partial evidence.
