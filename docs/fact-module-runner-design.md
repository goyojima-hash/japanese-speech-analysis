# Shared Fact-Module Runner Design

`jgrade_eval/fact_modules.py` owns selection policy validation and collection after shared evidence exists.

```text
EvidenceBundle
    -> run_fact_modules(evidence, selected_modules, range_extractor)
    -> FactModuleRun(active_modules, packets)
    -> merge_objective_data(base_data)
    -> add_packets_to_roleplay_input(base_input)
```

`FactModuleRun` contains only selected packet keys (`range_data`, `accuracy_data`, `coherence_data`). Fluency remains the base facts generated from `SpeechEvidence`; it is included in selections for consistent public reporting but has no second collector.

The runner defines supported module names and the common default: every implemented module. The API and console therefore execute the same module set when no selection is supplied. A caller may still name any supported subset to run a module alone or combine selected modules; this is selection policy, not a second collection implementation.

The interactive display reads optional packet keys and prints only those present. It has no direct imports or calls to individual fact modules.

`run_fact_modules` optionally emits deterministic start/result callbacks in Range, Accuracy, Coherence order. The console uses them only for progress and immediate rendering; API collection is unchanged and supplies no callbacks.
