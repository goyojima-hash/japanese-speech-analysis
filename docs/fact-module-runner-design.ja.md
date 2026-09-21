# 共通 Fact-Module Runner 設計

`jgrade_eval/fact_modules.py` が、共有 Evidence 生成後の選択ポリシー検証と収集を担当します。

```text
EvidenceBundle
    -> run_fact_modules(evidence, selected_modules, range_extractor)
    -> FactModuleRun(active_modules, packets)
    -> merge_objective_data(base_data)
    -> add_packets_to_roleplay_input(base_input)
```

`FactModuleRun` は選択済みのパケットキー（`range_data`、`accuracy_data`、`coherence_data`）だけを持ちます。Fluency は `SpeechEvidence` から生成する基礎事実であり、公開上の選択名の一貫性のために含めますが、二つ目の収集器は持ちません。

runner が対応モジュール名と共通の既定（実装済み全モジュール）を定義します。そのため、選択を渡さない API とターミナルは同じモジュール集合を実行します。呼び出し元は、単独利用または組み合わせのために対応済みの部分集合を明示選択できます。これは二つ目の収集実装ではなく、選択ポリシーです。

対話表示は任意のパケットキーを読み、存在するものだけを表示します。個別事実モジュールを直接 import／呼び出ししません。

`run_fact_modules` は Range、Accuracy、Coherence 順の決定的な開始／完了コールバックを任意に発行します。ターミナルは進捗表示と完了直後の表示だけに使い、API はコールバックを渡さないため収集は変わりません。
