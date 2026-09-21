# 現在の構成図 — 共通Evidence層と5要素モジュール

全体を読むための図です。共通Evidence層が最初に1回だけ事実を作り、各モジュールはその同じEvidenceを独立して読みます。採点・正誤・CEFR判定は、最後のAI Judgeと協議層にだけあります。

```mermaid
flowchart TB
    User([利用者]) --> Console["jgrade_eval/interactive.py<br/>run_interactive()"]
    Console --> Common
    subgraph Common["共通Evidence層 — 最初に1回だけ実行"]
        direction LR
        Pipeline["evidence/pipeline.py<br/>EvidencePipeline"]
        FluencyInput["audio_pipeline.py<br/>FluencyExtractor"]
        Tokenizer["evidence/linguistic.py<br/>SudachiTokenizer"]
        Bundle["evidence/models.py<br/>EvidenceBundle v1"]
        Pipeline --> FluencyInput --> Tokenizer --> Bundle
    end
    Common --> Fluency & Range & Accuracy
    Common -. "将来追加" .-> Interaction & Coherence
    subgraph Fluency["Fluencyモジュール（実装済み）"]
        F["事実: 発話率・ポーズ<br/>モーラ/秒"]
    end
    subgraph Range["Rangeモジュール（実装済み）"]
        R["range.py<br/>事実: 語彙統計・JLPT照合"]
    end
    subgraph Accuracy["Accuracyモジュール（実装済み）"]
        A["accuracy.py<br/>事実: 時刻・形態素観測"]
    end
    subgraph Interaction["Interactionモジュール（未実装）"]
        I["将来: 応答ラグ・ターン交替"]
    end
    subgraph Coherence["Coherenceモジュール（未実装）"]
        C["将来: 接続表現・文のつながり"]
    end
    Fluency & Range & Accuracy --> Facts
    Interaction & Coherence -.-> Facts
    Facts["客観的な事実パケット<br/>採点・正誤・CEFRなし"] --> Judge
    subgraph Evaluation["後段の評価層 — ここだけが評価する"]
        Judge["prompts.py<br/>AI Judge"] --> Deliberation["deliberation.py<br/>CEFR協議"]
    end
    Deliberation --> Result([CEFR推定・人間確認要否])
```

- 実線: 実装済みのデータフロー
- 点線: 将来のモジュール追加時の接続
- 詳しい呼び出し順は [UMLシーケンス図](accuracy-current-architecture.md) を参照
