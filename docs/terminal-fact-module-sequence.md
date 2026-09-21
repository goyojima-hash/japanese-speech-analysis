# Terminal Fact-Module Sequence

The terminal selects every implemented fact module, but module collection exists only once in `fact_modules.py`. The three modules consume the same immutable shared evidence and do not call one another.

```mermaid
sequenceDiagram
    actor User as 利用者
    participant Console as interactive.py
    participant Evidence as evidence/pipeline.py
    participant Runner as fact_modules.py
    participant Range as range.py
    participant Accuracy as accuracy.py
    participant Coherence as coherence.py
    participant Judge as AI Judge

    User->>Console: 音声を選択
    Console->>Evidence: build(audio_path)
    Evidence-->>Console: EvidenceBundle v1
    Console->>Runner: Range + Accuracy + Coherence
    Runner->>Range: 共有トークン
    Runner->>Accuracy: 共有Evidence
    Runner->>Coherence: 共有Evidence
    Runner-->>Console: 3つの事実パケット
    Console->>Judge: 基礎事実 + 3パケット
```
