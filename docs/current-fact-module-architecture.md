# Current Fact-Module Sequence

Both entry points use `fact_modules.py` after building common evidence. With no explicit selection, the interactive console and API both select every implemented module and use the identical runner and packet contract. A caller can still request a supported subset.

```mermaid
sequenceDiagram
    autonumber
    actor User as 利用者
    participant Console as interactive.py<br/>対話ターミナル
    participant API as api_service.py<br/>API
    participant Evidence as evidence/<br/>共通Evidence層
    participant Runner as fact_modules.py<br/>共通runner
    participant Range as range.py<br/>Range
    participant Accuracy as accuracy.py<br/>Accuracy
    participant Coherence as coherence.py<br/>Coherence
    participant Judge as AI Judge
    participant Delib as deliberation.py<br/>CEFR協議

    alt ターミナル実行
        User->>Console: 音声を選択
        Console->>Evidence: build(audio_path)
        Evidence-->>Console: EvidenceBundle v1
        Console->>Runner: 既定: 実装済み全モジュールを選択
        Runner->>Range: 共有 LinguisticEvidence
        Runner->>Accuracy: 共有 EvidenceBundle
        Runner->>Coherence: 共有 EvidenceBundle
        Runner-->>Console: 3つの事実パケット
        Console->>Judge: 基礎事実 + 3パケット
    else API 実行
        User->>API: 評価リクエスト
        API->>Evidence: build(audio_path)
        Evidence-->>API: EvidenceBundle v1
        API->>Runner: 既定: 実装済み全モジュールを選択
        Runner-->>API: 選択済み事実パケット
        API->>Judge: 基礎事実 + 選択パケット
    end
    Judge-->>Delib: CEFR推定
```
