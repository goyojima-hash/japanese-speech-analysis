# Current Fact-Module Flow

This compact view is the current implementation. Common evidence is created once, every implemented module outputs only facts, and only the Judge/deliberation layer evaluates. Both the terminal and API call the same module runner after common evidence and, by default, select all implemented modules.

```mermaid
flowchart TB
    User([利用者]) --> Terminal[interactive.py<br/>対話ターミナル]
    User --> API[api_service.py<br/>API]
    Terminal --> Evidence[evidence/pipeline.py<br/>共通 Evidence 層]
    API --> Evidence
    Evidence --> Bundle[(EvidenceBundle v1)]
    Bundle --> Fluency[Fluency]
    Bundle --> Runner[fact_modules.py<br/>共通 runner]
    Runner --> Range[Range]
    Runner --> Accuracy[Accuracy]
    Runner --> Coherence[Coherence]
    Fluency --> Packets[(客観データ)]
    Range --> Packets
    Accuracy --> Packets
    Coherence --> Packets
    Packets --> Judge[AI Judge] --> Delib[CEFR 協議] --> Output([表示 / API応答])
```
