# Interactionモジュール要件（提案）

## 目的

CEFRのInteraction（やりとり）に関して、連続した一人語り、面接形式の解答、複数話者の対話を同じ入口で受け、**LLMなし**で監査可能な観測事実を返す `InteractionModule` を設計する。モジュールはCEFR/JFSレベル、良否、正誤、課題達成度を判定しない。判定は、確定済みの設問・回答対応、共通Evidence、5軸のFact Packetを後段のAI Judgeへ渡した後にのみ行う。

## 固定方針

- 共通層と5モジュールはLLMを使わない。意味埋め込みによる話題推定、LLMによる要約・設問対応付けはv1に入れない。
- AI Judgeには生音声を渡さない。設問テキスト、確定した回答区間の文字起こし、共通Evidence、Fact Packetだけを渡す。
- 録音モード（`monologue`、`interview_dialogue`、`multi_party_dialogue`、`unknown`）と検出話者数を必ず出力する。
- 連続録音の区間分割・設問対応は候補であり、人が確定するまで**設問ごとのタスク評価**には使わない。学習者発話を特定できる場合は、設問なし・区間未確定でも録音全体の暫定CEFR推定に使える。
- Interactionは他のFact Moduleを呼ばず、共通Evidenceと共通の課題文脈だけを読む。Range、Accuracy、Coherence、Fluencyの出力は入力にしない。

## 利用形態

| 録音形態 | 必須または推奨の入力 | Interactionが出せる観測 |
| --- | --- | --- |
| 解答だけの一人語り | 設問集合、可能なら設問順 | 回答候補区間、無音・文末に基づく境界根拠、対話的表現候補 |
| 面接対話 | 設問集合、学習者話者の指定または確定 | 話者ターン、質問から回答までのラグ、重なり、回答候補区間、対話表現候補 |
| 複数話者対話 | 学習者話者の指定、必要なら設問集合 | 話者ターン、交替、重なり、相づち・確認・聞き返し候補 |

話者分離または学習者話者の特定ができない録音も失敗にしない。その場合は録音モードと未提供能力を明記し、学習者側だけと断定する指標は空にする。

## 入力契約

既存の `EvidenceBundle v1` は後方互換のため変更しない。新しい共通エンベロープ `AssessmentContext v1` を、Evidenceと一緒に入口からRunnerへ渡す。

```text
AssessmentContext v1
├── recording
│   ├── recording_mode: monologue | interview_dialogue | multi_party_dialogue | unknown
│   ├── mode_source: user_declared | detected | unknown
│   ├── learner_speaker_id: optional
│   └── prompt_audio_expected: boolean | unknown
├── prompt_set
│   └── prompts[]: { prompt_id, text, display_order }
├── transcript_spans[]
│   └── { span_id, start_sec, end_sec, transcript_text, speaker_id?, provenance }
├── candidate_answer_segments[]
│   └── { segment_id, start_sec, end_sec, span_ids, candidate_prompt_id?,
│         mapping_status, boundary_evidence[] }
└── confirmed_answer_segments[]
    └── { segment_id, prompt_id, learner_speaker_id?, confirmed_by, confirmed_at }
```

`prompt_set` は利用者が入力する。設問を音声内容から生成・推測しない。既存の単一 `roleplay_task` は、互換のため `prompt_set.prompts[0]` へ変換できる。

### 共通層で追加する事実

現在のEvidenceには時刻付き文字起こしチャンク・話者ターンがない。連続録音を正しく分けるため、次を共通処理として一度だけ作る。

1. **TranscriptSpan**: 時刻、文字列、文字起こし来歴を持つ最小の発話チャンク。
2. **SpeakerTurn（任意）**: チャンネルラベルまたは非LLM話者分離providerがある場合だけ作る。provider名・版・信頼度を残す。
3. **CandidateAnswerSegment**: VADの無音、話者交替、保守的な文末パターン、設問の登録順だけから作る候補区間。意味類似度は使わない。

既存 `EvidenceBundle v1` を上書きせず、`AssessmentEvidence v2 = { EvidenceBundle v1, AssessmentContext v1 }` として拡張する。既存の1音声評価は従来どおり `EvidenceBundle v1` だけで動く。

## Interaction Fact Packet

```text
InteractionFactPacket
├── module_id: "interaction"
├── input_provenance
├── recording_observation
│   ├── recording_mode, mode_source, detected_speaker_count
│   └── learner_speaker_id?, prompt_audio_present?
├── prompt_observations[]                 # 入力済み設問のIDと順序。品質判断なし
├── candidate_answer_segments[]            # 境界根拠、対応候補、確定状態
├── confirmed_answer_segments[]            # 人が確定した対応だけ
├── turn_observations[]                    # speaker_id、開始・終了、話者交替の事実
├── response_transition_observations[]     # 設問ターン→学習者ターンのラグ／重なり
├── overlap_observations[]                 # 話者間の同時発話区間
├── interaction_marker_candidates[]        # 確認・聞き返し等の辞書一致と位置
└── unavailable_capabilities[]
```

`interaction_marker_candidates` は、確認、聞き返し、応答開始、相づち、締めなどの版付きローカル辞書との**一致候補**である。意図、適切さ、理解、会話の成功を表さない。

パケットには `score`、`rating`、`cefr`、`jfs`、`quality`、`correct`、`successful_interaction` を含めない。

## 候補区間の決定的ルール

### 対話録音

1. 学習者話者が確定している場合、その話者の連続ターンを回答候補にする。
2. 直前に別話者のターンがある場合、交替時刻、無音ラグ、重なりを記録する。
3. 設問との対応は、明示的な人の選択を最優先する。未確定時は登録順に `candidate` とできるが、Judge入力には含めない。

### 解答だけの連続録音

1. VADの無音閾値と、保守的な文字起こし文末パターンをそれぞれ境界根拠として記録する。
2. 二つ以上の根拠がある場所だけを強い候補境界にする。一つしかない場合は弱い候補として残す。
3. 設問順との対応は候補提示だけに使う。設問を飛ばした、順番を替えた、複数設問に一続きで答えたケースを自動確定しない。

## Judgeへ渡す条件

`mapping_status = confirmed` の回答は、設問テキスト、回答の `TranscriptSpan`、該当する共通Evidenceのビュー、5軸のFact PacketをAI Judgeへ渡し、**設問単位のCEFR推定とタスク達成度**を出せる。

設問なし・区間未確定でも、録音が一人語りである、または対話録音で学習者話者が確定しているなら、学習者発話全体のEvidence Viewと5軸のFact PacketをAI Judgeへ渡せる。この結果は `evaluation_scope = recording_level`、`task_rating = unavailable`、`risk_flags = [task_context_missing, answer_mapping_unconfirmed]` と明記する。設問対応の良し悪し・個別設問の達成度は判定しない。

複数話者録音で学習者話者を特定できない場合だけ、客観データを返してJudgeを実行しない。誰のRange／Accuracy／Fluencyなのかを安全に特定できないためである。

対話で全話者の音声があっても、AI Judgeへは生音声を渡さない。Interaction Packetのターン・ラグ・重なり・辞書一致候補を渡す。

## 受け入れ条件

1. 既存の単一音声・単一 `roleplay_task` のAPI、CLI、Consoleは動作と既存4軸の直列化データを維持する。
2. 設問集合は空・1件・複数件を受け付け、`prompt_id` は一意である。
3. 自動候補だけではJudgeを呼ばない。確定済み対応だけが評価要求になる。
4. 一人語りでは `recording_mode=monologue`、話者数1を明記し、ターン交替指標を作らない。
5. 対話で話者区別が利用可能な場合だけ、話者別ターン、ラグ、重なりを出す。
6. 話者分離・時刻付きチャンク・設問テキストがない場合、例外ではなく名前付きの `unavailable_capabilities` を返す。
7. 同じ音声、同じContext、同じprovider版では候補とFact Packetは決定的である。
8. 各モジュールは他モジュールを呼ばず、共通Evidence／Contextを一度だけ読む。
