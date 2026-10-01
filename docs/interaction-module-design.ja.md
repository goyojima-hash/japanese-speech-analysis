# Interactionモジュール設計（提案）

## 結論

Interactionは、連続録音を「対話らしく評価する」モジュールではなく、録音モード、話者ターン、回答候補区間、設問対応の確定状態、交替時の時刻事実、表現辞書の一致候補を出す独立Fact Moduleとして実装する。これにより、解答だけの一人語りにもデータを出しつつ、相手が録音されていないのにターン交替を作ることを防ぐ。

## 境界

```text
入口（API / Console / CLI）
  → AssessmentContextを受理・永続化
  → 共通Evidenceを1回だけ構築
  → 共通の録音分割処理で候補を作る
  → 人が候補区間と設問の対応を確定（設問単位評価をする場合）
  → 5つの独立Fact Moduleを同じEvidence Viewへ実行
  → 確定済み回答なら設問単位、未確定なら録音単位でAI Judgeへ渡す
```

`AssessmentContext` は「入力された事実・候補・確認結果」を持つ。`EvidenceBundle` は「音声から抽出した事実」を持つ。二つを分けることで、設問テキストの登録やユーザー修正によって、音声解析キャッシュを無効化しない。

## 新規コンポーネント

| ファイル | 責務 | LLM |
| --- | --- | --- |
| `jgrade_eval/assessment_context.py` | 設問集合、録音モード、話者役割、候補・確定済み回答の検証と直列化 | 使わない |
| `jgrade_eval/recording_segments.py` | `TranscriptSpan`、任意の話者ターン、候補回答区間を作る | 使わない |
| `jgrade_eval/interaction.py` | `InteractionFactPacket` を共通Evidence／Contextから収集する | 使わない |
| `jgrade_eval/evidence/views.py` | 確定済み回答区間を各Fact Moduleへ渡す読み取り専用Viewを作る | 使わない |
| `jgrade_eval/fact_modules.py` | Interactionを含む5モジュールの選択・実行を統一する | 使わない |

音声チャネルの話者ラベルがない場合に使う話者分離providerは、`recording_segments.py` の任意依存にする。これはLLMではなく前処理providerであり、未設定・失敗時には単に話者別能力を `unavailable` とする。Interaction本体はproviderを直接呼ばない。

## データフロー

### 1. 設問と録音の受付

API multipart／JSON、Console、CLI manifestは同じ `AssessmentContext` JSONを受ける。設問は複数登録でき、各設問は安定した `prompt_id` と表示順を持つ。1件だけの既存 `roleplay_task` は互換変換する。

### 2. 共通Evidenceと録音分割

`EvidencePipeline.build(audio_path)` は既存どおり1回だけ実行する。拡張の`RecordingSegmenter`は、VAD区間、追加する時刻付き文字起こしチャンク、任意の話者ターンを読み、候補を作る。

話題類似度やLLMの「この答えはQ2」という推論は使わない。候補の根拠はすべて`boundary_evidence`に残す。

```text
dialogue_turn_window
  = 学習者ターンの連続範囲 + 直前の別話者ターンID

monologue_pause_terminal
  = VAD無音 + 保守的文末パターン + TranscriptSpan境界

ordered_prompt_candidate
  = 利用者登録済みの表示順だけによる暫定対応
```

### 3. 人による確定

候補はUIまたはAPIの更新要求で、結合・分割・設問の変更・「設問なし」に変更できる。`confirmed_by`、確定時刻、元候補IDを記録する。自動候補を保存することと、**設問単位の評価を可能にすること**は別である。

### 4. 各モジュールの実行

確定済み回答があれば、同じ`AssessmentEvidence`から読み取り専用の回答Viewを作る。未確定でも、一人語りまたは学習者話者が確定した対話なら、学習者の**録音全体View**を作る。Fluency、Range、Accuracy、Coherence、Interactionは同じViewを独立に読む。各パケットの作成順や並列実行は、出力の意味を変えない。

範囲・正確さ・まとまりは学習者の確定済み回答Viewだけを読む。Interactionは同じ回答Viewに加え、録音全体のターン・回答遷移観測を読む。これにより、試験官の語彙を学習者のRangeに混ぜない。

### 5. Judge入力

オーケストレータは、二種類の評価要求を作る。

| 条件 | `evaluation_scope` | Judgeに渡すもの | 出力の制約 |
| --- | --- | --- | --- |
| 設問・回答区間が確定 | `task_response` | 設問、回答文字起こし、5軸のFact Packet | CEFR推定とタスク達成度を出せる |
| 設問なし／区間未確定だが学習者発話を特定可能 | `recording_level` | 学習者発話全体、5軸のFact Packet | 暫定CEFRのみ。`task_rating=unavailable` とリスクフラグを付ける |
| 複数話者で学習者を特定不能 | `objective_only` | なし | 客観データのみ。Judgeを実行しない |

いずれも生音声はJudgeへ渡さない。

## 失敗・不足時の扱い

| 状態 | Interactionの出力 | Judge |
| --- | --- | --- |
| 設問なし | `prompt_set_empty`、候補区間は出せる | 録音レベルの暫定CEFRのみ。設問達成度は評価しない |
| 一人語り | `recording_mode=monologue`、ターン交替能力は未提供 | 設問がなくても録音レベルの暫定CEFRを実行できる |
| 話者分離なしの対話録音 | `speaker_turns_unavailable` | 学習者話者が別途指定されない限り、Judgeを実行しない |
| 候補が未確定 | `mapping_status=candidate` | 録音レベルの暫定CEFRは実行するが、設問単位のタスク評価はしない |
| 話者役割未指定 | `learner_speaker_unknown` | 複数話者なら客観データのみ。一人語りなら録音レベル評価を実行できる |

## Judgeプロンプトの追加制約

`interaction_data` は観測事実であり、候補境界、辞書一致、話者分離providerの出力を学習者の誤り・会話成功・CEFRレベルの単独根拠にしてはいけない。`unavailable_capabilities`を不利な根拠にしてはいけない。設問・回答対応が確定済みであることを、Judge入力側で明記する。

## なぜこの構成か

- **説明可能性**: 各区間に境界・対応付け・来歴を残せる。
- **互換性**: `EvidenceBundle v1` と既存4軸の出力を変えずに段階導入できる。
- **独立性**: InteractionがCoherenceの候補単位やRangeの語彙統計に依存しない。
- **将来性**: 話者分離providerや時刻アラインメントを追加しても、Fact Packetの評価責務は変わらない。
- **安全性**: 相手のいない一人語りを「低いInteraction」と誤解させず、未取得能力を明示できる。
