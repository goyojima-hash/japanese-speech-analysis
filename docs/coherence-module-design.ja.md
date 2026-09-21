# Coherenceモジュール設計

## 決定

`jgrade_eval/coherence.py` を `EvidenceBundle v1` 上の純粋かつ独立した収集器として実装します。`CoherenceFactPacket` を返し、共通前処理や既存モジュールのコードを変えません。v1では外部モデルを使わず、日本語学習者発話に対して監査可能な事実の基準線を先に作ります。

## 既存モジュールを守る理由

共通パイプラインは、高コストで横断的な処理を一度だけ実行しています。音声抽出、文字起こし、VAD／ポーズ、モーラ時刻、Sudachiトークン化です。Coherenceはこの不変の事実を読むだけです。`RangeExtractor`、`AccuracyModule`、`FluencyExtractor` を呼ばないので、既存アルゴリズム、出力形、依存バージョン、キャッシュキーを変えられません。

`EvidenceBundle v1` は変えません。将来のKWJA／GiNZA／埋め込みの出力は共通Evidenceではなく、Coherenceパケットに入る任意のモデル出力にします。これにより実験モデルが既存Evidenceキャッシュや出力契約を壊しません。

## データ型

```text
CoherenceFactPacket
├── module_id: "coherence"
├── input_provenance: map[str, str]
├── connective_observations: ConnectiveObservation[]
│   ├── token_index, surface, dictionary_form
│   ├── category: causal | contrast | additive | sequence | condition | conclusion | other
│   └── lexicon_version
├── candidate_units: CandidateUnit[]
│   ├── unit_index, token_start_index, token_end_index
│   ├── text_start_offset, text_end_offset
│   └── boundary_derivations: [connective:<category> | terminal_pattern:<id> | transcript_end]
├── repetition_observations: RepetitionObservation[]
│   ├── dictionary_form, unit_indexes, occurrence_count, unit_distances
├── pause_observations: TimedSpan[]
└── unavailable_capabilities: string[]
```

`CandidateUnit` は意図的に「候補」と呼びます。ASR文字起こしには信頼できる句読点がない場合が多く、ルール境界は検証済みの文境界ではありません。単位に点数・品質フィールドはありません。

## 決定的なv1アルゴリズム

### 1. 接続表現辞書

`CoherenceLexicon v1` は、トークンの見出し語／表層から説明カテゴリを引く小さな版付き辞書です。例は `だから`／`ので`（causal）、`しかし`／`けど`（contrast）、`そして`／`また`（additive）、`まず`／`次に`（sequence）です。適切さは判定せず、観測された表現だけを記録します。

### 2. 候補単位

トークン0から開始し、次の位置に境界を作ります。

1. 新しい関係を始める接続表現トークンの**直前**
2. 保守的な終止／丁寧表現パターンに一致するトークンの**直後**。ただし次トークン自体が接続表現境界なら除く
3. 文字起こしの終端

接続表現は、それが導入する候補単位に含めます。ソースのトークン・文字オフセットを保ちます。保守的パターンが見つからなければ、文字起こし全体の候補単位を一つだけ出します。文分割を捏造するより安全です。

### 3. 反復

内容語だけを対象に、複数の候補単位に出現する正規化見出し語をまとめます。出現位置と単位距離を記録します。同一単位内だけの反復、機能語だけの反復、多様性比、`redundant` のような判断は出しません。

### 4. ポーズ

`EvidenceBundle.speech.pause_segments` を時刻付きのままコピーします。v1では単位に紐付けません。信頼できる紐付けには、Evidence v1にない検証済みトークン時刻対応が必要です。

## モデルプロバイダー境界: 将来のv2

```python
class CoherenceObservationProvider(Protocol):
    provider_id: str
    version: str
    def observe(self, evidence: EvidenceBundle) -> CoherenceModelObservations: ...
```

KWJAは、日本語の係り受け、共参照、談話関係の予測を出せるため、第一の研究候補です。出力には `derivation="kwja"`、モデルrevision、実行来歴を付け、モデル由来の観測と明記します。決定的v1の事実を置き換えず、評価決定もしません。

## APIとJudge統合

`SUPPORTED_FACT_MODULES` に `"coherence"` を加えますが、`DEFAULT_FACT_MODULES` は変えません。選択時だけ `evaluate_speech_level` が `coherence_data` を `objective_data` と `roleplay_input` に入れます。未選択なら両方のキーは存在しません。Judgeプロンプトは候補単位、接続カテゴリ、反復、未提供能力が観測事実であり、単独で誤りやレベル根拠にしないことを明記します。

対話コンソールはAPIと同じfact-module runnerを使います。ターミナルの明示的な表示プロファイルは実装済みの全事実モジュールを選択し、`=== Coherence客観データ ===` を表示します。APIは応答互換性のため過去のデフォルト選択を維持します。

## 失敗時の挙動

| 状態 | パケットの挙動 |
| --- | --- |
| 空の文字起こし／トークン | 空コレクションと `no_linguistic_tokens` |
| 辞書に接続表現なし | 空の `connective_observations`。エラーにしない |
| 保守的境界なし | 文字起こし全体の候補単位を一つ |
| ポーズなし | 空のポーズコレクション。エラーにしない |
| オフセット欠落 | 該当文字範囲を省略し `token_offsets_unavailable` を追加 |
| 将来プロバイダー失敗 | 決定的v1の事実を保持し、プロバイダー固有の未提供能力を追加 |

## 回帰防止策

- Coherence選択前後で、Fluency／Range／Accuracyの直列化済み事実を固定fixtureで比較します。
- 計測Tokenizer fixtureで、全モジュール選択時に一回だけ共通トークン化されることを証明します。
- v1では `audio_pipeline.py`、`range.py`、`accuracy.py`、`evidence/` に本番差分がないことを確認します。
- モジュール隔離、デフォルト／非デフォルトAPI payload形をテストします。
- Coherence直列化の評価語を拒否し、provider／辞書版を厳密に検証します。
