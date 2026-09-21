# Coherenceモジュール要件

## 目的

CEFRの話し言葉の質的側面であるCoherence（まとまり）に対し、独立選択できる事実専用モジュールを追加します。既存の `EvidenceBundle v1` を読み、`CoherenceFactPacket` を出力します。まとまりの点数、CEFR/JFレベル、良し悪し、訂正、LLM結論は一切出しません。

## v1の範囲

- 共通の文字起こし、Sudachiトークン、文字オフセット、ポーズ区間を再利用します。ASR、VAD、Sudachi、Range、Fluency、Accuracyを再実行しません。
- バージョン付きローカル辞書から、明示的な接続表現を観測します。トークン位置、表層／見出し語、`causal`、`contrast`、`additive`、`sequence`、`conclusion`などの説明カテゴリを記録します。
- 決定的な**候補**談話単位を出します。候補境界は接続表現または終止／丁寧表現パターンから作り、導出理由を必ず記録します。文の正しさの判定ではありません。
- 候補単位をまたぐ語彙反復を、見出し語、単位番号、距離として出します。
- ポーズ区間は独立した時間観測としてコピーします。Evidence v1には検証済みのトークン時刻対応がないため、v1ではポーズを特定トークン境界に結び付けません。
- 係り受け、共参照、暗黙の談話関係、意味連続性埋め込み、トークンとポーズの対応は、未提供能力として明示します。
- Coherenceは明示選択時だけ動きます。既存APIのデフォルト（`fluency`、`range`）と、Fluency／Range／Accuracyの出力契約は変えません。

## 非目標

- v1ではLLM、KWJA、GiNZA、埋め込みモデル、意味類似度、文法検査、まとまりの点数、CEFR対応、`coherent`／`incoherent`ラベルを使いません。
- `EvidenceBundle v1` にフィールドを足さず、schemaを上げず、二重トークン化しません。
- `range_data` や `accuracy_data` は使いません。Coherenceは共通Evidenceだけに直接依存します。
- ASRの境界、接続表現、反復、ポーズを学習者の誤りと暗に扱いません。
- 最初の実装で既存の対話コンソールのデフォルトを変えません。コンソールでのモジュール選択UIは別途テストする統合作業です。

## 入力

| 入力 | 必須 | 用途 |
| --- | --- | --- |
| `EvidenceBundle.speech.raw_transcript_hiragana` | はい | 候補単位の文字列範囲と来歴 |
| `EvidenceBundle.speech.pause_segments` | はい | 独立した時間ポーズ観測 |
| `EvidenceBundle.linguistic.tokens` | はい | 接続表現、終止パターン、反復の観測 |
| `EvidenceBundle.linguistic` のTokenizer情報 | はい | 再現性 |

## 出力契約

`CoherenceFactPacket` には次だけを含めます。

- `module_id = "coherence"`
- `input_provenance`: Evidence schema、STTモデル、Tokenizer版／split mode、単位化ポリシー版、接続表現辞書版
- `connective_observations`: 観測トークンとカテゴリ
- `candidate_units`: 文字／トークン範囲と境界導出理由
- `repetition_observations`: 反復見出し語と単位位置
- `pause_observations`: コピーした時間ポーズ区間
- `unavailable_capabilities`: 名前付きの制限

出力キーに `score`、`rating`、`cefr`、`jfs`、`correct`、`error`、`quality`、判定booleanを含めません。

## 受け入れ条件

1. デフォルトAPIのFluency／Range事実はバイト単位で互換のままであり、デフォルトfact moduleは `fluency` と `range` のままです。
2. `selected_modules=["coherence"]` はCoherenceだけを呼び、Range／Accuracyを呼びません。
3. `selected_modules=["range", "accuracy", "coherence"]` で共通層は一度だけトークン化し、3モジュールは同じ `LinguisticEvidence` を読みます。
4. Coherenceは `EvidenceBundle`、`audio_pipeline.py`、`range.py`、`accuracy.py` を変更しません。
5. 空文字起こし、空トークン、ポーズなし、未知接続表現、不完全な時刻情報では、評価や例外ではなく空／未提供の事実を返します。
6. 同じEvidenceBundleとプロバイダー版なら、出力は決定的です。
7. JudgeプロンプトはCoherenceを観測事実として説明し、候補境界・未提供能力を学習者誤りと扱うことを禁止します。
