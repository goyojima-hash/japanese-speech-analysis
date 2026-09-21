# Coherence／Interactionモジュール向けモデル調査

日付: 2026-09-14  
対象: 今後の Coherence と Interaction の事実専用モジュールに使えるモデル候補。ここではモデルの導入・採用はしていません。

## 結論

1. **Coherence v1 は新モデルなしで実装します。** 現在の共有Sudachi Evidenceから、接続表現、発話単位、単位長、繰り返しを事実として出します。
2. **Coherence v2 の第一調査候補は KWJA です。** 日本語の係り受け、照応、談話関係まで扱える一方、共有トークナイザーを置換せず、任意のバージョン付きプロバイダーとして統合します。
3. **Interaction の最初の試作候補はローカル pyannote Community-1 です。** 話者ターンを出力し、ダウンロード後はオフライン利用でき、ASR時刻に合わせやすい専用タイムラインがあります。
4. **ターンテイキング予測モデルはまだ導入しません。** 最初のInteractionパケットには、ターン時刻・重なり・話者別区間という事実で十分です。予測器には日本語学習者対話のラベルが必要で、事実と評価の境界も曖昧にします。
5. **今回のために既存の日本語Wav2Vec2 ASRを交換しません。** すでに文字起こし・時刻入力を出しているため、交換には別途の日本語学習者テストセットが必要です。

## 候補一覧

| 対象 | 候補 | 出力できる事実 | 良い点 | 注意点 | 推奨 |
| --- | --- | --- | --- | --- | --- |
| Coherence v1 | 現在のSudachi Evidence | 接続表現、品詞列、単位長、語彙反復 | 依存追加なし、共有トークンを保つ | 暗黙の談話・照応は分からない | 最初に採用 |
| Coherence v2 | KWJA | 係り受け、格構造、橋渡し照応、共参照、談話関係 | 日本語専用の統合解析器、MIT | 別解析スタック。予測は真実ではなくモデル観測 | 任意プロバイダーとして検証 |
| Coherence v2 | GiNZA `ja_ginza_electra` | 文節・係り受け・構文 | MIT、Sudachi近縁 | 完全な談話／結束解析器ではない | 軽い構文比較候補 |
| 意味的拡張 | `multilingual-e5-large-instruct` | 隣接単位の埋め込みとコサイン類似度 | 94言語、MIT、標準的な実行方法 | 0.6B、512トークン打切り、類似度はまとまりの点数ではない | 実験のみ |
| Interaction v1 | `pyannote/speaker-diarization-community-1` | 話者ターン、重なり、ターン間隔、来歴 | ローカル／オフライン、ASR時刻に合わせやすい | 初回HFトークン・利用条件、CC-BY-4.0、日本語学習者対話の検証が必要 | 最初の試作 |
| Interaction代替 | NVIDIA NeMo Sortformer / MSDD | 話者活動・話者ラベル・ターン | end-to-end またはVAD＋埋め込み＋MSDDを選べる | 大きなNeMo依存、GPU前提寄り | 最初ではなく比較検証 |

## 根拠

### Coherence

このプロジェクトは既にSudachiPyを持ち、共通Evidence層で利用しています。そのため、接続表現、ポーズ／文字起こし境界から作る発話単位、トークン数、反復を最初の事実パケットにするのに新モデルは不要です。

より高度な任意プロバイダーとして最も有力なのは、日本語基盤モデル解析器の [KWJA](https://github.com/ku-nlp/kwja) です。公式説明とACL論文では、形態素解析、係り受け、格構造、橋渡し照応、共参照、談話関係を一つの解析器で扱うとされています。MITライセンスです。一方、関連する [cohesion analyzer](https://github.com/nobu-g/cohesion-analysis) は公開済みのbase／large checkpointを持ちますが、Juman++／KNPまたはKWJAを必要とする重い前処理があります。よって、どちらも `EvidenceBundle` のトークン化を置き換えず、別バージョンの `coherence_model_observations` として扱うべきです。[KWJA](https://github.com/ku-nlp/kwja)、[cohesion analyzer](https://github.com/nobu-g/cohesion-analysis)、[ACL論文](https://aclanthology.org/2023.acl-demo.52)

[GiNZA](https://github.com/megagonlabs/ginza) はMITライセンスの日本語Universal Dependencies解析器です。`ja_ginza_electra` もあり、構文だけを軽く拡張する比較候補になります。ただしKWJAほど談話・結束を扱いません。

`multilingual-e5-large-instruct` は意味的連続性向けの任意プロバイダーです。94言語、MITライセンス、Transformers／SentenceTransformersで実行可能、0.6Bパラメータで、長い入力は512トークンで打ち切られます。隣接単位のベクトルとコサイン類似度を出せますが、値を「まとまっている」と評価するのはモジュールの役割ではありません。[モデルカード](https://huggingface.co/intfloat/multilingual-e5-large-instruct)、[技術報告](https://arxiv.org/abs/2402.05672)

### Interaction

現在の一人音声とSilero VADではポーズは測れても、誰がいつ話したかは分かりません。Interactionには複数話者の入力契約と、応答ラグ／ターン交替の前に話者分離が必要です。

`pyannote/speaker-diarization-community-1` はモノラル16kHzを受け、必要時に再サンプリングし、ローカル実行できます。ダウンロード後はディスクからオフライン利用でき、ASR時刻に合わせるためのexclusive diarizationを返します。初回にHFの利用条件への同意とトークンが必要で、ライセンスはCC-BY-4.0です。公開ベンチマークには中国語会議音声が含まれますが、日本語学習者対話での保証は確認できませんでした。必ず自前検証が必要です。[モデルカード](https://huggingface.co/pyannote/speaker-diarization-community-1)、[多言語話者分離ベンチマーク](https://arxiv.org/abs/2509.26177)

NeMoは、TransformerベースのSortformerと、VAD＋TitaNet埋め込み＋MSDDという二系統を提供します。CUDA環境では有力な比較候補ですが、現在のmacOS中心プロジェクトには依存が重く、最初に選ぶ根拠はありません。[NeMoモデルガイド](https://docs.nvidia.com/nemo-framework/user-guide/25.07/nemotoolkit/asr/speaker_diarization/models.html)

### 現在のASRとAccuracy

現在の `vumichien/wav2vec2-large-xlsr-japanese-hiragana` は、日本語Common Voiceでのfine-tune、Apache-2.0、リポジトリ約2.52GBと明記されています。Accuracyで校正済みの情報を出すなら、まず同じCTC経路を日本語学習者データで検証してからにします。confidenceらしい数値のためだけに第二ASRを追加しません。[モデルカード](https://huggingface.co/vumichien/wav2vec2-large-xlsr-japanese-hiragana)

## 採用前の検証

1. 手動文字起こし、話者ターン、重なり境界を持つ日本語学習者の保持テストセットを作り、きれいな音声・雑音・重なりを含めます。
2. CoherenceはSudachiの決定的な事実とKWJA／GiNZAの観測を比較し、出力来歴、実行時間、メモリ、失敗例を記録します。最初からCEFRラベルで比較しません。
3. Interactionは応答ラグを計算する前に、話者分離誤り率、発話見落とし、話者混同を測ります。2025年の多言語研究でも、見落としと話者混同が主な誤り源でした。
4. 全プロバイダー出力に、モデルID、revision、device、入力schema、未提供能力を入れます。
5. 事実抽出が安定した後だけ、その事実パケットを後段Judgeに渡してレベル評価します。
