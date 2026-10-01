# 音響時刻を実測する手順

この検証は通常のAPI・ターミナル・5モジュール・Judgeを変更しない独立実験です。モデルの実行成功は、音響時刻の正確さを保証しません。

## 1. 比較候補を保存

リポジトリのディレクトリで実行します。保存先は新規ディレクトリだけを指定できます。

```sh
.venv/bin/python -m jgrade_eval.alignment_benchmark_cli prepare \
  audio/indonesian_japanese_1min.mp3 \
  audio/vietnamese_japanese_1min.mp3 \
  audio/burmese_japanese_1min.mp3 \
  --output-dir outputs/alignment-benchmark-new
```

同じ音声・文字起こし・モデルの推論結果から、従来のCTC argmax時刻と新しいCTC Viterbi時刻を保存します。モデル改訂番号は取得できた場合だけ記録し、取得できなければnullです。音声のSHA-256とライブラリ版も記録します。

## 2. 人が音声を聞いて正解を入力

`manifest.json`で各音声と`references/001.json`などの対応を確認します。`predictions/`は機械の予測なので編集しません。

- まず、ひらがな文字起こしが実際の発話に合っているか確認します。`transcript_review`を`status: reviewed`、`source: human`、`reviewed_by: 確認者名`にします。
- 各対象区間の`start_sec`と`end_sec`に、音声を聞いて確認した開始・終了時刻を秒で入力します。`review_status: reviewed`、`review_source: human`、`reviewed_by: 確認者名`を設定します。
- `suggested_start_sec/end_sec`は機械候補です。その値を聞かずに正解へコピーしないでください。候補は0.5秒以上のCTC空白で分割しただけで、設問区切り・話者交替を検出したものではありません。
- `start_offset/end_offset`は文字列上の0始まり・終端を含まない範囲です。区間を追加・修正する際は文字と音声の対応を確認し、重複区間を作らないでください。確認不能な区間はpendingのまま残します。
- `recording_type`は確認して`monologue`、`dialogue`、`answer_only`などを記入します。未確認なら`unknown`。録音条件の注記は`condition_tags`に記入できます。

文字起こし自体を修正した場合、旧予測とは比較できず`incompatible_source`になります。修正前のASR条件と修正後の逐語条件は分ける必要があります。今回のprepareはASR条件専用です。人手逐語条件は既存の`alignment_experiment --transcript-file`で別実験できますが、2方式の比較セットへの取り込みは今後の拡張です。

初回は少数の明瞭な区間だけ確認しても構いません。ただし、偏りのない精度比較には不明瞭な区間・境界も含め、可能なら2人で確認します。候補区間だけの結果から、全モーラ境界・全話者・全録音形式の精度を主張しないでください。

### 試聴用クリップと確認表を作る

```sh
.venv/bin/python -m jgrade_eval.alignment_review \
  outputs/alignment-benchmark-new --output-dir outputs/alignment-review-new
```

新しいフォルダに前後1秒の余白付きWAVと`review.csv`を保存します。`--padding-sec 2`などで余白を変更できます。元音声・予測・正解は変更しません。既存の出力先は上書きしません。

クリップ内の再生位置は0秒からなので、元音声の正解時刻は**クリップ内の確認時刻 + clip_origin_sec**です。CSVの`gold_start_sec/end_sec`へ記入するときも元音声上の秒数を使います。切り出しは機械候補に依存するので、境界がクリップ外の場合、ASRが違う場合、複数話者の重複などがある場合は必ず元音声を聞いて確認してください。試聴WAVは16kHz・mono・PCM16の補助音声で、元音声の代替の正解ではありません。

**CSVはメモ用で、自動取り込み・自動承認はしません。** 確認結果は前述の`references/*.json`へ手動で転記し、文字起こしと区間のhuman/reviewed/確認者を設定してください。CSVの文字セルには数式実行防止のため先頭のアポストロフィが付く場合があります。正確なsample/segment IDは試聴フォルダのmanifestに保存しています。

## 3. 誤差を測定

```sh
.venv/bin/python -m jgrade_eval.alignment_benchmark_cli evaluate \
  outputs/alignment-benchmark-new --report outputs/alignment-benchmark-new/report.json
```

既存のreportは上書きしません。注釈を直して再測定するときは別のreport名を指定します。

開始・終了・合計のMAE、中央値、p95、符号付き平均誤差、対応率、1秒超の外れ件数、50/100/200/500ms以内の割合を方式別・録音形式別に出します。「対応できた境界だけの割合」と「人手確認済みの全境界に対する割合」を併記するので、未対応区間を隠して精度を高く見せません。pendingの区間は未検証として残り、ゼロ誤差扱いにはなりません。

人手正解がない状態では誤差はnullです。許容誤差の採用基準はまだ決めておらず、まず実測結果を集める段階です。

## 人手確認の優先順位を別モデルで探す

```sh
.venv/bin/python -m jgrade_eval.alignment_cross_model \
  outputs/alignment-benchmark-new --output-dir outputs/alignment-cross-model-new
```

保存済みASRのひらがなを同じ元音声に対して別モデルで再整列し、`report.json`の`review_priority`へ未対応区間・時刻差の大きい区間から並べます。差は「比較側 - 既存側」で、正解への誤差ではありません。`accuracy_verified`と`gold_modified`はfalseです。モデル間の一致を正解として承認する処理はありません。

WhisperXの日本語既定モデルの重みを固定版で使いますが、WhisperX本体は導入しません。最初は約1.3GBの重み取得が必要です。音声はローカルで推論し、有料API・Judgeは使いません。両モデルは同じWav2Vec2系で、ひらがなASRにも依存します。特に比較側は漢字混じり表記で学習しているので、不一致を既存方式の誤りと決めつけないでください。

既存datasetを変更せず、別フォルダに比較側の整列結果とレポートだけを保存します。人手正解による測定とは独立した補助検証です。[選定根拠と設計](alignment-cross-model-design.ja.md)も参照してください。
