# 文字起こし・音声時刻対応の設計

## 調査と判断

現行の `FluencyExtractor` は、CTCのargmaxラベルからモーラ時刻を抽出する。これは録音全体の文字起こしを与えて音響的に再整列する強制アラインメントではない。末尾ラベルが時刻列に入らない場合もあるため、ラベル列と文字起こしを照合せずに文字位置を割り当てるのは危険である。[PyTorchの強制アラインメント解説](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html)は音響フレームと既知Transcriptの対応を別処理として説明し、当該TorchAudio APIの廃止予定も記載する。[WhisperX](https://github.com/m-bain/whisperX)は日本語モデルを用意する一方、未登録文字、重なり発話、話者分離の限界を挙げる。[pyannoteの話者設定資料](https://docs.pyannote.ai/tutorials/speaker-configuration)も重なり発話を区別して扱う。これらの性能をJ-GRADEの学習者音声で検証していないため、今回は導入しない。

## データ契約

共通Evidenceの派生データ `transcript-alignment.v1` は、`status=complete|partial|unavailable`、`method=ctc_argmax_exact_prefix`、`aligned_chars`、`transcript_chars`、`reason`、単位ごとの文字範囲・時刻範囲・ラベルを持つ。`candidate` は時刻の確定やASR正解を意味しない。`evidence.v1` 本体の保存形式は変更せず、必要時に `SpeechEvidence` から決定的に再生成する。APIで客観データの返却を指定した場合は、既存の `objective_data` の形を変えず、同階層の `transcript_alignment` に出す。客観データの返却を無効にした場合は、この派生データも返さない。

照合は先頭から逐次行い、各ラベルが `raw_transcript_hiragana` の現在位置に完全一致するときだけ文字範囲を進める。時刻は有限・正・単調で録音長内でなければならない。途中でラベルが食い違う場合、後続を推測せず `unavailable` にする。ラベル列がTranscriptの真の接頭辞なら `partial` として、その接頭辞だけを公開する。空や欠測は `unavailable`。

Interactionは既存のVADポーズ由来の時刻候補を維持し、その区間内に完全に含まれる連続した対応単位だけから `start_offset/end_offset/transcript_text` の候補を追加する。`confirmed_by` は付けない。設問への割当ても行わない。`TaskContext` の確認済み回答条件は変更せず、候補だけでAI観測を実行しない。

## 導入境界

変更対象は共通層の派生関数、公開APIの追加フィールド、Interaction packetの候補補強だけ。追加候補はJudge入力に転送せず、従来のJudge契約とRange、Accuracy、Coherence、Fluencyの既存フィールドを変えない。新しいアラインメントモデルや音声直接AI入力は別実験として扱う。
