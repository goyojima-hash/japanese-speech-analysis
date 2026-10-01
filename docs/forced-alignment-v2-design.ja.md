# 音響時刻再整列 v2: 設計

## 判断

第一候補は既存の `vumichien/wav2vec2-large-xlsr-japanese-hiragana` のCTC出力を使う再整列である。既存のひらがな文字起こしと語彙体系に合い、新しい音響モデルを必要としない。ただし同じモデル自身の文字起こしを再整列しても時刻が必ず改善するわけではないため、実測前に本番経路へ入れない。比較対象は日本語アラインメントを持つWhisperXと日本語MFA。MMS系の非商用重みはAWS運用候補にしない。出典は [CTC-segmentation](https://github.com/lumaku/ctc-segmentation)、[WhisperX](https://github.com/m-bain/whisperX)、[日本語MFA](https://huggingface.co/MontrealCorpusTools/japanese_mfa)、[MMSモデルのライセンス](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)。MFAの重みがCC BY 4.0でも学習データには別条件のものがあるため、商用AWS利用前に利用条件を別途確認する。

## 現行コードでの境界

`fluency.transcribe()` は30秒ごとの音声とCTC logitsを返す。`FluencyExtractor.extract()` はそのlogitsから既存の `mora_timings` を作り、共通Evidenceへ渡すが、logitsは保存しない。`get_mora_timings()` は固定0.02秒のフレーム幅と出力フレーム数によるチャンクオフセットを使い、末尾の非blankラベルを確定しない場合がある。この時刻列を「正解」として再整列へ入力しない。旧Fluencyの値を修正するなら別の互換性検証を要する。

## 段階1: 隔離された比較実験

本番コードへ直接依存を追加せず、オフラインの実験ランナーで同一音声・同一文字起こしを各方式に入力する。第一候補はチャンクごとの生logits（softmax前）、正しいblank ID、トークナイザ、音声サンプル数とlogitフレーム数を使う。各チャンクの絶対開始時刻は `chunk_start_sample / sample_rate`、フレーム幅は `chunk_sample_count / logit_frame_count / sample_rate` から算出し、固定0.02秒の累積誤差を持ち込まない。チャンク境界をまたぐ文字は無理に配分せず、前後の短い音声重複を使うか `unavailable` にする。重複の扱いと再結合規則を実験条件として固定する。

ASR文字起こしと、人手修正した逐語文字起こしの両方を別条件として試す。読みへの正規化、句読点、言い直し、フィラー、長音、小書き文字は文字オフセット写像を維持する。未知文字や音声にない単語を削除して「完全一致」に見せない。音声のVAD区間と照合し、長い無音・重なり発話・設問音声混入は保留理由を分ける。

最初の実装は `jgrade_eval/ctc_forced_alignment.py` のCTC Viterbi経路と `jgrade_eval/alignment_experiment.py` の独立コマンドである。`numpy` と既存Wav2Vec2だけで動き、新しい音響モデルやCTCライブラリを通常依存に加えない。出力は実験用JSONのみ。文字起こしの誤りを検出できると主張せず、音響的な確信度と人手境界誤差の測定は後続タスクとする。

## 出力契約（実験用、未公開）

最初の実装の `forced-alignment.v2` は `status=complete|unavailable`、`reason`、`audio_sha256`、`transcript_sha256`、`model_id`、`algorithm`、`sample_rate`、`units` を持つ。各単位は元のひらがな文字列上の半開区間 `[start_offset,end_offset)` と `[start_sec,end_sec]` を持つ。`complete` は指定文字列についてCTC経路が見つかった意味に限り、発話内容や時刻が正しい証明ではない。生のスコアは確率と誤認されないよう初期実装では公開しない。モデルrevisionと正規化versionを固定した永続キャッシュはまだ作らない。録音長外、逆順、失敗した照合は黙って補間しない。

## 検証と採用ゲート

学習者の一人語り、複数設問、対話、言い直し、長い沈黙、重なり発話を含む同一評価セットを用意する。境界を人手注釈し、一部は二重注釈して人間間の揺れを把握する。開始・終了それぞれの絶対誤差の中央値/95パーセンタイル、設定した許容幅内の割合、完全・部分・保留率、1秒超の大外れ率を、録音形式と音声条件別に報告する。速度とメモリはMac CPU/MPSとAWS候補環境で別計測する。公開ベンチマークのASR CERを境界精度の代用にしない。

比較の結果、用途別の事前基準を満たし、旧方式より改善し、失敗時に安全に保留できる場合だけ、共通Evidenceの**派生サイドカー**としてオプトイン導入する。`evidence.v1` と `transcript-alignment.v1` は維持する。Interactionへの入力切替も別のオプトインとし、候補だけを添え、`confirmed_by` や設問割当てを生成しない。Judge入力へ新フィールドを渡さない。キャッシュ、API、ターミナルの互換テストと5モジュールの結果比較を通してから既定化を判断する。

## 未確定事項

- 回答区間に必要な時刻許容幅と、境界注釈の粒度。
- 既存モデルのCTC出力に対する再整列アルゴリズムのMac/AWS依存性。
- WhisperXの日本語モデルと、J-GRADEのひらがな文字起こしの正規化互換性。
- 日本語MFAの運用ライセンス確認と、学習者発話での境界精度。
