# 別モデルによる時刻候補比較: 設計

## 選定根拠と限界

[WhisperX公式alignment実装](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py)では日本語の既定モデルに[jonatasgrosmanモデル](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese)を使う。公式カードは日本語Common Voice/CSS10/JSUTでの学習、16kHz入力、Apache-2.0を記載する。今回採用するのは比較側の重みだけであり、WhisperXのアルゴリズム・精度を再現するものではない。

固定改訂は`cf031e020336460d15a417eba710bbc5bb43be9a`。語彙2341文字中ひらがな74文字を持ち、今回の保存済みASRの全文字を扱えることを確認。ただし漢字混じり表記で学習しているモデルへ全ひらがなを強制整列するため、文字が辞書にあることは発話への適合性を保証しない。元モデルと同じWav2Vec2系で学習データも重なり得るため、統計的独立な正解との比較ではない。ひらがなASR誤り・表記の違い・モデル差・整列アルゴリズムを不一致だけで区別できない。

## 実装

`alignment_cross_model.py`に独立CLIを追加。prepare済みbenchmarkを読取り、同じ音声を16kHz monoで30秒ごとに推論する。長さ不足の極短末尾は明示的に未対応にして、偽時刻を作らない。モデルは1回だけロードし、固定revision・`trust_remote_code=False`・`weights_only=True`で読む。追加の有料API・評価LLMは使わない。

既存CTC Viterbi整列で全ひらがなを音声全体へ整列。保存済みの候補区間の文字範囲を、既存と別モデルの時刻単位へ対応させて差を出す。結果のキーは`start_delta_ms/end_delta_ms`であり、人手誤差を示す`error_ms`とは別。区間の最大差順に一覧化し、どちらが正しいかは決定しない。文字対応不能の区間は理由付きのabstentionとする。

同一テキストという条件での比較で、異なるASR書き起こし同士の意味対応は行わない。これにより元ASRを共有したバイアスも残る。推論結果とレポートのみ新規outputへ保存する。gold・既存予測は変更せず、通常経路との接続もしない。

## テスト

既知の差、文字欠落、無効単位、音声・文字列違い、モデル不足、上書き禁止、推論1回ロードと30秒チャンクの来歴を検証。追加実装は80%以上のカバレッジ、既存全テスト、3実音声の保存前後ハッシュ確認を実施する。
