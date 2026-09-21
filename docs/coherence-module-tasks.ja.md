# Coherenceモジュール実装計画

## Phase 0 — 回帰基準線とバグハンター事前分析

1. 決定的Evidence fixtureで既存Fluency、Range、Accuracyの事実パケット／API snapshotを固定します。
2. 実装前に、3モジュール選択時の二重トークン化と、未選択モジュールでデフォルトpayloadが変わる二つの失敗を再現します。
3. 5 Whysと特性要因図を日英の回帰レポートに残します。

### 事前失敗分析

| リスク | 早期検出 | 防止策 |
| --- | --- | --- |
| CoherenceがRangeのトークン化を変える | Tokenizer呼び出し数が1超 | 一つの共有`LinguisticEvidence` fixtureを注入し、同一値を検証 |
| デフォルトAPI契約が変わる | API snapshot差分 | `DEFAULT_FACT_MODULES` を変更しない |
| ルールが隠れた採点になる | 禁止keyテストで評価語が通る | 厳密なパケットschemaとprompt境界テスト |
| 句読点のないASRから文を捏造 | 候補単位に導出理由がない | 全境界に導出理由を必須にし、なければ一単位へfallback |
| 将来モデルで共通キャッシュが無効化 | Evidence schema/cache keyが変化 | モデル出力をEvidenceBundleの外に置く |

## Phase 1 — 要件とREDテスト

1. 接続カテゴリ、決定的候補単位fallback、反復、ポーズコピー、空入力、評価語禁止のテストを追加します。
2. Coherence単独、3テキストモジュール全選択、デフォルト互換性のAPI選択テストを追加します。
3. Range／Accuracyを呼ばないことと、共有トークン化一回の回帰テストを追加します。
4. 集中テストをREDで実行し、TDDチェックポイントをコミットします。

## Phase 2 — コア収集器

1. パケットdataclass、版付き辞書、決定的収集器を持つ `jgrade_eval/coherence.py` を追加します。
2. `EvidenceBundle v1` を維持し、既存モジュールファイルを変えません。
3. 集中テストをGREENで実行し、GREENチェックポイントをコミットします。

## Phase 3 — APIとJudge境界

1. APIモジュールregistryだけに明示的な `coherence` 選択を追加します。
2. 選択リクエストにだけ `coherence_data` を加え、Judgeプロンプト境界を更新します。
3. API／デフォルト回帰テストを再実行し、チェックポイントをコミットします。

## Phase 4 — 検証と引き渡し

1. 集中テスト、全テスト、コンパイル、`git diff --check` を実行します。
2. Coherence前コミットとの差分で、Fluency、Range、Accuracy、Evidenceへの予期せぬ変更がないか比較し、あればブロッカーにします。
3. バグハンターの5 Whys、特性要因図、タイムライン、日英レポート、再発防止を検証します。
4. この実装ではKWJA、GiNZA、埋め込み、pyannoteを導入せず、対話コンソールのデフォルトも変えません。

## 保留作業

- 教師レビュー済みの接続表現／境界辞書。
- ポーズを単位に紐付けるための検証済みトークン時刻対応。
- 日本語学習者音声テストセットでのKWJAプロバイダー実験。
- 明示的なコンソールモジュール選択UIとInteraction入力契約。
