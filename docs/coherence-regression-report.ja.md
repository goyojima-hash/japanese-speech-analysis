# Coherence V1 回帰・バグハンターレポート

## 対象

独立した事実専用 Coherence V1 モジュールを対象とするレポートです。共通 Evidence 層および既存の Fluency、Range、Accuracy の契約を変えないことを確認するため、実装前に再現した失敗を記録します。

## 再現した失敗と解決

| 再現器 | REDで観測した状態 | 解決 | 回帰防止 |
| --- | --- | --- | --- |
| Coherence収集器／API選択 | `ModuleNotFoundError` と未対応の `coherence` 選択 | `CoherenceModule` と明示選択のAPI registryを追加 | 単体・API選択テスト |
| Judge境界 | promptに `coherence_data` を事実専用とする説明がない | 候補単位・未提供能力・観測だけから品質／誤り／レベルを決めることを禁止 | Promptテスト |
| 候補単位の終端来歴 | 最終候補単位に `transcript_end` 導出がない | 最終単位に終端導出を追加 | 収集器テスト |

## 5 Whys: Coherence が評価を歪める可能性はなぜ生じるか

1. Judge がルール生成した候補境界を、学習者の誤りや品質スコアとして扱う可能性がある。
2. 明示的な境界がなければ、談話らしい観測は評価的に見える可能性がある。
3. ASR文字起こしの句読点やトークン時刻対応は、検証済みの文注釈ではない。
4. このプロジェクトでは、日本語学習者音声で検証済みの高度な談話モデルがまだない。
5. したがってV1は、追跡可能な観測と名前付きの制限だけを出し、Judgeにも各観測単独から推論しないよう指示する必要がある。

防止した根本原因: 決定的な抽出結果を熟達度の判定と混同すること。

## 特性要因分析

| 領域 | リスク | 実装した対策 |
| --- | --- | --- |
| データ | ASR境界が不完全 | candidate表記、導出理由、fallback、未提供能力 |
| ルール | 辞書が隠れた採点になる | 版付きの説明カテゴリ、score/rating/errorなし |
| 統合 | 既存モジュールを再実行／変更する | Coherenceは `EvidenceBundle` を読むだけ、デフォルト不変 |
| API | 未選択モジュールが応答を変える | 明示選択時だけ `coherence_data` を追加 |
| Judge | 観測を過剰解釈する | 事実専用のPrompt境界とテスト |
| 運用 | 将来モデルがキャッシュを無効化する | V1ではモデル追加・Evidence schema変更なし |

## タイムライン

1. 収集器、API選択、Prompt境界のREDテストを追加した。
2. 未実装モジュール／registryとPrompt文言不足を意図どおり失敗として確認した。
3. 決定的収集器、明示選択の配線、事実専用Judge指示を追加した。
4. 終端来歴の再現器を追加してREDを確認し、`transcript_end` 来歴を追加した。
5. Range、Accuracy、Coherence が共有トークン化を一回だけ読むことを確認した。
6. 集中テスト、全テスト、コンパイル、空白チェックを実行した。

## 再発防止の確認

- デフォルトAPI出力は `fluency` と `range` のままで、`coherence_data` を含まない。
- 明示的な `selected_modules=("coherence",)` はRangeを動かさない。
- CoherenceにはAIモデル、score、CEFR/JFS、正誤、error、qualityの出力フィールドがない。
- `evidence/`、`audio_pipeline.py`、`range.py`、`accuracy.py` の本番変更はない。
