# 正確さモジュール 回帰防止レポート

## 問題

既存のFluency／Rangeの事実出力契約を変えずに、正確さを追加する。正確さは観測値だけを返し、正誤・スコア・CEFR・LLM結論は後段の責務とする。

## 再現手順／RED証拠

1. `uv run python -m unittest tests.test_accuracy tests.test_evidence` を実行する。
2. 新規契約テストで `ModuleNotFoundError: No module named 'jgrade_eval.accuracy'` を確認する。
3. 同じ実行で、既存Evidence／Range／Fluency互換テスト7件は成功していることを確認する。

## タイムライン

- Evidence v1の実装は `eeecf7c` にコミット済み。
- 本体コード前にAccuracy契約テストとFluency互換スナップショットを追加。
- 意図どおり未実装モジュールのREDを再現。

## 5 Whys

1. なぜ新規テストはREDか。`AccuracyModule` が存在しない。
2. なぜ存在しないか。設計のみで実装していなかった。
3. なぜコードより先にテストか。事実のみの契約を固定し、採点の混入を防ぐため。
4. なぜFluency／Rangeも検証するか。共有Evidence拡張は既存出力やトークン再利用を壊し得るため。
5. なぜ残すか。将来の能力追加ごとに既存モジュール互換性を証明するため。

## フィッシュボーン

- コード: Evidence v1にAccuracy収集器・FactPacketがない。
- データ: 現在のWav2Vec2受け渡しはモーラ時刻を持つが、確率要約を持たない。
- 統合: APIにはAccuracyを選択する経路がない。
- 回帰リスク: Evidenceスキーマ変更がFluency直列化とRangeの共有トークン利用へ影響する。

## 修正と再発防止

- ASR時刻観測、形態素観測、任意参照差分、未対応能力だけを返す独立収集器を実装する。
- 確率要約はコンパクトな提供者と較正ができるまで未対応として明示する。
- API選択機能は単体契約がGREENになってから追加する。
- 完了前に集中テスト、全テスト、出力契約回帰を実行する。

## 修正と検証結果

- `jgrade_eval/accuracy.py` を追加。モーラ時刻ASR観測、共有形態素観測、明示的参照文との差分、未対応能力だけを返す。
- 任意API `fact_modules` を追加。既定は従来どおり `fluency` と `range` であり、Accuracyは選択されない限り実行もJudge送信もしない。
- Judgeプロンプトに境界を追加。`accuracy_data` は観測値であり、未対応／未較正ASR情報や参照差分を学習者の誤り指示として扱わない。
- 集中テスト17件、全テスト76件、コンパイル、`git diff --check` を成功確認。
- Accuracy実装を直前の共通Evidenceチェックポイント `eeecf7c` と比較し、`jgrade_eval/audio_pipeline.py`、`jgrade_eval/range.py`、`jgrade_eval/evidence/` に差分がないことを確認。Accuracyは追加的で、選択時だけ動く。
- 既存実音声の客観データ基準 `outputs/jgrade_objective_data.json` を確認。Wav2Vec2／Sileroの同じ抽出元を記録している。Accuracyが既存抽出コードを変更していないため、新たなライブ推論は不要だった。
- 現在の `uv` 環境にはcoverageツールがない（`No module named coverage`）。成功したカバレッジ測定として扱わず記録する。
