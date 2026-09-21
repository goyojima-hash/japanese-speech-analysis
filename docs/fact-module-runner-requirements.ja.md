# 共通 Fact-Module Runner 要件

## 目的

`EvidenceBundle` 生成後の事実モジュール実行経路を一つにします。API と対話ターミナルは同じ runner を呼び、Range、Accuracy、Coherence について同じパケット形式を受け取ります。

## 要件

- runner は生成済みの `EvidenceBundle` を受け取り、音声抽出やトークン化を再実行しない。
- モジュール選択、検証、Range の Evidence 消費、パケット組み立ては runner のみに置く。
- `selected_modules` を省略した runner は、実装済みの全事実モジュール（Fluency、Range、Accuracy、Coherence）を選択する。
- API と対話ターミナルは同じ「実装済み全モジュール」既定を使う。モジュール単独利用のため、呼び出し元が部分集合を明示選択することは引き続き可能とする。
- ターミナル進捗は、共通 Evidence、Fluency、Range、Accuracy、Coherence、Judge、協議の順とする。各モジュールのパケットは、その収集完了直後に表示する。
- 両方の呼び出し元は runner 結果から Judge 入力を組み立て、選択済みパケットは呼び出し元固有の条件分岐なしでコピーする。
- 未選択パケットは客観データと Judge 入力の両方に存在しない。
- `EvidenceBundle`、Fluency、Range、Accuracy、Coherence の抽出アルゴリズムは変更しない。

## 受け入れ条件

1. API とターミナルは個別の Range / Accuracy / Coherence 収集器ではなく `run_fact_modules` を呼ぶ。
2. 選択された各テキストモジュールは、一つの `LinguisticEvidence` インスタンスを読む。
3. ターミナル出力に `=== Coherence客観データ ===` があり、Judge 入力に `coherence_data` がある。
4. API とターミナルの既定実行は実装済み4モジュールすべてを報告し、客観データと Judge 入力の両方に Range、Accuracy、Coherence パケットを含む。
