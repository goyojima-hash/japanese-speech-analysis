# 正確さモジュール 実装計画

1. `AccuracyFactPacket` の失敗テストを先に書き、score／正誤／CEFRフィールドを拒否する。完了。
2. 最初の実装ではEvidence v1を維持し、モーラ時刻からASR時刻観測を作る。確率要約・Evidence v2は、モデル選定と検証後に追加する。生logitは保存しない。
3. `accuracy.py` に独立した `collect(bundle, reference=None)` を実装し、共有Sudachiトークンを使う。完了。
4. 明示的参照文だけに編集差分を許可し、未対応能力を理由付きで返す。完了。
5. APIの `fact_modules` 選択を追加し、既定のFluency／Range互換性を維持する。完了。
6. モジュール独立性、Range／Fluency出力契約、API、Judge境界を回帰テストで検証する。教師レビュー済みfixtureと確率較正は次段階。
