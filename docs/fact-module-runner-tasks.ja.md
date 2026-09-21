# 共通 Fact-Module Runner 実装計画

1. ターミナルの Coherence 出力／Judge 入力と共通 runner 選択を証明する RED テストを追加する。
2. 選択定数、`FactModuleRun`、runner を持つ `fact_modules.py` を追加する。
3. API とターミナルを runner 利用へリファクタリングし、各呼び出し元から直接の事実モジュール収集を除く。
4. ターミナル表示を任意の Coherence データに対応させる。
5. API の既定をターミナルの既定に合わせ、呼び出し元が部分集合を明示指定しない限り、両方で実装済み全モジュールを選択する。
6. 集中テスト、全テスト、コンパイル、空白チェックを実行し、Bug Hunter 回帰レポートを残す。
