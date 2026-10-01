# 発話行為観測と項目別基準の段階導入案（旧案）

> この案に含まれる特定の設問・サンプル数の前提は採用しない。現在の汎用設計と実装状況は [発話行為観測と項目別達成度の分離・検証](../docs/task-assessment-shadow-design.ja.md) を参照。

## 目的と不変条件

タスク達成度の「発話内容の観測」と「設問固有の基準への変換」を分け、直接判定との比較で有効性を確認する。開発用最小対8組の8/8は仮説の根拠であり、一般化性能の証明とは扱わない。

- 共通 `EvidenceBundle v1` と５軸のFact Moduleは変更しない。相互依存もLLM呼び出しも加えない。
- AIによる発話行為の解釈はJudge層に置き、「客観Fact」ではなく推定観測として保存する。
- `AutoLevelJudgeResult`、公開APIの既存 `task_rating` / `final_cefr_level`、CLI表示、既存のCEFR協議を初期段階で変更しない。
- 生音声はJudgeへ送らない。Transcript、共有Evidenceの公開可能部分、選択されたFact packet、設問情報だけを使う。
- 現状のAPIとCLIはJudge入力（Fluency選択時の指標）と達成度の集約規則が異なる。最初は両者の旧挙動を維持するアダプタを介し、共通サービス化は差分を固定した後に別段階で行う。

## 境界とデータ契約

既存経路: `EvidencePipeline.build` → `run_fact_modules` → `judge_auto_cefr_*` → `deliberate_auto_cefr` → 既存結果。

追加する比較経路: 同じJudge入力から `TaskObservation` → `TaskRubric` → `TaskAssessment`。旧経路の `task_rating` を上書きしない。`assessment_mode=off` をデフォルトにし、明示的に `shadow` を選んだ場合だけ追加のJudge呼び出しを行う。比較結果は任意の追加フィールド `task_assessment_shadow` に置き、従来の利用側はこれを無視できる。`off` 時は呼び出し回数、費用、遅延、応答形式を維持する。`shadow` の通信には独立した時間上限・例外境界を設け、失敗は旧評価の失敗にしない。

`TaskContext v1` はFact Module選択と独立した入力契約。`prompt_id`、原文、承認済みの `task_type/rubric_id`、回答の対応状態、対象区間、文字起こしとの対応根拠を持つ。現行の `roleplay_task` と `interaction_context` から後方互換で生成できる範囲だけ生成し、足りない値を推測で埋めない。Interactionを外しても入力された設問情報はJudgeへ渡す。設問本文からの自動 `task_type` 推定は比較実験としては可能でも、検証済み達成度の適用条件にはしない。

`TaskObservation` は `schema_version`、`prompt_id`、`segment_id`、`act_type`、`target_slot`、`observed_values`、`transcript_span`（原文引用＋位置）、`uncertainty`、`judge_id/model`、`risk_flags` を持つ。推定観測には「なし」「不明」を区別して設け、引用が元Transcriptに一致しない出力は無効とする。複数Judgeの観測が食い違う場合は安易に多数決で確定せず要確認にする。区間内引用かどうかまで検証するには、文字起こしと時刻の対応が必要である。

`TaskRubric` は人が作成・承認する設問型別の設定。`rubric_id/version`、適用可能な `task_type`、必須観測、◎○△×への決定規則、`unknown`時の保留条件、根拠資料、承認者を持つ。`task_type` は承認済み設問メタデータで指定し、任意の設問文から自動決め打ちしない。LLMのプロンプト内に埋め込むのではなく、版管理したルールとして読み込み、純粋関数で適用する。初回は「日付確認」１種類のみ。[JFロールプレイテスト・テスター用マニュアル](https://www.jfstandard.jpf.go.jp/pdf/roleplay/JFS_roleplaytest_all_20241003.pdf)を参考にするが、J-GRADE独自の具体規則を公式基準そのものと称さない。

`TaskAssessment` は `status=applied|not_applicable|insufficient_context|conflicted|invalid_observation`、適用時だけ `rating`、`rule_id`、`rubric_version`、観測参照、要確認理由を持つ。達成度の４段階と不明／適用不能を混同しない。設問と回答区間の対応が `confirmed` で、**その区間の文字起こしも検証可能**な場合だけ項目別基準を適用する。現在のEvidenceは録音全体のTranscriptと音声区間時刻を持つが、テキストと区間の正確な対応は持たない。初回は単一設問・録音全体がその回答だと明示確認されたケースに限定する。複数設問の録音は将来の時刻付きSTT／アライメントまたは人手で確認した区間別Transcriptが整うまで `insufficient_context` とする。設問不明でも従来どおり暫定CEFR推定は続けられる。

CEFR判定は独立して現行Judge→協議を維持する。新しいTaskAssessmentをCEFRへ入力するのは別の検証・版上げ後とし、同一データから導いた値を独立証拠として二重計上しない。

## 導入手順（１PRずつ、各段階で巻き戻し可能）

1. **契約とベースライン**：既存のAPI/CLI JSON・表示、５軸Fact packet、現行Judge出力を固定テストにする。API/CLIのJudge入力と集約の差を明文化する。開発用8組とは別の非公開・未使用評価セットを用意し、人手ラベルを可能なら複数名で確定する。直接判定の指標を記録する。変更はテスト／仕様のみ。
2. **比較用の観測経路**：５軸から独立した `TaskContext v1` と、Judge層の観測専用構造化呼び出し・厳格な検証を追加。例示した日付確認、かつ単一設問・録音全体が回答の確認済み例に限定し、観測が失敗しても旧経路は成功させる。API/CLIそれぞれの既存Judgeアダプタは残し、共通観測サービスを呼ぶ。デフォルト `off`、検証時だけ `shadow`。既存フィールドは不変。
3. **固定基準と比較**：版管理ルーブリックの決定的な変換を追加。無効な引用、設問対応未確定、基準未登録、Judge不一致は保留。旧判定と新判定の差分、根拠、遅延／費用を監査記録に残す。外部表示は明確に「検証中」とする。
4. **採用判定と共通化**：凍結した未使用セットで直接判定と同一条件比較。項目別の誤陽性・誤陰性、◎○△×の一致率、保留率、人手一致、入力欠損時の挙動、モデル間差、費用・遅延を見る。事前に決めた合格基準を満たした項目だけ明示的な `validated` モードで採用する。`validated` は新しい版のAPI／明示的な利用者選択でだけ正式 `task_rating` に反映し、旧判定は `legacy_task_rating` として残す。既存API版の `task_rating` は変えない。API/CLIの既存差分を解消するのは別の互換性テストと移行を伴う変更とし、影響を黙って混ぜない。不合格項目は `shadow` のまま／停止する。新項目は新しいルーブリックと独立検証を必須とする。

## 想定ファイル境界（実装時の提案）

- 既存を維持: `jgrade_eval/evidence/*`, `jgrade_eval/fact_modules.py`, `jgrade_eval/{range,accuracy,coherence,interaction}.py`。
- 新設: `jgrade_eval/task_context.py`（５軸から独立した設問入力）、`jgrade_eval/task_observation.py`（Judge出力契約・検証）、`jgrade_eval/task_rubrics.py`（版管理と純粋な照合）、`jgrade_eval/judge_service.py`（比較経路と旧Judgeアダプタの共通境界）。
- 最小変更: `jgrade_eval/{api_service,interactive}.py` は旧Judgeの挙動を保つアダプタ呼び出しと、比較経路の明示的な有効化・表示のみ。`jgrade_eval/{live_judges,prompts,models}.py` に観測専用の別契約を追加し、旧 `AutoLevelJudgeResult` を破壊しない。
- 回帰: `tests/test_jgrade_api.py`, `tests/test_jgrade_eval.py`, `tests/test_fact_modules.py` と新しい観測／基準／CLI-API同値性テスト。

## ロールバック

`assessment_mode=off` に戻すだけで既存Judge経路が動く。比較経路の失敗は既存の `task_rating` とCEFRを変更しない。採用後も原始観測、ルーブリック版、旧判定を保存し、特定版の基準だけを停止できるようにする。
