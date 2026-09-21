# 日本語学習者の発話を対象にしたJ-GRADEモデル選定結論

日付: 2026-09-14  
決定状況: 調査による結論。新モデルの導入・採用はまだ行いません。

## 結論

J-GRADEにおいて、外国人学習者のCEFR／JFレベルを直接決める**単独の「最適モデル」はありません。** 最適なのは、校正されたEvidenceパイプラインです。独立した事実抽出器がバージョン付き観測事実を作り、その後に複数JudgeがJFスタンダードのCan-doで評価し、不一致を人間確認へ出します。

これは評価対象に合っています。国際交流基金は、JFスタンダードのレベルを文法・語彙量だけではなく「何ができるか」で定義しています。また話し言葉の質的側面として、Range、Accuracy、Fluency、Interaction、Coherenceを並列に示しています。[JFスタンダード概要](https://www.jfstandard.jpf.go.jp/summary/ja/render.do)、[JFスタンダードガイドブック p.86](https://www.jfstandard.jpf.go.jp/pdf/web_whole_en.pdf)

## モジュールごとの具体的な決定

| J-GRADEでの役割 | 現時点の最適選択 | J-GRADEに適する理由 | してはいけないこと |
| --- | --- | --- | --- |
| 共通の文字起こし／時刻 | `vumichien/wav2vec2-large-xlsr-japanese-hiragana` とSilero VADを維持 | すでに共通の事実基盤を作っています。日本語学習者ASRの根拠なしに交換すると、上流のリスクを移すだけです。 | ASR出力やconfidenceを学習者の正確さにする。 |
| Coherence v1 | **追加モデルなし**、共有Sudachiの事実 | 透明な接続表現、単位長、反復、境界観測で検証可能な基準線を作れ、独立モジュール性を守れます。 | 汎用LLMから始める、類似度を熟達度点と呼ぶ。 |
| Coherence v2 | **KWJAを保持テスト後に任意プロバイダーとして使う** | 談話関係、係り受け、格構造、橋渡し照応、共参照まで扱う日本語専用候補で、汎用埋め込みより評価軸に近いです。 | 共有トークナイズを置換する、談話関係の予測を真実と扱う。 |
| Interaction v1 | **会話入力ができてからpyannote Community-1** | Interactionには話者ターンと重なりの事実が不可欠で、ローカルの話者分離とASR時刻照合ができます。 | 一人音声で応答ラグを計算する、学習者対話ラベルなしのターン予測器を使う。 |
| レベル判定 | **教師校正済みの複数Judgeパネル** | CEFR／JFは課題達成と5軸の根拠を総合する構成概念です。汎用ベンチマークで選ぶのでなく、訓練された人間評価者と照合して検証する必要があります。 | 校正されていない単独モデルを正解とする。 |

`multilingual-e5-large-instruct` は主候補ではなく実験候補に留めます。多言語の文ベクトル類似度は出せますが、JFスタンダードの課題達成・やりとり・まとまりそのものを表しません。[モデルカード](https://huggingface.co/intfloat/multilingual-e5-large-instruct)

## Coherenceの観測にKWJAを選ぶ理由（汎用LLMではない）

KWJAは形態素、構文、格構造、橋渡し照応、共参照に加え、談話関係を扱う日本語統合解析器です。その出力は任意のCoherenceパケットに入れる**モデル由来の観測事実**として適しています。ただしレベル分類器ではありません。関連する結束性解析器はより重い前処理／モデルスタックを必要とするため、プロバイダー境界に隔離してベンチマークすべきです。[KWJA ACLシステム論文](https://aclanthology.org/2023.acl-demo.52)、[cohesion analyzer](https://github.com/nobu-g/cohesion-analysis)

## 最初のInteraction試作にpyannoteを選ぶ理由（NeMoではない）

Interactionの事実には、誰がいつ話したかが必要です。`pyannote/speaker-diarization-community-1` はローカル実行、ダウンロード後のオフライン実行、16kHz入力／再サンプリング、そしてASR時刻と話者ターンを合わせるためのexclusive timelineを提供します。そのため現行パイプラインへの最小の有効拡張です。初回ダウンロードではHFトークンと利用条件同意が必要で、CC-BY-4.0ライセンスです。[モデルカード](https://huggingface.co/pyannote/speaker-diarization-community-1)

NeMoにはend-to-endのSortformerと、VAD＋TitaNet＋MSDDがあります。CUDA環境では比較価値がありますが、現在のmacOS中心ローカル環境には運用スタックが大きく、最初の標準候補にはしません。[NeMoガイド](https://docs.nvidia.com/nemo-framework/user-guide/25.07/nemotoolkit/asr/speaker_diarization/models.html)

## 「最適」と言うために必要な根拠

汎用の日本語／多言語ベンチマークだけでは足りません。J-GRADEは第二言語としての日本語発話を評価するからです。I-JASには12母語群の日本語学習者1,000名の発話と作文が含まれ、C-JASには自然な学習者／母語話者会話が約46.5時間含まれます。どちらも有望な評価資料ですが、利用・公開条件を確認してから使います。C-JAS書き起こしはCC BY-NC-ND 4.0です。[I-JAS](https://www2.ninjal.ac.jp/jll/lsaj/ijas-document.html)、[C-JAS](https://mmsrv.ninjal.ac.jp/c-jas/en/index.html)

採用前には、課題プロンプト、音声、正解文字起こし、必要時の話者ターン／重なり境界、複数の教師による独立CEFR／JFラベルを持つ保持テストセットを作ります。選定指標は次です。

1. ネイティブ音声だけでなく、学習者音声でのASR誤りと時刻誤り。
2. 学習者対話での話者分離誤り率、発話見落とし、話者混同。
3. モジュール事実の安定性、来歴、実行時間、失敗時の挙動。
4. 人間ラベルとの後段一致、confidence校正、母語・課題・録音条件・レベル別の公平性。
5. Judgeの不一致またはEvidence不足時の必須人間確認。

## 実装順

1. 既存EvidenceBundleだけでCoherence v1を作る。
2. 日本語学習者発話の検証セットを収集・ラベル付けする。
3. 決定的なCoherence基準線とKWJAを任意プロバイダーとして比較する。
4. 複数話者入力契約を作ってから、pyannoteでInteractionの事実を試作する。
5. CUDA導入または低遅延要件が出たときだけ、pyannoteとNeMoを比較する。
6. 既存の複数Judgeを教師評価者に校正する。ここまで済んで初めて、J-GRADEは根拠ある熟達度判定を主張できます。
