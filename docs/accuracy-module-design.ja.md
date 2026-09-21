# 正確さモジュール 設計

## 決定

正確さは分類器ではなく事実収集器とする。共通層へ任意のASRアラインメント能力を追加し、Accuracyは既存の共有形態素を再トークン化せず利用する。

## 信頼度と正確さを混同しない

Wav2Vec2 CTCのlogitはモデルが出したトークン確率であり、学習者の発音・文法の正確さではない。そのため確率由来の値には、導出方法、モデルID、デコード設定、`calibration_status="uncalibrated"` を必ず添える。

## 実装済みv1と将来の提供者境界

実装済みv1は安定したモーラ時刻から `AsrObservation` を直接作るため、`EvidenceBundle`、Fluency事実、Range事実を変更しない。確率フィールドはnullで、能力状態は `unavailable` である。

将来、検証済みのコンパクトなCTC確率提供者を選ぶ場合だけ、以下の追加的なEvidence v2拡張を使う。

```text
EvidenceBundle v2（追加的変更）
└── asr_alignment
    ├── decoder_provenance
    ├── units: unit / start / end / posterior / margin
    ├── derivation = ctc_frame_softmax
    └── calibration_status = uncalibrated

AccuracyFactPacket
├── ASR観測
├── 形態素・局所パターン観測
├── 任意参照文との差分
└── 未対応能力
```

フレーム全体のlogitは保存しない。将来のASRアダプターは、既に生成しているlogitから発話単位の要約値だけを作る。

## 収集器

- ASR: CTCで得た単位区間ごとに、選択確率の平均と次点との差を記録する。得られない場合は未対応とする。
- 形態素: `particle_after_noun`、`auxiliary_after_verb`、活用形など、観測された並びをパターンIDとして記録する。妥当／不適切とは判定しない。
- 参照差分: 参照文が明示された時だけ、決められた正規化規則と編集距離で `equal` / `insert` / `delete` / `replace` を記録する。

## 実行境界

`ModuleRunner` が選択された能力を先に伝える。RangeだけならCTC確率抽出をせず、AccuracyだけならRange/Fluencyの解釈モジュールを呼ばない。AI Judgeだけが集めたFactPacketを評価する。

## 保留

- 日本語学習者音声での信頼度較正。
- 音素レベル発音分析に使う音響モデル・辞書・G2P方針。
- Judgeが解釈する局所パターンの教師レビュー済みルール群。
