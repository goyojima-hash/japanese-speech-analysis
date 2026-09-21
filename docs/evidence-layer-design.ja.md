# 共通Evidence層 設計

## 決定

共有するのは、バージョン付きの客観的事実だけとする。各軸モジュールは独立に選択でき、軸固有の解釈だけを行う。CEFR/JFS評価は、収集された事実を後段AI Judgeが行う。

## 構成

- `evidence/models.py`: 不変かつ直列化可能なEvidenceレコード。
- `evidence/speech.py`: 既存Fluency出力を事実のみのSpeechEvidenceへ変換するアダプター。
- `evidence/linguistic.py`: SudachiPyによる共有形態素解析。
- `evidence/cache.py`: 来歴キーによる任意JSONキャッシュ。
- `evidence/pipeline.py`: Evidence生成と任意キャッシュ。
- `range.py`: 従来APIを保ちつつ、共有トークンを受け取る入口を追加。

## 境界

`EvidenceBundle` に入るのは文字起こし、時刻、形態素、モデル・辞書来歴などの事実のみである。数値能力スコア、文法誤りラベル、流暢さgrade、CEFR、AI Judge結論は入れない。

## 拡張

将来のAccuracy、Interaction、Coherenceの情報は、v1の意味を変えず、能力フィールドを追加して導入する。
