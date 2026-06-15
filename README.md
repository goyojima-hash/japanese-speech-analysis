# J-GRADE STT 検証環境

外国人日本語学習者の発話を **ひらがなのみ・意味補正なし** でテキスト化するSTTモデルの検証環境。

## 使用モデル

| ラベル | モデルID | ライセンス |
|--------|---------|---------|
| モデルA（推奨） | sakasegawa/japanese-wav2vec2-large-hiragana-ctc | Apache-2.0 |
| モデルB | slplab/wav2vec2-xls-r-300m-japanese-hiragana | 記載なし |
| モデルC | vumichien/wav2vec2-large-xlsr-japanese-hiragana | Apache-2.0 |

---

## 環境構築手順

### 前提

- macOS（Apple Silicon 推奨）
- Git
- [uv](https://docs.astral.sh/uv/)
- インターネット接続（モデルの初回ダウンロードに必要）

### 1. リポジトリのクローン

```bash
git clone --recurse-submodules https://github.com/goyojima-hash/japanese-speech-analysis.git
cd japanese-speech-analysis
```

すでにクローン済みの場合は、submoduleを初期化してください。

```bash
git submodule update --init --recursive
```

### 2. Python環境の構築

```bash
uv python install 3.11
uv sync --frozen
```

`.python-version`と`uv.lock`に基づいて、プロジェクト内の`.venv/`へ環境が作成されます。

### 3. モデルA チェックポイントのダウンロード（約600MB）

```bash
uv run python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="sakasegawa/japanese-wav2vec2-large-hiragana-ctc",
    filename="best-medium-ep5-inference.pt",
    local_dir="hiragana-asr/models/checkpoints",
)
print("ダウンロード完了")
PY
```

### 4. 3モデルの読み込み確認

```bash
uv run python check_models.py
```

---

## フォルダ構成

```
japanese-speech-analysis/
├── audio/                          # 音声ファイル（MP3/WAV等）
│   ├── burmese_japanese_1min.mp3
│   ├── indonesian_japanese_1min.mp3
│   └── vietnamese_japanese_1min.mp3
├── hiragana-asr/                   # モデルA用submodule
│   ├── models/checkpoints/
│   │   └── best-medium-ep5-inference.pt  # チェックポイント（要ダウンロード）
│   └── src/asr/                    # モデルAの推論コード
├── .venv/                          # uvが作成するPython仮想環境
├── pyproject.toml                  # 直接依存関係
├── uv.lock                         # 解決済み依存関係
├── check_models.py                 # フェーズ1: モデル読み込み確認
├── phase2_single_test.py           # フェーズ2: 単体動作確認
├── phase3_batch.py                 # フェーズ3: 一括処理
├── results.csv                     # フェーズ3の出力結果
└── report.md                       # 検証レポート
```

---

## 実行手順

### フェーズ1: モデル読み込み確認

```bash
uv run python check_models.py
```

3モデルすべて「✓ 読み込み成功」と表示されれば OK。

### フェーズ2: 単体動作確認（1ファイル）

```bash
uv run python phase2_single_test.py
```

デフォルトは `audio/vietnamese_japanese_1min.mp3` を対象に3モデルで実行。  
対象ファイルを変更する場合はスクリプト内の `AUDIO_FILE` を編集。

### フェーズ3: 一括処理（全ファイル）

```bash
uv run python phase3_batch.py
```

`audio/` フォルダ内の全音声ファイルを3モデルで処理し、`results.csv` に保存。

#### 音声ファイルの命名規則

ファイル名に話者属性を含めること（自動判定される）：

| 含める文字列 | 判定される話者属性 |
|------------|----------------|
| `burmese` または `myanmar` | burmese |
| `indonesian` または `indonesia` | indonesian |
| `vietnamese` または `vietnam` | vietnamese |

---

## 音声ファイルの前処理

スクリプトは音声ファイルを自動で以下の形式に変換して処理します：

- サンプリングレート: **16kHz**（モデルの要求仕様）
- チャンネル: **モノラル**
- 対応形式: MP3, WAV, M4A, FLAC, OGG

変換は `librosa` が担当するため、別途変換ツールは不要。

---

## 出力 CSV の列説明

| 列名 | 内容 |
|------|------|
| ファイル名 | 処理した音声ファイル名 |
| 話者属性 | ファイル名から自動判定（burmese / indonesian / vietnamese） |
| モデル名 | HuggingFace モデルID |
| モデルラベル | モデルA / B / C |
| 出力テキスト | 文字起こし結果 |
| 漢字混入フラグ | True / False |
| カタカナ混入フラグ | True / False（「ー」長音符は除外） |
| 処理時間_秒 | 推論にかかった秒数 |
| 音声長_秒 | 音声ファイルの長さ（秒） |

---

## トラブルシューティング

### `torchaudio.load` でエラーが出る

torchaudio 2.11以降はFFmpegが必要です。このプロジェクトでは `librosa` で代替しているため問題ありません。

### モデルAのロードで `UNEXPECTED` キーの警告が出る

```
project_hid.bias | UNEXPECTED
```

推論には影響しません。無視して問題ありません。

### メモリ不足・速度低下

3モデル同時ロードでApple Silicon GPUのメモリ競合が発生する場合があります。  
1モデルずつ実行することで処理速度が大幅に改善します（フェーズ2参照）。

---

## 検証結果サマリー

→ 詳細は [report.md](report.md) を参照

| モデル | 漢字混入 | カタカナ混入 | 処理速度（60秒/単独） | 推奨度 |
|--------|---------|------------|-------------------|------|
| **A (sakasegawa)** | 0% | 0% | **約10秒** | **★★★ 推奨** |
| B (slplab) | 0% | 0% | 約34秒 | ★★ 参考 |
| C (vumichien) | 0% | 0% | 約302秒 | ★ 実用困難 |
