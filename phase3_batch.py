"""フェーズ3: 複数音声一括検証スクリプト
全音声ファイル × 3モデルで文字起こしを実行し、結果をCSVに保存する。
「ー」（U+30FC 長音符）はカタカナ混入としてカウントしない。
"""
import sys
import os
import time
import re
import datetime

import librosa
import numpy as np
import torch
import pandas as pd

# hiragana-asr の src を参照
HIRAGANA_ASR_DIR = os.path.join(os.path.dirname(__file__), "hiragana-asr")
sys.path.insert(0, HIRAGANA_ASR_DIR)

CHECKPOINT_PATH = os.path.join(HIRAGANA_ASR_DIR, "models/checkpoints/best-medium-ep5-inference.pt")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results.csv")
TARGET_SR = 16000
CHUNK_SEC = 30

# ---- ユーティリティ ----

def has_kanji(text):
    return bool(re.search(r'[一-鿿㐀-䶿]', text))

def has_katakana(text):
    # 「ー」（U+30FC 長音符）は除外
    filtered = re.sub(r'ー', '', text)
    return bool(re.search(r'[゠-ヿ]', filtered))

def get_speaker(filename):
    name = filename.lower()
    if 'burmese' in name or 'myanmar' in name:
        return 'burmese'
    elif 'indonesian' in name or 'indonesia' in name:
        return 'indonesian'
    elif 'vietnamese' in name or 'vietnam' in name:
        return 'vietnamese'
    return 'unknown'

def load_audio_16k(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y

def chunk_audio(audio, sr, chunk_sec):
    chunk_len = sr * chunk_sec
    return [audio[i:i+chunk_len] for i in range(0, len(audio), chunk_len)]

# ---- デバイス設定 ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    device = torch.device("cpu")
    device_name = "CPU"

print("=" * 65)
print("フェーズ3: 複数音声一括検証")
print(f"デバイス: {device_name}")
print(f"音声フォルダ: {AUDIO_DIR}")
print("=" * 65)

# 音声ファイル一覧
audio_files = sorted([
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg'))
])
print(f"対象ファイル数: {len(audio_files)}")
for f in audio_files:
    print(f"  - {f} (話者: {get_speaker(f)})")
print()

# ---- モデルロード ----
print("モデルを読み込み中...")

from src.asr.model import load_checkpoint
from src.asr.kana_vocab import KanaVocab
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

model_a = load_checkpoint(CHECKPOINT_PATH)
model_a.to(device); model_a.eval()
feat_ext_a = Wav2Vec2FeatureExtractor.from_pretrained("reazon-research/japanese-wav2vec2-large")
kana_vocab = KanaVocab()
print("  ✓ モデルA (sakasegawa) 読み込み完了")

proc_b = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
model_b = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
model_b.to(device); model_b.eval()
print("  ✓ モデルB (slplab) 読み込み完了")

proc_c = Wav2Vec2Processor.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
model_c = Wav2Vec2ForCTC.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
model_c.to(device); model_c.eval()
print("  ✓ モデルC (vumichien) 読み込み完了\n")

# ---- 推論関数 ----

def infer_model_a(chunks):
    texts = []
    for chunk in chunks:
        inputs = feat_ext_a(chunk, sampling_rate=TARGET_SR, return_tensors="pt", return_attention_mask=True)
        iv = inputs.input_values.to(device)
        am = inputs.attention_mask.to(device)
        with torch.no_grad():
            out = model_a(iv, attention_mask=am)
            pred_ids = out["kana_logits"].squeeze(0).argmax(dim=-1)
        texts.append(kana_vocab.decode(pred_ids.tolist()))
    return "".join(texts)

def infer_model_b(chunks):
    texts = []
    for chunk in chunks:
        inputs = proc_b(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_b(iv).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts.append(proc_b.batch_decode(pred_ids)[0])
    return "".join(texts)

def infer_model_c(chunks):
    texts = []
    for chunk in chunks:
        inputs = proc_c(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_c(iv).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts.append(proc_c.batch_decode(pred_ids)[0])
    return "".join(texts)

MODEL_INFO = [
    ("A", "sakasegawa/japanese-wav2vec2-large-hiragana-ctc", infer_model_a),
    ("B", "slplab/wav2vec2-xls-r-300m-japanese-hiragana",    infer_model_b),
    ("C", "vumichien/wav2vec2-large-xlsr-japanese-hiragana", infer_model_c),
]

# ---- 一括処理 ----
records = []
total = len(audio_files) * len(MODEL_INFO)
done = 0

for filename in audio_files:
    path = os.path.join(AUDIO_DIR, filename)
    speaker = get_speaker(filename)
    print(f"\n{'─'*65}")
    print(f"ファイル: {filename} (話者: {speaker})")
    print(f"{'─'*65}")

    audio = load_audio_16k(path)
    duration = len(audio) / TARGET_SR
    chunks = chunk_audio(audio, TARGET_SR, CHUNK_SEC)
    print(f"  読み込み完了: {duration:.1f}秒 / {len(chunks)}チャンク")

    for label, model_name, infer_fn in MODEL_INFO:
        done += 1
        print(f"\n  [{done}/{total}] モデル{label} 処理中...", end="", flush=True)
        t0 = time.perf_counter()
        try:
            text = infer_fn(chunks)
            elapsed = time.perf_counter() - t0
            kanji_flag = has_kanji(text)
            kata_flag = has_katakana(text)
            print(f" 完了 ({elapsed:.1f}秒)")
            print(f"    出力: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"    漢字混入:{kanji_flag} / カタカナ混入:{kata_flag}")
            records.append({
                "ファイル名": filename,
                "話者属性": speaker,
                "モデル名": model_name,
                "モデルラベル": f"モデル{label}",
                "出力テキスト": text,
                "漢字混入フラグ": kanji_flag,
                "カタカナ混入フラグ": kata_flag,
                "処理時間_秒": round(elapsed, 2),
                "音声長_秒": round(duration, 1),
            })
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f" エラー: {e}")
            records.append({
                "ファイル名": filename,
                "話者属性": speaker,
                "モデル名": model_name,
                "モデルラベル": f"モデル{label}",
                "出力テキスト": f"ERROR: {e}",
                "漢字混入フラグ": None,
                "カタカナ混入フラグ": None,
                "処理時間_秒": round(elapsed, 2),
                "音声長_秒": round(duration, 1),
            })

# ---- CSV保存 ----
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n\nCSV保存完了: {OUTPUT_CSV}")

# ---- サマリー表示 ----
print("\n" + "="*65)
print("処理サマリー")
print("="*65)

valid = df[~df["出力テキスト"].str.startswith("ERROR", na=False)]

print("\n▼ モデル別集計")
model_summary = valid.groupby("モデルラベル").agg(
    ファイル数=("ファイル名", "count"),
    漢字混入率=("漢字混入フラグ", lambda x: f"{x.mean()*100:.0f}%"),
    カタカナ混入率=("カタカナ混入フラグ", lambda x: f"{x.mean()*100:.0f}%"),
    平均処理時間=("処理時間_秒", lambda x: f"{x.mean():.1f}秒"),
).reset_index()
print(model_summary.to_string(index=False))

print("\n▼ 話者属性別 × モデル別 処理時間")
pivot = valid.pivot_table(
    values="処理時間_秒", index="話者属性", columns="モデルラベル", aggfunc="mean"
).round(1)
print(pivot.to_string())

print("\n▼ 全件一覧")
for _, row in df.iterrows():
    kanji = "漢字あり" if row["漢字混入フラグ"] else "漢字なし"
    kata  = "カタカナあり" if row["カタカナ混入フラグ"] else "カタカナなし"
    print(f"  {row['ファイル名'][:30]:30s} | {row['モデルラベル']} | {kanji} | {kata} | {row['処理時間_秒']}秒")
