"""フェーズ2: 単体動作確認スクリプト
音声ファイル1本でモデルA・B・C全てを実行し、出力内容を確認する。
"""
import sys
import os
import time
import re
import unicodedata

import librosa
import numpy as np
import torch

# hiragana-asr の src を参照
HIRAGANA_ASR_DIR = os.path.join(os.path.dirname(__file__), "hiragana-asr")
sys.path.insert(0, HIRAGANA_ASR_DIR)

CHECKPOINT_PATH = os.path.join(HIRAGANA_ASR_DIR, "models/checkpoints/best-medium-ep5-inference.pt")
AUDIO_FILE = os.path.join(os.path.dirname(__file__), "audio/vietnamese_japanese_1min.mp3")
TARGET_SR = 16000
CHUNK_SEC = 30  # 長い音声はチャンク分割

# ---- ユーティリティ ----

def has_kanji(text):
    return bool(re.search(r'[一-鿿㐀-䶿]', text))

def has_katakana(text):
    return bool(re.search(r'[゠-ヿ]', text))

def load_audio_16k(path):
    """librosaで読み込み → 16kHz・モノラルに変換"""
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    print(f"  音声読み込み完了: {len(y)/TARGET_SR:.1f}秒 / {TARGET_SR}Hz / モノラル")
    return y

def chunk_audio(audio, sr, chunk_sec):
    """音声を chunk_sec 秒ごとに分割"""
    chunk_len = sr * chunk_sec
    return [audio[i:i+chunk_len] for i in range(0, len(audio), chunk_len)]

def check_output(text, model_label):
    kanji = has_kanji(text)
    kata = has_katakana(text)
    empty = len(text.strip()) == 0
    print(f"\n  --- チェック結果 ({model_label}) ---")
    print(f"  ひらがなのみ:   {'✓' if not kanji and not kata and not empty else '✗'}")
    print(f"  漢字混入:       {'あり ✗' if kanji else 'なし ✓'}")
    print(f"  カタカナ混入:   {'あり ✗' if kata else 'なし ✓'}")
    print(f"  空出力:         {'あり ✗' if empty else 'なし ✓'}")

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

print("=" * 60)
print(f"フェーズ2: 単体動作確認")
print(f"対象ファイル: {os.path.basename(AUDIO_FILE)}")
print(f"デバイス: {device_name}")
print("=" * 60)

audio = load_audio_16k(AUDIO_FILE)
chunks = chunk_audio(audio, TARGET_SR, CHUNK_SEC)
print(f"  チャンク数: {len(chunks)} ({CHUNK_SEC}秒ごと)\n")

results = {}

# ======================================================================
# モデルA: sakasegawa (hiragana-asr リポジトリ)
# ======================================================================
print("\n" + "="*60)
print("モデルA: sakasegawa/japanese-wav2vec2-large-hiragana-ctc")
print("="*60)
try:
    from src.asr.model import load_checkpoint
    from src.asr.kana_vocab import KanaVocab
    from transformers import Wav2Vec2FeatureExtractor

    model_a = load_checkpoint(CHECKPOINT_PATH)
    model_a.to(device)
    model_a.eval()

    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained("reazon-research/japanese-wav2vec2-large")
    kana_vocab = KanaVocab()

    texts_a = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        inputs = feat_ext(chunk, sampling_rate=TARGET_SR, return_tensors="pt", return_attention_mask=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            out = model_a(input_values, attention_mask=attention_mask)
            pred_ids = out["kana_logits"].squeeze(0).argmax(dim=-1)
        text = kana_vocab.decode(pred_ids.tolist())
        texts_a.append(text)
        print(f"  チャンク{i+1}: {text}")

    elapsed_a = time.perf_counter() - t0
    full_text_a = "".join(texts_a)
    print(f"\n  【モデルA 全文出力】\n  {full_text_a}")
    print(f"  処理時間: {elapsed_a:.1f}秒")
    check_output(full_text_a, "モデルA")
    results["A"] = {"text": full_text_a, "time": elapsed_a, "success": True}

except Exception as e:
    print(f"  ✗ エラー: {e}")
    import traceback; traceback.print_exc()
    results["A"] = {"text": "", "time": 0, "success": False, "error": str(e)}

# ======================================================================
# モデルB: slplab
# ======================================================================
print("\n" + "="*60)
print("モデルB: slplab/wav2vec2-xls-r-300m-japanese-hiragana")
print("="*60)
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc_b = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
    model_b = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
    model_b.to(device)
    model_b.eval()

    texts_b = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        inputs = proc_b(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_b(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = proc_b.batch_decode(pred_ids)[0]
        texts_b.append(text)
        print(f"  チャンク{i+1}: {text}")

    elapsed_b = time.perf_counter() - t0
    full_text_b = "".join(texts_b)
    print(f"\n  【モデルB 全文出力】\n  {full_text_b}")
    print(f"  処理時間: {elapsed_b:.1f}秒")
    check_output(full_text_b, "モデルB")
    results["B"] = {"text": full_text_b, "time": elapsed_b, "success": True}

except Exception as e:
    print(f"  ✗ エラー: {e}")
    import traceback; traceback.print_exc()
    results["B"] = {"text": "", "time": 0, "success": False, "error": str(e)}

# ======================================================================
# モデルC: vumichien
# ======================================================================
print("\n" + "="*60)
print("モデルC: vumichien/wav2vec2-large-xlsr-japanese-hiragana")
print("="*60)
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    proc_c = Wav2Vec2Processor.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
    model_c = Wav2Vec2ForCTC.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
    model_c.to(device)
    model_c.eval()

    texts_c = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        inputs = proc_c(chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_c(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = proc_c.batch_decode(pred_ids)[0]
        texts_c.append(text)
        print(f"  チャンク{i+1}: {text}")

    elapsed_c = time.perf_counter() - t0
    full_text_c = "".join(texts_c)
    print(f"\n  【モデルC 全文出力】\n  {full_text_c}")
    print(f"  処理時間: {elapsed_c:.1f}秒")
    check_output(full_text_c, "モデルC")
    results["C"] = {"text": full_text_c, "time": elapsed_c, "success": True}

except Exception as e:
    print(f"  ✗ エラー: {e}")
    import traceback; traceback.print_exc()
    results["C"] = {"text": "", "time": 0, "success": False, "error": str(e)}

# ======================================================================
# 総合サマリー
# ======================================================================
print("\n\n" + "="*60)
print("総合サマリー")
print("="*60)
for label, r in results.items():
    if r["success"]:
        kanji = has_kanji(r["text"])
        kata = has_katakana(r["text"])
        print(f"モデル{label}: ✓ 成功 | 処理時間={r['time']:.1f}s | 漢字={'あり' if kanji else 'なし'} | カタカナ={'あり' if kata else 'なし'}")
    else:
        print(f"モデル{label}: ✗ 失敗 ({r.get('error','不明')})")
