"""
デモ: 外国人日本語学習者の音声 → ひらがな文字起こし
モデルA (sakasegawa) のみ使用 — 最速・推奨モデル
"""
import sys, os, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hiragana-asr"))

import librosa
import torch
from src.asr.model import load_checkpoint
from src.asr.kana_vocab import KanaVocab
from transformers import Wav2Vec2FeatureExtractor

HIRAGANA_ASR_DIR = os.path.join(os.path.dirname(__file__), "hiragana-asr")
CHECKPOINT      = os.path.join(HIRAGANA_ASR_DIR, "models/checkpoints/best-medium-ep5-inference.pt")
AUDIO_DIR       = os.path.join(os.path.dirname(__file__), "audio")
SR              = 16000
CHUNK_SEC       = 30

AUDIO_FILES = [
    ("burmese_japanese_1min.mp3",     "ミャンマー人"),
    ("indonesian_japanese_1min.mp3",  "インドネシア人"),
    ("vietnamese_japanese_1min.mp3",  "ベトナム人"),
]

# ---- デバイス（処理に使うハードウェア）を自動選択 ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    hw = "Apple Silicon GPU"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    hw = "NVIDIA GPU"
else:
    device = torch.device("cpu")
    hw = "CPU"

print("=" * 60)
print("  ひらがなSTT（音声文字起こし）デモ")
print(f"  使用ハードウェア: {hw}")
print(f"  モデル: sakasegawa/japanese-wav2vec2-large-hiragana-ctc")
print("=" * 60)

# ---- モデルを読み込む（最初の1回だけ） ----
print("\nモデルを読み込み中...")
model = load_checkpoint(CHECKPOINT)
model.to(device)
model.eval()
feat_ext   = Wav2Vec2FeatureExtractor.from_pretrained("reazon-research/japanese-wav2vec2-large")
kana_vocab = KanaVocab()
print("  準備完了!\n")

# ---- 音声ファイルを1つずつ処理 ----
for filename, speaker in AUDIO_FILES:
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        print(f"[スキップ] {filename} が見つかりません\n")
        continue

    print(f"{'─' * 60}")
    print(f"  話者: {speaker}  /  ファイル: {filename}")
    print(f"{'─' * 60}")

    # 音声を読み込んで16kHz・モノラル（1チャンネル）に変換
    audio, _ = librosa.load(path, sr=SR, mono=True)
    duration  = len(audio) / SR
    print(f"  音声の長さ: {duration:.0f}秒")

    # 30秒ごとのかたまり（チャンク）に分割して処理
    chunk_len = SR * CHUNK_SEC
    chunks    = [audio[i:i+chunk_len] for i in range(0, len(audio), chunk_len)]

    texts = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks, 1):
        inputs    = feat_ext(chunk, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
        iv        = inputs.input_values.to(device)
        am        = inputs.attention_mask.to(device)
        with torch.no_grad():
            out      = model(iv, attention_mask=am)
            pred_ids = out["kana_logits"].squeeze(0).argmax(dim=-1)
        text = kana_vocab.decode(pred_ids.tolist())
        texts.append(text)
        print(f"  [{i}/{len(chunks)}] {text}")

    elapsed = time.perf_counter() - t0
    full    = "".join(texts)

    # 漢字・カタカナが混入していないかチェック
    has_kanji    = bool(re.search(r'[一-鿿]', full))
    has_katakana = bool(re.search(r'[゠-ヿ]', re.sub('ー', '', full)))

    print(f"\n  【全文】 {full}")
    print(f"  処理時間: {elapsed:.1f}秒  (音声{duration:.0f}秒を{duration/elapsed:.1f}倍速で処理)")
    print(f"  漢字混入: {'あり ✗' if has_kanji else 'なし ✓'}  /  カタカナ混入: {'あり ✗' if has_katakana else 'なし ✓'}")
    print()

print("=" * 60)
print("  デモ完了！")
print("=" * 60)
