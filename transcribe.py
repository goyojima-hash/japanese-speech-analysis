"""
ひらがな文字起こしツール（3モデル比較）

使い方:
  python transcribe.py 音声ファイル.mp3
  python transcribe.py 音声ファイル1.mp3 音声ファイル2.mp3 ...
  python transcribe.py  （引数なしで audio/ フォルダの全ファイルを処理）

対応フォーマット: MP3 / WAV / M4A / FLAC / OGG

モデル:
  🟢 A: sakasegawa  — ReazonSpeech 35,000時間学習・Dual CTC【推奨】
         音声を分割せず全体を一括処理 + SWD（精度向上デコード）
  🔵 B: vumichien   — XLSR-53ベース・日本語ひらがなCTCの定番
  🟡 C: slplab      — XLS-R 300Mベース・多言語対応バランス型
"""
import sys, os, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hiragana-asr"))

import librosa
import torch
from src.asr.model import load_checkpoint
from src.asr.kana_vocab import KanaVocab
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

# ---- パス設定 ----
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
HIRAGANA_ASR_DIR = os.path.join(BASE_DIR, "hiragana-asr")
CHECKPOINT       = os.path.join(HIRAGANA_ASR_DIR, "models/checkpoints/best-medium-ep5-inference.pt")
AUDIO_DIR        = os.path.join(BASE_DIR, "audio")
AUDIO_EXTS       = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
SR               = 16000

# モデルA: 全体を一括処理（分割なし）
# モデルB/C: 10秒チャンク（短くして精度向上）
CHUNK_SEC_BC = 10

# ---- 対象ファイルを決定 ----
args = sys.argv[1:]

if args:
    targets = []
    for a in args:
        path = a if os.path.isabs(a) else os.path.join(os.getcwd(), a)
        if not os.path.exists(path):
            alt = os.path.join(AUDIO_DIR, a)
            if os.path.exists(alt):
                path = alt
            else:
                print(f"[エラー] ファイルが見つかりません: {a}")
                sys.exit(1)
        targets.append(path)
else:
    targets = sorted(
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(AUDIO_EXTS)
    )
    if not targets:
        print(f"[エラー] {AUDIO_DIR} に音声ファイルが見つかりません。")
        print("使い方: python transcribe.py 音声ファイル.mp3")
        sys.exit(1)

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

print("=" * 62)
print("  ひらがな文字起こしツール（3モデル比較）")
print(f"  ハードウェア: {hw}")
print(f"  対象ファイル数: {len(targets)}")
print("=" * 62)

# ---- 3モデルをまとめて読み込む ----
print("\n各モデルを準備中...")

print("  🟢 モデルA (sakasegawa) 読み込み中...")
model_a    = load_checkpoint(CHECKPOINT)
model_a.to(device); model_a.eval()
feat_ext_a = Wav2Vec2FeatureExtractor.from_pretrained("reazon-research/japanese-wav2vec2-large")
kana_vocab = KanaVocab()

print("  🔵 モデルB (vumichien) 読み込み中...")
proc_b  = Wav2Vec2Processor.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
model_b = Wav2Vec2ForCTC.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
model_b.to(device); model_b.eval()

print("  🟡 モデルC (slplab) 読み込み中...")
proc_c  = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
model_c = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
model_c.to(device); model_c.eval()

print("  全モデル準備完了!\n")

# ---- ひらがなチェック ----
def has_kanji(text):
    return bool(re.search(r'[一-鿿㐀-䶿]', text))

def has_katakana(text):
    return bool(re.search(r'[゠-ヿ]', re.sub('ー', '', text)))

def purity(text):
    if has_kanji(text) or has_katakana(text):
        return "漢字/カタカナ混入あり ✗"
    return "ひらがなのみ ✓"

# ---- SWD（Spike Window Decoding）----
# CTCのスパイク（文字が予測される箇所）周辺だけを注目してデコードする精度向上手法
def swd_decode(logits, window=1):
    probs      = logits.squeeze(0).softmax(dim=-1)
    blank_prob = probs[:, 0]
    is_spike   = blank_prob < 0.5
    if not is_spike.any():
        return logits.squeeze(0).argmax(dim=-1)
    T            = probs.shape[0]
    spike_idx    = is_spike.nonzero(as_tuple=True)[0]
    active       = torch.zeros(T, dtype=torch.bool, device=logits.device)
    for idx in spike_idx:
        s = max(0, idx.item() - window)
        e = min(T, idx.item() + window + 1)
        active[s:e] = True
    pred_ids          = torch.zeros(T, dtype=torch.long, device=logits.device)
    pred_ids[active]  = logits.squeeze(0)[active].argmax(dim=-1)
    return pred_ids

# ---- 推論関数 ----

def infer_a(audio):
    """モデルA: 音声全体を一括処理（分割なし）+ SWD"""
    inputs = feat_ext_a(audio, sampling_rate=SR, return_tensors="pt", return_attention_mask=True)
    iv = inputs.input_values.to(device)
    am = inputs.attention_mask.to(device)
    with torch.no_grad():
        out      = model_a(iv, attention_mask=am)
        logits   = out["kana_logits"]
        pred_ids = swd_decode(logits)
    return kana_vocab.decode(pred_ids.tolist())

def infer_b(chunks):
    """モデルB: 10秒チャンクで処理"""
    texts = []
    for chunk in chunks:
        inputs = proc_b(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_b(iv).logits
        texts.append(proc_b.batch_decode(torch.argmax(logits, dim=-1))[0])
    return "".join(texts)

def infer_c(chunks):
    """モデルC: 10秒チャンクで処理"""
    texts = []
    for chunk in chunks:
        inputs = proc_c(chunk, sampling_rate=SR, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model_c(iv).logits
        texts.append(proc_c.batch_decode(torch.argmax(logits, dim=-1))[0])
    return "".join(texts)

MODELS = [
    ("🟢 A", "sakasegawa  (全体一括処理 + SWD)【推奨】", infer_a, "whole"),
    ("🔵 B", "vumichien   (XLSR-53ベース・10秒チャンク)",  infer_b, "chunk"),
    ("🟡 C", "slplab      (XLS-R 300Mベース・10秒チャンク)", infer_c, "chunk"),
]

# ---- 音声ファイルを1つずつ処理 ----
for path in targets:
    filename = os.path.basename(path)
    print(f"{'═' * 62}")
    print(f"  ファイル: {filename}")
    print(f"{'═' * 62}")

    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:
        print(f"  [エラー] 読み込み失敗: {e}\n")
        continue

    duration  = len(audio) / SR
    chunk_len = SR * CHUNK_SEC_BC
    chunks_bc = [audio[i:i+chunk_len] for i in range(0, len(audio), chunk_len)]
    print(f"  音声の長さ: {duration:.0f}秒\n")

    for label, desc, infer_fn, mode in MODELS:
        print(f"  {label} {desc}")
        t0 = time.perf_counter()
        try:
            if mode == "whole":
                full = infer_a(audio)
            else:
                full = infer_fn(chunks_bc)
            elapsed = time.perf_counter() - t0
            print(f"  【全文】 {full}")
            print(f"  処理時間: {elapsed:.1f}秒  ({duration/elapsed:.1f}倍速)  |  {purity(full)}")
        except Exception as e:
            print(f"  [エラー] {e}")
        print()

print("=" * 62)
print("  完了！")
print("=" * 62)
