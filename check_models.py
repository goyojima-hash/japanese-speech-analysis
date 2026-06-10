"""フェーズ1: 3モデル読み込み確認スクリプト"""
import sys
import os

# hiragana-asr の src を参照できるようにパスを通す
HIRAGANA_ASR_DIR = os.path.join(os.path.dirname(__file__), "hiragana-asr")
sys.path.insert(0, HIRAGANA_ASR_DIR)

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

results = {}

# ---- モデルA: sakasegawa (カスタム .pt チェックポイント) ----
print("\n" + "="*60)
print("モデルA: sakasegawa/japanese-wav2vec2-large-hiragana-ctc")
print("="*60)
try:
    from src.asr.model import load_checkpoint
    from src.asr.kana_vocab import KanaVocab
    CHECKPOINT = os.path.join(HIRAGANA_ASR_DIR, "models/checkpoints/best-medium-ep5-inference.pt")
    print(f"  → チェックポイント読み込み中: {CHECKPOINT}")
    model_a = load_checkpoint(CHECKPOINT)
    model_a.eval()
    kana_vocab = KanaVocab()
    print(f"  ✓ 読み込み成功")
    print(f"  ✓ Vocab size (kana): {kana_vocab.size}")
    print(f"  ✓ パラメータ数: {sum(p.numel() for p in model_a.parameters()):,}")
    results["A"] = "SUCCESS"
except Exception as e:
    print(f"  ✗ エラー: {e}")
    results["A"] = f"FAILED: {e}"

# ---- モデルB: slplab ----
print("\n" + "="*60)
print("モデルB: slplab/wav2vec2-xls-r-300m-japanese-hiragana")
print("="*60)
try:
    print("  → Processor読み込み中...")
    proc_b = Wav2Vec2Processor.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
    print("  → Model読み込み中...")
    from transformers import Wav2Vec2ForCTC
    model_b = Wav2Vec2ForCTC.from_pretrained("slplab/wav2vec2-xls-r-300m-japanese-hiragana")
    model_b.eval()
    vocab_b = proc_b.tokenizer.get_vocab()
    print(f"  ✓ 読み込み成功")
    print(f"  ✓ Vocab size: {len(vocab_b)}")
    print(f"  ✓ パラメータ数: {sum(p.numel() for p in model_b.parameters()):,}")
    results["B"] = "SUCCESS"
except Exception as e:
    print(f"  ✗ エラー: {e}")
    results["B"] = f"FAILED: {e}"

# ---- モデルC: vumichien ----
print("\n" + "="*60)
print("モデルC: vumichien/wav2vec2-large-xlsr-japanese-hiragana")
print("="*60)
try:
    print("  → Processor読み込み中...")
    proc_c = Wav2Vec2Processor.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
    print("  → Model読み込み中...")
    from transformers import Wav2Vec2ForCTC
    model_c = Wav2Vec2ForCTC.from_pretrained("vumichien/wav2vec2-large-xlsr-japanese-hiragana")
    model_c.eval()
    vocab_c = proc_c.tokenizer.get_vocab()
    print(f"  ✓ 読み込み成功")
    print(f"  ✓ Vocab size: {len(vocab_c)}")
    print(f"  ✓ パラメータ数: {sum(p.numel() for p in model_c.parameters()):,}")
    results["C"] = "SUCCESS"
except Exception as e:
    print(f"  ✗ エラー: {e}")
    results["C"] = f"FAILED: {e}"

# ---- サマリー ----
print("\n\n=== 読み込み結果サマリー ===")
for label, status in results.items():
    mark = "✓" if status == "SUCCESS" else "✗"
    print(f"  {mark} モデル{label}: {status}")

# ---- デバイス確認 ----
print("\n=== デバイス確認 ===")
if torch.backends.mps.is_available():
    print("  → Apple Silicon GPU (MPS) 使用可能")
elif torch.cuda.is_available():
    print(f"  → CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  → CPU使用")
