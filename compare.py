"""
ひらがな × 自然な日本語 比較

ひらがなモデル（--hiragana-model で切り替え）:
  sakasegawa  : ReazonSpeech 35,000時間学習、幻覚なし（デフォルト）
  vumichien   : XLSR-53 多言語ベース、外国語訛りに強い可能性あり

日本語テキスト: Whisper large-v3

使い方:
  python compare.py                                 # sakasegawa（デフォルト）
  python compare.py --hiragana-model vumichien      # vumichien に切り替え
  python compare.py audio/*.mp3 --out result.txt
"""
import sys, os, time, argparse, gc
import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel

AUDIO_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
_SR        = 16000
_CHUNK_SEC = 30


# ── ひらがなモデル: sakasegawa ────────────────────────────────────
def load_sakasegawa():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hiragana-asr"))
    from transcriber_a import HiraganaTranscriber
    return HiraganaTranscriber()

def transcribe_sakasegawa(tr, path: str) -> str:
    return tr.transcribe(path)


# ── ひらがなモデル: vumichien ─────────────────────────────────────
def load_vumichien():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    model_id = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    return {"model": model, "processor": processor, "device": device}

def _vumichien_infer(ctx, audio: np.ndarray) -> str:
    inputs = ctx["processor"](audio, sampling_rate=_SR, return_tensors="pt", padding=True)
    iv = inputs.input_values.to(ctx["device"])
    with torch.no_grad():
        logits = ctx["model"](iv).logits
    ids = torch.argmax(logits, dim=-1)
    return ctx["processor"].batch_decode(ids)[0]

def transcribe_vumichien(ctx, path: str) -> str:
    audio, _ = librosa.load(path, sr=_SR, mono=True)
    duration = len(audio) / _SR
    if duration <= _CHUNK_SEC:
        return _vumichien_infer(ctx, audio)
    chunk_len = _SR * _CHUNK_SEC
    parts = [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len)]
    return "".join(_vumichien_infer(ctx, p) for p in parts)


# ── Whisper ───────────────────────────────────────────────────────
def whisper_transcribe(model: WhisperModel, path: str) -> str:
    segments, _ = model.transcribe(path, language="ja", beam_size=5, vad_filter=True)
    return "".join("".join(seg.text.split()) for seg in segments)


# ── メイン ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="音声ファイル（省略で audio/ 全件）")
    parser.add_argument("--out", metavar="FILE", help="結果をテキスト保存")
    parser.add_argument("--hiragana-model", choices=["sakasegawa", "vumichien"],
                        default="vumichien", dest="hiragana_model",
                        help="ひらがなモデル選択（デフォルト: vumichien）")
    args = parser.parse_args()

    targets = args.files if args.files else sorted(
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(AUDIO_EXTS)
    )
    if not targets:
        print("[エラー] 音声ファイルが見つかりません。")
        sys.exit(1)

    # ── Phase 1: ひらがな文字起こし ──────────────────────────────
    print(f"▶ Phase 1: ひらがな文字起こし（{args.hiragana_model}）")
    print("─" * 64)
    t0 = time.perf_counter()
    if args.hiragana_model == "sakasegawa":
        tr = load_sakasegawa()
        print(f"  ロード完了 ({time.perf_counter() - t0:.1f}秒)  デバイス: {tr.device}\n")
        do_transcribe = lambda path: transcribe_sakasegawa(tr, path)
    else:
        ctx = load_vumichien()
        print(f"  ロード完了 ({time.perf_counter() - t0:.1f}秒)  デバイス: {ctx['device']}\n")
        do_transcribe = lambda path: transcribe_vumichien(ctx, path)

    transcriptions = []
    for i, path in enumerate(targets, 1):
        filename = os.path.basename(path)
        print(f"  [{i}/{len(targets)}] {filename} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        hiragana = do_transcribe(path)
        print(f"{time.perf_counter() - t0:.1f}秒")
        transcriptions.append({"file": filename, "hiragana": hiragana})

    # GPU メモリ解放
    if args.hiragana_model == "sakasegawa":
        del tr
    else:
        del ctx
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Phase 2: Whisper で日本語テキスト ────────────────────────
    print("\n▶ Phase 2: 日本語テキスト変換（Whisper large-v3）")
    print("─" * 64)
    t0 = time.perf_counter()
    whisper = WhisperModel("large-v3", device="auto", compute_type="auto")
    print(f"  ロード完了 ({time.perf_counter() - t0:.1f}秒)\n")

    for item in transcriptions:
        path = next(p for p in targets if os.path.basename(p) == item["file"])
        print(f"  [{item['file']}] ...", end=" ", flush=True)
        t0 = time.perf_counter()
        item["japanese"] = whisper_transcribe(whisper, path)
        print(f"{time.perf_counter() - t0:.1f}秒")

    # ── 結果表示 ──────────────────────────────────────────────────
    for r in transcriptions:
        print(f"\n{'═' * 64}")
        print(f"  {r['file']}")
        print(f"{'═' * 64}")
        print(f"\n  【ひらがな（{args.hiragana_model}）】")
        print(f"  {r['hiragana']}")
        print(f"\n  【日本語テキスト（Whisper）】")
        print(f"  {r['japanese']}")

    if args.out:
        lines = []
        for r in transcriptions:
            lines += [
                f"# {r['file']}",
                "",
                f"[ひらがな ({args.hiragana_model})]",
                r["hiragana"],
                "",
                "[日本語テキスト (Whisper)]",
                r["japanese"],
                "",
                "─" * 40,
                "",
            ]
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n保存完了: {args.out}")


if __name__ == "__main__":
    main()
