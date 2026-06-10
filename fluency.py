"""
流暢度・音節分析
  - ひらがな + モーラタイミング : vumichien (Wav2Vec2)
  - ポーズ・発話区間            : silero-VAD

使い方:
  python fluency.py                 # audio/ 全件
  python fluency.py audio/foo.mp3
"""
import sys, os, time, argparse
import numpy as np
import librosa
import torch
import pykakasi
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from silero_vad import load_silero_vad, get_speech_timestamps

AUDIO_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
_SR        = 16000
_CHUNK_SEC = 30
_FRAME_SEC = 0.02

_kks = pykakasi.kakasi()

def to_romaji(text: str) -> str:
    return "".join(item["hepburn"] for item in _kks.convert(text))


# ── silero-VAD ────────────────────────────────────────────────────

def detect_speech_and_pauses(vad_model, audio_np, duration):
    wav = torch.FloatTensor(audio_np)
    speech_ts = get_speech_timestamps(wav, vad_model, sampling_rate=_SR,
                                      return_seconds=True)
    pauses = []
    prev_end = 0.0
    for seg in speech_ts:
        if seg["start"] - prev_end >= 0.15:
            pauses.append({"start": prev_end, "end": seg["start"],
                           "duration": seg["start"] - prev_end})
        prev_end = seg["end"]
    if duration - prev_end >= 0.15:
        pauses.append({"start": prev_end, "end": duration,
                       "duration": duration - prev_end})
    return speech_ts, pauses


# ── vumichien (Wav2Vec2) ──────────────────────────────────────────

def load_vumichien():
    model_id  = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model     = Wav2Vec2ForCTC.from_pretrained(model_id)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    model.to(device).eval()
    return model, processor, device


def _infer(model, processor, device, audio):
    inputs = processor(audio, sampling_rate=_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
    return text, logits.cpu()


def transcribe(model, processor, device, path):
    audio, _ = librosa.load(path, sr=_SR, mono=True)
    duration  = len(audio) / _SR
    if duration <= _CHUNK_SEC:
        text, logits = _infer(model, processor, device, audio)
        return text, [logits], duration, audio
    chunk_len = _SR * _CHUNK_SEC
    chunks = [audio[i:i + chunk_len] for i in range(0, len(audio), chunk_len)]
    texts, all_logits = [], []
    for chunk in chunks:
        t, l = _infer(model, processor, device, chunk)
        texts.append(t); all_logits.append(l)
    return "".join(texts), all_logits, duration, audio


def get_mora_timings(all_logits, blank_id, vocab):
    mora_timings = []
    offset = 0.0
    for logits in all_logits:
        probs    = logits.softmax(dim=-1)[0]
        pred_ids = probs.argmax(dim=-1).tolist()
        prev_id = start_frame = None
        for t, tid in enumerate(pred_ids):
            if tid == blank_id:
                if prev_id is not None:
                    mora = vocab[prev_id] if prev_id < len(vocab) else "?"
                    if mora and mora not in ("|", " ", ""):
                        mora_timings.append({"mora": mora,
                                             "start": offset + start_frame * _FRAME_SEC,
                                             "end":   offset + t * _FRAME_SEC})
                    prev_id = None
            elif tid != prev_id:
                if prev_id is not None:
                    mora = vocab[prev_id] if prev_id < len(vocab) else "?"
                    if mora and mora not in ("|", " ", ""):
                        mora_timings.append({"mora": mora,
                                             "start": offset + start_frame * _FRAME_SEC,
                                             "end":   offset + t * _FRAME_SEC})
                prev_id, start_frame = tid, t
        offset += probs.shape[0] * _FRAME_SEC
    return mora_timings


# ── 評価・可視化 ──────────────────────────────────────────────────

def grade(speech_pct, speed, max_pause):
    """流暢度グレード（S/A/B/C/D）"""
    if   speech_pct >= 80 and speed >= 7.0 and max_pause < 1.0: return "S"
    elif speech_pct >= 70 and speed >= 6.0 and max_pause < 1.5: return "A"
    elif speech_pct >= 60 and speed >= 5.0 and max_pause < 2.0: return "B"
    elif speech_pct >= 50 and speed >= 4.0:                      return "C"
    else:                                                         return "D"

def bar(ratio, width=30, fill="█", empty="░"):
    n = round(ratio * width)
    return fill * n + empty * (width - n)

def speed_timeline(speech_ts, mora_timings, duration):
    """1秒ごとの発話速度をブロック文字で表現"""
    symbols = {"fast": "■", "normal": "▪", "slow": "▫", "pause": "　"}
    result = []
    for sec in range(int(duration)):
        in_speech = any(s["start"] <= sec + 0.5 < s["end"] for s in speech_ts)
        if not in_speech:
            result.append("pause")
            continue
        moras_in_sec = sum(1 for m in mora_timings if sec <= m["start"] < sec + 1)
        if   moras_in_sec >= 8: result.append("fast")
        elif moras_in_sec >= 5: result.append("normal")
        else:                   result.append("slow")
    return "".join(symbols[s] for s in result)


# ── メイン ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    targets = args.files if args.files else sorted(
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR) if f.lower().endswith(AUDIO_EXTS)
    )
    if not targets:
        print("[エラー] 音声ファイルが見つかりません。"); sys.exit(1)

    print("▶ silero-VAD 読み込み中..."); vad = load_silero_vad(); print("  完了")
    print("▶ vumichien (Wav2Vec2) 読み込み中...")
    t0 = time.perf_counter()
    v_model, v_proc, v_device = load_vumichien()
    vocab    = v_proc.tokenizer.convert_ids_to_tokens(range(v_proc.tokenizer.vocab_size))
    blank_id = v_proc.tokenizer.pad_token_id or 0
    print(f"  完了 ({time.perf_counter()-t0:.1f}秒)  デバイス: {v_device}\n")

    for path in targets:
        filename = os.path.basename(path)
        print(f"  [{filename}] 処理中...", end=" ", flush=True)
        t0 = time.perf_counter()

        hiragana, all_logits, duration, audio = transcribe(v_model, v_proc, v_device, path)
        mora_timings = get_mora_timings(all_logits, blank_id, vocab)
        speech_ts, pauses = detect_speech_and_pauses(vad, audio, duration)

        speech_sec  = sum(s["end"] - s["start"] for s in speech_ts)
        pause_total = sum(p["duration"] for p in pauses)
        speech_pct  = speech_sec / duration * 100
        speed       = len(mora_timings) / duration
        max_pause   = max((p["duration"] for p in pauses), default=0)
        avg_pause   = pause_total / len(pauses) if pauses else 0
        g           = grade(speech_pct, speed, max_pause)
        speed_label = ("速い" if speed >= 9 else "普通" if speed >= 6
                       else "ゆっくり" if speed >= 4 else "かなりゆっくり")

        hiragana_clean = "".join(hiragana.split())
        romaji_clean   = to_romaji(hiragana_clean)
        timeline_str   = speed_timeline(speech_ts, mora_timings, duration)
        top_pauses     = sorted(pauses, key=lambda p: p["duration"], reverse=True)[:3]

        print(f"{time.perf_counter()-t0:.1f}秒")

        W = 64
        print(f"\n{'═'*W}")
        print(f"  {filename}")
        print(f"{'═'*W}")

        # ── 文字起こし ──────────────────────────────────────────
        print(f"\n  【ひらがな】（vumichien / Wav2Vec2）")
        print(f"  {hiragana_clean}")
        print(f"\n  【ローマ字】")
        print(f"  {romaji_clean}")

        # ── 総合評価 ────────────────────────────────────────────
        grade_desc = {"S":"ネイティブに近い流暢さ","A":"かなり流暢","B":"概ね流暢",
                      "C":"やや不流暢","D":"かなり不流暢"}
        print(f"\n  {'─'*W}")
        print(f"  総合評価 : {g}  （{grade_desc[g]}）")
        print(f"  {'─'*W}")
        speech_bar = bar(speech_sec / duration)
        print(f"  発話時間   {speech_sec:4.0f}秒  {speech_bar}  {speech_pct:.0f}%")
        pause_bar  = bar(pause_total / duration, fill="░", empty=" ")
        print(f"  無音・間   {pause_total:4.0f}秒  {pause_bar}  {100-speech_pct:.0f}%")
        print(f"  発話速度       {speed:.1f} モーラ/秒  →  {speed_label}")
        print(f"  認識音節数     {len(mora_timings)} モーラ")
        print(f"  詰まり回数     {len(pauses)} 回  （平均 {avg_pause:.1f}秒 / 最長 {max_pause:.1f}秒）")

        # ── 速度タイムライン ────────────────────────────────────
        print(f"\n  {'─'*W}")
        print(f"  発話速度タイムライン  ■=速い(8+)  ▪=普通(5-8)  ▫=遅い(<5)  　=無音")
        print(f"  {'─'*W}")
        # 10秒ごとに区切って表示
        for i in range(0, int(duration), 10):
            chunk = timeline_str[i:i+10]
            end   = min(i+10, int(duration))
            print(f"  {i:2d}〜{end:2d}秒 │{chunk}│")
        print(f"         0秒{'':6}10秒{'':5}20秒{'':5}30秒{'':5}40秒{'':5}50秒{'':5}60秒")

        # ── 長いポーズ ──────────────────────────────────────────
        if top_pauses:
            print(f"\n  {'─'*W}")
            print(f"  詰まりが長かった箇所")
            print(f"  {'─'*W}")
            for i, p in enumerate(top_pauses, 1):
                pause_bar2 = "━" * int(p["duration"] * 8)
                print(f"  {i}位  {p['start']:4.0f}秒あたり  {p['duration']:.1f}秒間  {pause_bar2}")

        # ── 発話区間ごとの音節 ──────────────────────────────────
        print(f"\n  {'─'*W}")
        print(f"  発話区間ごとの音節  （速度: ■8+ ▪5-8 ▫<5）")
        print(f"  {'─'*W}")
        print(f"  {'番号':4} {'時間帯':18} {'長さ':5} {'音節':5} {'速度':8} ひらがな")
        print(f"  {'':4} {'':18} {'':5} {'':5} {'':8} ローマ字")
        print(f"  {'─'*W}")
        for i, seg in enumerate(speech_ts, 1):
            seg_moras = [m for m in mora_timings
                         if seg["start"] <= m["start"] < seg["end"]]
            if not seg_moras:
                continue
            hira_seg   = "".join(m["mora"] for m in seg_moras)
            roma_seg   = to_romaji(hira_seg)
            seg_dur    = seg["end"] - seg["start"]
            mora_rate  = len(seg_moras) / seg_dur if seg_dur > 0 else 0
            spd_sym    = "■" if mora_rate >= 8 else "▪" if mora_rate >= 5 else "▫"
            print(f"  [{i:2}] {seg['start']:5.1f}〜{seg['end']:5.1f}秒"
                  f"  {seg_dur:4.1f}秒  {len(seg_moras):3}音節"
                  f"  {mora_rate:4.1f}/秒{spd_sym}"
                  f"  {hira_seg}")
            print(f"  {'':47}  {roma_seg}")
        print()


if __name__ == "__main__":
    main()
