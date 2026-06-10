"""
J-GRADE 本番用ひらがな文字起こし（モデルA専用）
モデル: sakasegawa/japanese-wav2vec2-large-hiragana-ctc
    - ReazonSpeech 35,000時間学習・Dual CTC・LMなし
    - Apache-2.0 ライセンス（商用利用可）

【スクリプトとして使う】
  python transcriber_a.py audio.mp3
  python transcriber_a.py audio1.mp3 audio2.mp3 --out result.txt

【モジュールとして使う（API/バッチ統合）】
  from transcriber_a import HiraganaTranscriber

  tr = HiraganaTranscriber()                    # モデルは1回だけロード
  text = tr.transcribe("audio.mp3")             # ファイルパスから
  text = tr.transcribe_array(audio_np)          # NumPy配列から（16kHz・モノラル）
"""
import sys, os, re, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hiragana-asr"))

import librosa
import numpy as np
import torch
from src.asr.model import load_checkpoint
from src.asr.kana_vocab import KanaVocab
from transformers import Wav2Vec2FeatureExtractor

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT = os.path.join(_BASE_DIR, "hiragana-asr/models/checkpoints/best-medium-ep5-inference.pt")
_SR         = 16000
_CHUNK_SEC  = 30   # wav2vec2の最適処理長（30秒超は自動分割）


class HiraganaTranscriber:
    """モデルA（sakasegawa）専用ひらがな文字起こし。

    インスタンス化時にモデルをロードし、以降は transcribe() を何度でも呼べる。
    API サーバや長時間バッチでは起動時に1インスタンス作成して使い回すこと。
    """

    def __init__(self, device: str | None = None):
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._model = load_checkpoint(_CHECKPOINT)
        self._model.to(self.device)
        self._model.eval()
        self._feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(
            "reazon-research/japanese-wav2vec2-large"
        )
        self._vocab = KanaVocab()

    # ------------------------------------------------------------------ public

    def transcribe(self, path: str) -> str:
        """音声ファイルを読み込み、ひらがなテキストを返す。"""
        audio, _ = librosa.load(path, sr=_SR, mono=True)
        return self.transcribe_array(audio)

    def transcribe_array(self, audio: np.ndarray) -> str:
        """16kHz・モノラルの NumPy 配列からひらがなテキストを返す。"""
        duration = len(audio) / _SR
        if duration <= _CHUNK_SEC:
            return self._infer(audio)
        chunk_len = _SR * _CHUNK_SEC
        parts = [audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len)]
        return "".join(self._infer(p) for p in parts)

    # ----------------------------------------------------------------- private

    def _infer(self, audio: np.ndarray) -> str:
        inputs = self._feat_ext(
            audio, sampling_rate=_SR,
            return_tensors="pt", return_attention_mask=True,
        )
        iv = inputs.input_values.to(self.device)
        am = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            out = self._model(iv, attention_mask=am)
            pred_ids = _swd_decode(out["kana_logits"], device=self.device)
        return self._vocab.decode(pred_ids.tolist())


# -------------------------------------------------------------------- helpers

def _swd_decode(logits: torch.Tensor, window: int = 1, device=None) -> torch.Tensor:
    """Spike Window Decoding: CTCスパイク周辺に絞ってデコード（精度向上）。"""
    probs    = logits.squeeze(0).softmax(dim=-1)
    is_spike = probs[:, 0] < 0.5
    if not is_spike.any():
        return logits.squeeze(0).argmax(dim=-1)
    T      = probs.shape[0]
    active = torch.zeros(T, dtype=torch.bool, device=logits.device)
    for idx in is_spike.nonzero(as_tuple=True)[0]:
        s, e = max(0, idx.item() - window), min(T, idx.item() + window + 1)
        active[s:e] = True
    pred_ids          = torch.zeros(T, dtype=torch.long, device=logits.device)
    pred_ids[active]  = logits.squeeze(0)[active].argmax(dim=-1)
    return pred_ids


def _purity(text: str) -> str:
    if re.search(r'[一-鿿㐀-䶿]', text):
        return "漢字混入あり"
    if re.search(r'[゠-ヿ]', re.sub('ー', '', text)):
        return "カタカナ混入あり"
    return "ひらがなのみ ✓"


# -------------------------------------------------------------------- CLI

def _main():
    parser = argparse.ArgumentParser(
        description="ひらがな文字起こし（モデルA専用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="音声ファイル（複数可）")
    parser.add_argument("--out",    metavar="FILE", help="結果をテキストファイルに保存")
    parser.add_argument("--device", metavar="DEV",  help="使用デバイス（mps/cuda/cpu）")
    args = parser.parse_args()

    print("モデルA を準備中...")
    t_load = time.perf_counter()
    tr = HiraganaTranscriber(device=args.device)
    print(f"  準備完了 ({time.perf_counter() - t_load:.1f}秒)  デバイス: {tr.device}\n")

    output_lines = []
    for path in args.files:
        if not os.path.exists(path):
            print(f"[エラー] ファイルが見つかりません: {path}")
            continue

        filename = os.path.basename(path)
        audio, _ = librosa.load(path, sr=_SR, mono=True)
        duration  = len(audio) / _SR

        t0   = time.perf_counter()
        text = tr.transcribe_array(audio)
        elapsed = time.perf_counter() - t0

        print(f"[{filename}]")
        print(f"  {text}")
        print(f"  処理時間: {elapsed:.1f}秒 ({duration / elapsed:.1f}倍速) | {_purity(text)}\n")
        output_lines.append(f"# {filename}\n{text}\n")

    if args.out and output_lines:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"テキスト保存完了: {args.out}")


if __name__ == "__main__":
    _main()
