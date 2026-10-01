"""Run CTC forced alignment locally without changing J-GRADE's evaluation path.

Usage: python -m jgrade_eval.alignment_experiment audio.wav
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .ctc_forced_alignment import CTCLogitChunk, align_ctc_chunks


MODEL_ID = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
_CHUNK_SECONDS = 30


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def analyze_audio(
    audio_path: Path,
    *,
    transcript_hiragana: str | None = None,
    sample_rate: int = 16_000,
) -> dict:
    """Return a versioned experiment report; never invoke the Judge."""

    from fluency import load_vumichien, transcribe

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    model, processor, device = load_vumichien()
    asr_text, all_logits, duration, audio = transcribe(
        model, processor, device, str(audio_path)
    )
    transcript = transcript_hiragana if transcript_hiragana is not None else "".join(asr_text.split())
    chunk_samples = sample_rate * _CHUNK_SECONDS
    chunks: list[CTCLogitChunk] = []
    for index, logits in enumerate(all_logits):
        start_sample = index * chunk_samples
        count = min(chunk_samples, len(audio) - start_sample)
        frame_logits = np.asarray(logits[0])
        chunks.append(CTCLogitChunk(start_sample, count, frame_logits))
    alignment = align_ctc_chunks(
        chunks, transcript, processor.tokenizer.get_vocab(),
        blank_id=processor.tokenizer.pad_token_id,
        sample_rate=sample_rate, model_id=MODEL_ID,
    )
    return {
        "source": {
            "audio_sha256": _hash_file(audio_path),
            "transcript_sha256": sha256(transcript.encode("utf-8")).hexdigest(),
            "model_id": MODEL_ID,
            "sample_rate": sample_rate,
            "duration_sec": duration,
        },
        "transcript_source": "supplied" if transcript_hiragana is not None else "asr",
        "asr_transcript_hiragana": "".join(asr_text.split()),
        "alignment": alignment.to_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimental Japanese CTC forced alignment")
    parser.add_argument("audio_path", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--transcript-file", type=Path,
                        help="UTF-8 hiragana transcript to align instead of ASR output")
    source.add_argument("--transcript-hiragana",
                        help="Hiragana transcript to align instead of ASR output")
    args = parser.parse_args(argv)
    transcript = args.transcript_hiragana
    if args.transcript_file is not None:
        transcript = args.transcript_file.read_text(encoding="utf-8").strip()
    report = analyze_audio(args.audio_path, transcript_hiragana=transcript)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
