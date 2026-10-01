"""Offline prepare/evaluate commands; no normal module or Judge path changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from .alignment_benchmark import evaluate_sample, make_reference_template, summarize
from .alignment_experiment import MODEL_ID, _hash_file
from .ctc_forced_alignment import CTCLogitChunk, align_ctc_chunks
from .evidence.alignment import build_transcript_alignment
from .evidence.models import MoraTiming, SpeechEvidence


def capture(audio_path: Path) -> dict:
    from fluency import load_vumichien, transcribe, get_mora_timings

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    before_hash = _hash_file(audio_path)
    model, processor, device = load_vumichien()
    text, logits, duration, audio = transcribe(model, processor, device, str(audio_path))
    if before_hash != _hash_file(audio_path):
        raise ValueError("Audio changed during capture")
    text = "".join(text.split())
    vocab = processor.tokenizer.get_vocab()
    labels = ["?"] * (max(vocab.values()) + 1)
    for label, token_id in vocab.items():
        labels[token_id] = label
    blank = processor.tokenizer.pad_token_id
    raw = get_mora_timings(logits, blank, labels)
    speech = SpeechEvidence(text, "", duration, (), (),
                            tuple(MoraTiming.from_dict(m) for m in raw), (),
                            (("stt_model", MODEL_ID),))
    chunks = [CTCLogitChunk(i * 480_000, min(480_000, len(audio) - i * 480_000),
                           np.asarray(chunk[0])) for i, chunk in enumerate(logits)]
    aligned = align_ctc_chunks(chunks, text, vocab, blank_id=blank,
                               sample_rate=16_000, model_id=MODEL_ID)
    libraries = {}
    for package in ("torch", "transformers", "numpy", "librosa"):
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None
    return dict(schema_version="alignment-predictions.v1", sample_id=audio_path.stem,
                source=dict(audio_sha256=before_hash, duration_sec=duration,
                            model_id=MODEL_ID, model_revision=getattr(getattr(model, "config", None),
                                                                     "_commit_hash", None),
                            sample_rate=16_000, libraries=libraries),
                transcript_hiragana=text, transcript_source="asr", baseline_raw_mora_timings=raw,
                methods=dict(ctc_argmax=build_transcript_alignment(speech).to_dict(),
                             ctc_viterbi=aligned.to_dict()))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def prepare(audio_paths: list[Path], output_dir: Path) -> dict:
    if not audio_paths or len({p.stem for p in audio_paths}) != len(audio_paths):
        raise ValueError("Provide recordings with unique, nonempty sample IDs")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("predictions", "references"):
        (output_dir / name).mkdir()
    samples, hashes = [], set()
    for audio in audio_paths:
        print(f"Capturing both timing methods: {audio.name}", file=sys.stderr)
        prediction = capture(audio)
        digest = prediction["source"]["audio_sha256"]
        if digest in hashes:
            raise ValueError("Duplicate audio content would inflate the benchmark")
        hashes.add(digest)
        reference = make_reference_template(prediction)
        # Use ordinal paths: sample IDs can contain arbitrary local filename characters.
        filename = f"{len(samples) + 1:03d}.json"
        _write(output_dir / "predictions" / filename, prediction)
        _write(output_dir / "references" / filename, reference)
        samples.append(dict(sample_id=prediction["sample_id"], audio_path=str(audio.resolve()),
                            prediction=f"predictions/{filename}", reference=f"references/{filename}"))
    manifest = dict(schema_version="alignment-dataset.v1", status="complete",
                    created_at=datetime.now(timezone.utc).isoformat(), samples=samples)
    _write(output_dir / "manifest.json", manifest)
    return manifest


def _read_local(directory: Path, relative: str) -> dict:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or Path(relative).is_absolute():
        raise ValueError("Manifest paths must remain inside the dataset directory")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(directory: Path) -> dict:
    manifest = _read_local(directory, "manifest.json")
    if manifest.get("schema_version") != "alignment-dataset.v1" or manifest.get("status") != "complete":
        raise ValueError("A complete supported capture manifest is required")
    if not manifest.get("samples"):
        raise ValueError("A benchmark requires at least one recording")
    results, hashes = [], set()
    for sample in manifest["samples"]:
        prediction = _read_local(directory, sample["prediction"])
        reference = _read_local(directory, sample["reference"])
        digest = prediction["source"]["audio_sha256"]
        if digest in hashes:
            raise ValueError("Duplicate audio in manifest")
        hashes.add(digest)
        results.append(evaluate_sample(reference, prediction))
    return summarize(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human-reference alignment benchmark (offline only)")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Capture both methods and unreviewed annotation templates")
    prep.add_argument("audio_paths", nargs="+", type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    evaluation = commands.add_parser("evaluate", help="Measure only human-reviewed references")
    evaluation.add_argument("dataset_dir", type=Path)
    evaluation.add_argument("--report", type=Path, help="Save a new JSON report (never overwrite)")
    args = parser.parse_args(argv)
    report = prepare(args.audio_paths, args.output_dir) if args.command == "prepare" else evaluate(args.dataset_dir)
    if args.command == "evaluate" and args.report is not None:
        with args.report.open("x", encoding="utf-8") as target:
            target.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
