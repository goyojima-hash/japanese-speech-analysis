"""Compare model timing proposals, never manufacture human gold or CEFR scores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from .alignment_benchmark import _span
from .alignment_benchmark_cli import _read_local, _write, evaluate
from .alignment_experiment import _hash_file
from .ctc_forced_alignment import CTCLogitChunk, ForcedAlignmentResult, align_ctc_chunks

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
MODEL_REVISION = "cf031e020336460d15a417eba710bbc5bb43be9a"


def load_comparison_model():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, weights_only=True
    )
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device).eval()
    return model, processor, device


def align_audio(path: Path, transcript: str, resources: tuple) -> dict:
    from importlib.metadata import version

    import librosa
    import torch

    digest = _hash_file(path)
    audio, rate = librosa.load(path, sr=16000, mono=True)
    if _hash_file(path) != digest:
        raise ValueError("Audio changed during decoding")
    model, processor, device = resources
    chunks = []
    too_short = False
    for start in range(0, len(audio), 480000):
        clip = audio[start : start + 480000]
        if len(clip) < 400:
            too_short = True
            break
        inputs = processor(clip, sampling_rate=rate, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(
                **{name: tensor.to(device) for name, tensor in inputs.items()}
            ).logits
        chunks.append(CTCLogitChunk(start, len(clip), logits[0].detach().cpu().numpy()))
    alignment = (
        ForcedAlignmentResult(
            transcript, (), "unavailable", "audio_chunk_too_short", MODEL_ID
        )
        if too_short
        else align_ctc_chunks(
            chunks,
            transcript,
            processor.tokenizer.get_vocab(),
            blank_id=processor.tokenizer.pad_token_id,
            sample_rate=rate,
            model_id=MODEL_ID,
        )
    )
    if _hash_file(path) != digest:
        raise ValueError("Audio changed during inference")
    return {
        "schema_version": "cross-model-alignment.v1",
        "source": {
            "audio_sha256": digest,
            "duration_sec": len(audio) / rate,
            "sample_rate": rate,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": str(device),
            "libraries": {
                p: version(p) for p in ("torch", "transformers", "numpy", "librosa")
            },
        },
        "transcript_hiragana": transcript,
        "alignment": alignment.to_dict(),
    }


def compare_candidates(primary: dict, alternate: dict, segments: list[dict]) -> dict:
    if primary["transcript_hiragana"] != alternate["transcript_hiragana"] or any(
        primary["source"][key] != alternate["source"][key]
        for key in ("audio_sha256", "duration_sec")
    ):
        raise ValueError("Only identical audio and text can be compared")
    text, duration = primary["transcript_hiragana"], primary["source"]["duration_sec"]
    items, ids = [], set()
    for segment in segments:
        a, b = segment["start_offset"], segment["end_offset"]
        sid = segment["segment_id"]
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b <= len(text)
            or sid in ids
        ):
            raise ValueError("Invalid or duplicate candidate segment")
        ids.add(sid)
        first, reason1 = _span(
            primary["methods"].get("ctc_viterbi", {}), segment, text, duration
        )
        second, reason2 = _span(alternate["alignment"], segment, text, duration)
        item = {
            "sample_id": primary["sample_id"],
            "segment_id": sid,
            "start_offset": a,
            "end_offset": b,
            "transcript_text": text[a:b],
            "needs_human_review": True,
            "status": "compared" if first and second else "unavailable",
            "primary_reason": reason1,
            "alternate_reason": reason2,
            "start_delta_ms": None,
            "end_delta_ms": None,
            "max_abs_delta_ms": None,
        }
        if first and second:
            start, end = (second[0] - first[0]) * 1000, (second[1] - first[1]) * 1000
            item.update(
                start_delta_ms=start,
                end_delta_ms=end,
                max_abs_delta_ms=max(abs(start), abs(end)),
            )
        items.append(item)
    differences = [
        abs(i[k])
        for i in items
        if i["status"] == "compared"
        for k in ("start_delta_ms", "end_delta_ms")
    ]
    return {
        "schema_version": "model-timing-agreement.v1",
        "accuracy_verified": False,
        "eligible_segments": len(items),
        "compared_segments": sum(i["status"] == "compared" for i in items),
        "median_abs_boundary_delta_ms": median(differences) if differences else None,
        "max_abs_boundary_delta_ms": max(differences) if differences else None,
        "items": sorted(
            items,
            key=lambda i: (i["status"] == "compared", -(i["max_abs_delta_ms"] or 0)),
        ),
    }


def run(dataset: Path, output: Path) -> dict:
    evaluate(dataset)
    manifest = _read_local(dataset, "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    resources = load_comparison_model()
    reports = []
    for index, sample in enumerate(manifest["samples"]):
        primary = _read_local(dataset, sample["prediction"])
        ref = _read_local(dataset, sample["reference"])
        if (
            ref["transcript_hiragana"] != primary["transcript_hiragana"]
            or ref["audio_sha256"] != primary["source"]["audio_sha256"]
        ):
            raise ValueError("Reference source mismatch")
        audio = Path(sample["audio_path"])
        if _hash_file(audio) != primary["source"]["audio_sha256"]:
            raise ValueError("Original audio changed")
        alternate = align_audio(audio, primary["transcript_hiragana"], resources)
        comparison = compare_candidates(primary, alternate, ref["segments"])
        _write(output / f"alignment-{index + 1:03d}.json", alternate)
        reports.append(comparison)
    items = sorted(
        [i for r in reports for i in r["items"]],
        key=lambda i: (i["status"] == "compared", -(i["max_abs_delta_ms"] or 0)),
    )
    report = {
        "schema_version": "cross-model-review-priority.v1",
        "status": "complete",
        "accuracy_verified": False,
        "gold_modified": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_dataset": str(dataset.resolve()),
        "eligible_segments": len(items),
        "compared_segments": sum(r["compared_segments"] for r in reports),
        "samples": reports,
        "review_priority": items,
    }
    _write(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Japanese timing models, not human-reference accuracy"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.output_dir)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("samples", "review_priority")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
