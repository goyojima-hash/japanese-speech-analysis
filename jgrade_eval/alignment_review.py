"""Export local listening aids without approving or modifying gold references."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from math import ceil, floor, isfinite
from pathlib import Path

from .alignment_benchmark_cli import _read_local, _write, evaluate
from .alignment_experiment import _hash_file


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _csv_safe(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def export_review(dataset: Path, output: Path, *, padding_sec: float = 1.0) -> dict:
    """Write clips and a blank review sheet; original dataset is read-only."""
    import librosa
    import soundfile as sf

    if not _finite_number(padding_sec) or padding_sec < 0:
        raise ValueError("padding_sec must be finite and nonnegative")
    evaluate(dataset)
    manifest = _read_local(dataset, "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "clips").mkdir()
    items = []
    for sample in manifest["samples"]:
        pred = _read_local(dataset, sample["prediction"])
        ref = _read_local(dataset, sample["reference"])
        if (
            sample["sample_id"] != pred["sample_id"]
            or ref["sample_id"] != pred["sample_id"]
        ):
            raise ValueError("Sample identity mismatch")
        if (
            ref["audio_sha256"] != pred["source"]["audio_sha256"]
            or ref["transcript_hiragana"] != pred["transcript_hiragana"]
        ):
            raise ValueError("Reference and prediction source mismatch")
        source = Path(sample["audio_path"])
        if not source.is_absolute():
            raise ValueError("Original audio path must be absolute")
        if _hash_file(source) != pred["source"]["audio_sha256"]:
            raise ValueError(f"Original audio changed: {source.name}")
        wave, rate = librosa.load(source, sr=16000, mono=True)
        if _hash_file(source) != pred["source"]["audio_sha256"]:
            raise ValueError("Audio changed during decoding")
        duration = len(wave) / rate
        if abs(duration - pred["source"]["duration_sec"]) > 1 / rate:
            raise ValueError("Decoded audio duration does not match capture")
        for segment in ref["segments"]:
            a, b = segment["start_offset"], segment["end_offset"]
            if (
                type(a) is not int
                or type(b) is not int
                or not 0 <= a < b <= len(ref["transcript_hiragana"])
            ):
                raise ValueError("Invalid candidate text offsets")
            start, end = (
                segment.get("suggested_start_sec"),
                segment.get("suggested_end_sec"),
            )
            row = {
                "sample_id": sample["sample_id"],
                "segment_id": segment["segment_id"],
                "audio_path": str(source),
                "start_offset": a,
                "end_offset": b,
                "transcript_text": ref["transcript_hiragana"][a:b],
                "suggested_start_sec": start,
                "suggested_end_sec": end,
                "clip_path": None,
                "clip_origin_sec": None,
                "clip_end_sec": None,
                "gold_start_sec": None,
                "gold_end_sec": None,
                "reviewed_by": None,
                "review_status": "pending",
                "note": "Listen to original audio if boundaries fall outside clip",
            }
            if start is None and end is None:
                row["note"] = "No timing proposal; listen to original audio"
            else:
                if (
                    not _finite_number(start)
                    or not _finite_number(end)
                    or not 0 <= start < end <= duration
                ):
                    raise ValueError("Invalid candidate timing")
                low = max(0, floor((start - padding_sec) * rate))
                high = min(len(wave), ceil((end + padding_sec) * rate))
                relative = f"clips/{len(items) + 1:04d}.wav"
                sf.write(output / relative, wave[low:high], rate, subtype="PCM_16")
                row.update(
                    clip_path=relative,
                    clip_origin_sec=low / rate,
                    clip_end_sec=high / rate,
                )
            items.append(row)
    fields = [
        "sample_id",
        "segment_id",
        "audio_path",
        "start_offset",
        "end_offset",
        "transcript_text",
        "suggested_start_sec",
        "suggested_end_sec",
        "clip_path",
        "clip_origin_sec",
        "clip_end_sec",
        "gold_start_sec",
        "gold_end_sec",
        "reviewed_by",
        "review_status",
        "note",
    ]
    with (output / "review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: _csv_safe(value) for key, value in item.items()} for item in items
        )
    report = {
        "schema_version": "alignment-listening-aid.v1",
        "status": "complete",
        "source_dataset": str(dataset.resolve()),
        "padding_sec": padding_sec,
        "clip_count": sum(item["clip_path"] is not None for item in items),
        "gold_modified": False,
        "items": items,
    }
    _write(output / "manifest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export unverified listening clips for human timing review"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding-sec", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = export_review(args.dataset, args.output_dir, padding_sec=args.padding_sec)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "items"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
