"""Offline VAD overlap observations; not timing gold, correctness, or scores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from .alignment_benchmark import _number, _span
from .alignment_benchmark_cli import _read_local, _write, evaluate
from .alignment_experiment import _hash_file

VAD_CONFIG = {
    "threshold": 0.5,
    "sampling_rate": 16000,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "return_seconds": False,
}


def diagnose_alignment(
    alignment: dict,
    speech_segments: list[dict],
    *,
    duration_sec: float,
    transcript: str,
) -> dict:
    if not _number(duration_sec) or duration_sec <= 0:
        raise ValueError("Invalid recording duration")
    previous = 0.0
    gaps = []
    for speech in speech_segments:
        start, end = speech["start"], speech["end"]
        if (
            not _number(start)
            or not _number(end)
            or not previous <= start < end <= duration_sec
        ):
            raise ValueError("Invalid, overlapping or nonmonotonic VAD span")
        if start - previous >= 0.5:
            gaps.append({"start": previous, "end": start})
        previous = end
    if duration_sec - previous >= 0.5:
        gaps.append({"start": previous, "end": duration_sec})
    units = alignment.get("units", [])
    result = {
        "schema_version": "alignment-vad-observations.v1",
        "accuracy_verified": False,
        "status": "unavailable",
        "reason": alignment.get("reason") or "missing_alignment_units",
        "vad_status": "speech_detected"
        if speech_segments
        else "vad_detected_no_speech",
        "aligned_units": len(units),
        "outside_unit_count": None,
        "fully_outside_unit_count": None,
        "outside_speech_duration_sec": None,
        "long_pause_threshold_sec": 0.5,
        "long_pause_candidates": gaps,
        "observations": [],
    }
    if not units:
        return result
    try:
        span, _ = _span(
            alignment,
            {
                "start_offset": units[0]["start_offset"],
                "end_offset": units[-1]["end_offset"],
            },
            transcript,
            duration_sec,
        )
        valid = span is not None and all(
            unit["end_sec"] <= duration_sec for unit in units
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        result["reason"] = "invalid_alignment_geometry"
        return result
    observations = []
    for unit in units:
        start, end = unit["start_sec"], unit["end_sec"]
        length = end - start
        overlap = min(
            length,
            sum(
                max(0.0, min(end, s["end"]) - max(start, s["start"]))
                for s in speech_segments
            ),
        )
        outside = max(0.0, length - overlap)
        relation = (
            "fully_inside"
            if outside <= 1e-9
            else "fully_outside"
            if overlap <= 1e-9
            else "partially_outside"
        )
        observations.append(
            {
                **unit,
                "relation": relation,
                "speech_overlap_ratio": overlap / length,
                "outside_speech_duration_sec": outside,
                "crossed_long_pause_candidates": [
                    g for g in gaps if start < g["start"] and end > g["end"]
                ],
            }
        )
    result.update(
        status="observed",
        reason=None,
        observations=observations,
        outside_unit_count=sum(o["relation"] != "fully_inside" for o in observations),
        fully_outside_unit_count=sum(
            o["relation"] == "fully_outside" for o in observations
        ),
        outside_speech_duration_sec=sum(
            o["outside_speech_duration_sec"] for o in observations
        ),
    )
    return result


def run(dataset: Path, output: Path) -> dict:
    import librosa
    import silero_vad
    import torch

    evaluate(dataset)
    manifest = _read_local(dataset, "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    model = silero_vad.load_silero_vad()
    resource = Path(silero_vad.__file__).parent / "data" / "silero_vad.jit"
    samples = []
    for sample in manifest["samples"]:
        prediction = _read_local(dataset, sample["prediction"])
        audio_path = Path(sample["audio_path"])
        expected_hash = prediction["source"]["audio_sha256"]
        if _hash_file(audio_path) != expected_hash:
            raise ValueError("Original audio changed")
        audio, rate = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / rate
        if abs(duration - prediction["source"]["duration_sec"]) > 1 / rate:
            raise ValueError("Decoded audio duration does not match capture")
        raw_spans = silero_vad.get_speech_timestamps(
            torch.from_numpy(audio).float(), model, **VAD_CONFIG
        )
        if _hash_file(audio_path) != expected_hash:
            raise ValueError("Audio changed during VAD inference")
        spans = [
            {"start": s["start"] / rate, "end": s["end"] / rate} for s in raw_spans
        ]
        diagnostics = {
            name: diagnose_alignment(
                alignment,
                spans,
                duration_sec=duration,
                transcript=prediction["transcript_hiragana"],
            )
            for name, alignment in prediction["methods"].items()
        }
        samples.append(
            {
                "sample_id": prediction["sample_id"],
                "audio_sha256": expected_hash,
                "duration_sec": duration,
                "speech_segments": spans,
                "methods": diagnostics,
            }
        )
    report = {
        "schema_version": "alignment-vad-check.v1",
        "status": "complete",
        "accuracy_verified": False,
        "gold_modified": False,
        "source_dataset": str(dataset.resolve()),
        "vad_provenance": {
            "model": "silero-vad",
            "package_version": version("silero-vad"),
            "model_resource_sha256": _hash_file(resource)
            if resource.is_file()
            else None,
            "parameters": VAD_CONFIG,
        },
        "samples": samples,
    }
    _write(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Observe CTC timing overlaps with VAD; not accuracy"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.output_dir)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
