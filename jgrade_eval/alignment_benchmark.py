"""Human-reference timing measurements. Never treats machine proposals as gold."""

from __future__ import annotations

import itertools
from math import isfinite
from statistics import mean, median

TOLERANCES_MS = (50, 100, 200, 500)


def make_reference_template(prediction: dict, gap_sec: float = 0.5) -> dict:
    """Propose gaps for listening, not utterance/speaker/question boundaries."""
    units = prediction["methods"].get("ctc_viterbi", {}).get("units", [])
    groups: list[list[dict]] = []
    for unit in units:
        if not groups or unit["start_sec"] - groups[-1][-1]["end_sec"] >= gap_sec:
            groups.append([])
        groups[-1].append(unit)
    if not groups and prediction["transcript_hiragana"]:
        groups = [
            [
                {
                    "start_offset": 0,
                    "end_offset": len(prediction["transcript_hiragana"]),
                    "start_sec": None,
                    "end_sec": None,
                }
            ]
        ]
    return {
        "schema_version": "alignment-reference.v1",
        "sample_id": prediction["sample_id"],
        "audio_sha256": prediction["source"]["audio_sha256"],
        "duration_sec": prediction["source"]["duration_sec"],
        "recording_type": "unknown",
        "condition_tags": [],
        "transcript_hiragana": prediction["transcript_hiragana"],
        "transcript_review": {"status": "pending", "source": None, "reviewed_by": None},
        "segments": [
            {
                "segment_id": f"u{i + 1:03d}",
                "kind": "utterance_candidate",
                "start_offset": group[0]["start_offset"],
                "end_offset": group[-1]["end_offset"],
                "start_sec": None,
                "end_sec": None,
                "review_status": "pending",
                "review_source": None,
                "reviewed_by": None,
                "proposal_source": "ctc_gap_candidate",
                "suggested_start_sec": group[0]["start_sec"],
                "suggested_end_sec": group[-1]["end_sec"],
            }
            for i, group in enumerate(groups)
        ],
    }


def _number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _human(review: dict, status: str, source: str) -> bool:
    if review.get(status) != "reviewed":
        if review.get(status) != "pending":
            raise ValueError("Review status must be pending or reviewed")
        return False
    if (
        review.get(source) != "human"
        or not isinstance(review.get("reviewed_by"), str)
        or not review["reviewed_by"].strip()
    ):
        raise ValueError("Reviewed gold requires human source and a named reviewer")
    return True


def _validate_gold(ref: dict) -> list[dict]:
    if ref.get("schema_version") != "alignment-reference.v1":
        raise ValueError("Unsupported reference schema")
    duration, text = ref["duration_sec"], ref["transcript_hiragana"]
    if not _number(duration) or duration <= 0 or not isinstance(text, str):
        raise ValueError("Invalid reference duration or transcript")
    transcript_reviewed = _human(ref["transcript_review"], "status", "source")
    eligible, ids = [], set()
    previous_offset, previous_end = 0, 0.0
    for segment in ref["segments"]:
        sid = segment["segment_id"]
        if not isinstance(sid, str) or not sid or sid in ids:
            raise ValueError("Reference segment IDs must be unique and nonempty")
        ids.add(sid)
        if not _human(segment, "review_status", "review_source"):
            continue
        a, b = segment["start_offset"], segment["end_offset"]
        start, end = segment["start_sec"], segment["end_sec"]
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b <= len(text)
            or not _number(start)
            or not _number(end)
            or not 0 <= start < end <= duration
            or a < previous_offset
            or start < previous_end
        ):
            raise ValueError(f"Invalid or overlapping reviewed segment: {sid}")
        previous_offset, previous_end = b, end
        if transcript_reviewed:
            eligible.append(segment)
    return eligible


def _span(method: dict, segment: dict, transcript: str, duration: float):
    units = method.get("units", [])
    previous_offset, previous_end = 0, 0.0
    for unit in units:
        try:
            a, b, start, end = (
                unit[k] for k in ("start_offset", "end_offset", "start_sec", "end_sec")
            )
            valid = (
                type(a) is int
                and type(b) is int
                and 0 <= a < b <= len(transcript)
                and unit["text"] == transcript[a:b]
                and _number(start)
                and _number(end)
                and 0 <= start < end <= duration + 0.05
                and a >= previous_offset
                and start >= previous_end
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            return None, "invalid_prediction_units"
        previous_offset, previous_end = b, end
    selected = [
        u
        for u in units
        if u["start_offset"] >= segment["start_offset"]
        and u["end_offset"] <= segment["end_offset"]
    ]
    if (
        not selected
        or selected[0]["start_offset"] != segment["start_offset"]
        or selected[-1]["end_offset"] != segment["end_offset"]
        or any(
            a["end_offset"] != b["start_offset"]
            for a, b in itertools.pairwise(selected)
        )
    ):
        return None, "missing_units"
    return (selected[0]["start_sec"], selected[-1]["end_sec"]), None


def _distribution(values: list[float]) -> dict:
    absolute = sorted(abs(value) for value in values)
    p95 = None
    if absolute:
        position = (len(absolute) - 1) * 0.95
        low = int(position)
        p95 = absolute[low] + (
            absolute[min(low + 1, len(absolute) - 1)] - absolute[low]
        ) * (position - low)
    return {
        "count": len(values),
        "mae_ms": mean(absolute) if absolute else None,
        "median_ms": median(absolute) if absolute else None,
        "p95_ms": p95,
        "signed_mean_ms": mean(values) if values else None,
    }


def _metrics(scored: list[dict], abstentions: list[dict], eligible: int) -> dict:
    starts = [s["start_error_ms"] for s in scored]
    ends = [s["end_error_ms"] for s in scored]
    boundaries = starts + ends
    outliers = sum(abs(value) > 1000 for value in boundaries)
    return {
        "eligible_segments": eligible,
        "matched_segments": len(scored),
        "coverage": len(scored) / eligible if eligible else None,
        "start": _distribution(starts),
        "end": _distribution(ends),
        "boundaries": _distribution(boundaries),
        "outliers_over_1s": outliers,
        "outlier_matched_boundary_rate": outliers / len(boundaries)
        if boundaries
        else None,
        "tolerance_ms": {
            str(t): {
                "matched_boundary_rate": sum(abs(v) <= t + 1e-7 for v in boundaries)
                / len(boundaries)
                if boundaries
                else None,
                "all_reference_boundary_rate": sum(
                    abs(v) <= t + 1e-7 for v in boundaries
                )
                / (2 * eligible)
                if eligible
                else None,
            }
            for t in TOLERANCES_MS
        },
        "scored": scored,
        "abstentions": abstentions,
    }


def evaluate_sample(reference: dict, prediction: dict) -> dict:
    eligible = _validate_gold(reference)
    compatible = (
        reference["audio_sha256"] == prediction["source"]["audio_sha256"]
        and reference["transcript_hiragana"] == prediction["transcript_hiragana"]
        and reference["sample_id"] == prediction["sample_id"]
        and reference["duration_sec"] == prediction["source"]["duration_sec"]
    )
    methods = {}
    for name in sorted(set(prediction["methods"]) | {"ctc_argmax", "ctc_viterbi"}):
        method = prediction["methods"].get(name, {})
        scored, abstentions = [], []
        for segment in eligible:
            span, reason = (
                _span(
                    method,
                    segment,
                    reference["transcript_hiragana"],
                    reference["duration_sec"],
                )
                if compatible
                else (None, "source_mismatch")
            )
            if span is None:
                abstentions.append(
                    {"segment_id": segment["segment_id"], "reason": reason}
                )
            else:
                scored.append(
                    {
                        "segment_id": segment["segment_id"],
                        "start_error_ms": (span[0] - segment["start_sec"]) * 1000,
                        "end_error_ms": (span[1] - segment["end_sec"]) * 1000,
                    }
                )
        methods[name] = _metrics(scored, abstentions, len(eligible))
    status = (
        "incompatible_source"
        if not compatible
        else "pending_reference"
        if not eligible
        else "measured"
        if any(m["matched_segments"] for m in methods.values())
        else "no_usable_predictions"
    )
    return {
        "sample_id": reference["sample_id"],
        "recording_type": reference.get("recording_type", "unknown"),
        "status": status,
        "pending_segments": len(reference["segments"]) - len(eligible),
        "methods": methods,
    }


def summarize(results: list[dict]) -> dict:
    if len({r["sample_id"] for r in results}) != len(results):
        raise ValueError("Duplicate samples would inflate the benchmark")
    names = sorted({name for result in results for name in result["methods"]})

    def aggregate(subset):
        return {
            name: _metrics(
                [
                    s
                    for r in subset
                    for s in r["methods"].get(name, {}).get("scored", [])
                ],
                [
                    s
                    for r in subset
                    for s in r["methods"].get(name, {}).get("abstentions", [])
                ],
                sum(
                    r["methods"].get(name, {}).get("eligible_segments", 0)
                    for r in subset
                ),
            )
            for name in names
        }

    return {
        "schema_version": "alignment-benchmark-report.v1",
        "samples": results,
        "pending_samples": sum(r["status"] == "pending_reference" for r in results),
        "methods": aggregate(results),
        "by_recording_type": {
            kind: aggregate([r for r in results if r["recording_type"] == kind])
            for kind in sorted({r["recording_type"] for r in results})
        },
    }
