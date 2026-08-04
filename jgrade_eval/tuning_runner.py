from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .audio_pipeline import FluencyExtractor
from .consensus import aggregate_task_rating
from .deliberation import deliberate_auto_cefr
from .live_judges import (
    JudgeFailure,
    ProviderSpec,
    judge_auto_cefr_with_live_panel,
    judge_auto_cefr_with_live_panel_partial,
    load_env_file,
)
from .mock_judges import judge_auto_cefr_with_mock_panel
from .models import CEFR_LEVELS, AutoLevelJudgeResult, Rating
from .tuning_profile import TuningProfile, compose_auto_cefr_system_prompt, load_profile
from .tuning_store import make_run_id, read_json, write_json


def run_tuning_dataset(
    dataset: dict[str, Any],
    profile: TuningProfile,
    *,
    base_dir: Path,
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    items = list(dataset.get("items", []))
    extractor: FluencyExtractor | None = None
    records: list[dict[str, Any]] = []

    for item in items:
        sample_id = str(item.get("sample_id") or Path(str(item["audio_path"])).stem)
        try:
            objective_data = item.get("objective_data")
            if not objective_data:
                extractor = extractor or FluencyExtractor()
                audio_path = _resolve_path(base_dir, Path(str(item["audio_path"])))
                objective_data = extractor.extract(audio_path)

            roleplay_input = _build_roleplay_input(item, objective_data)
            judge_results, judge_failures = _run_judges(
                roleplay_input,
                profile,
                timeout_sec=timeout_sec,
            )
            if not judge_results:
                records.append(
                    _error_record(
                        item,
                        sample_id,
                        "all_judges_failed",
                        [failure_to_dict(failure) for failure in judge_failures],
                        objective_data,
                    )
                )
                continue

            deliberation = deliberate_auto_cefr(
                judge_results,
                objective_data=objective_data,
                profile=profile,
            )
            decision = deliberation.final_decision
            predicted_cefr = decision.final_cefr_level
            task_rating = aggregate_task_rating(judge_results)
            records.append(
                {
                    "sample_id": sample_id,
                    "status": "ok",
                    "audio_path": str(item.get("audio_path", "")),
                    "human_cefr": _optional_level(item.get("human_cefr")),
                    "human_rating": _optional_rating_value(item.get("human_rating")),
                    "raw_predicted_cefr": deliberation.raw_decision.final_cefr_level,
                    "predicted_cefr": predicted_cefr,
                    "predicted_task_rating": task_rating.value,
                    "cefr_correct": (
                        predicted_cefr == _optional_level(item.get("human_cefr"))
                        if item.get("human_cefr")
                        else None
                    ),
                    "adjacent_correct": (
                        is_adjacent_level(
                            _optional_level(item.get("human_cefr")),
                            predicted_cefr,
                        )
                        if item.get("human_cefr")
                        else None
                    ),
                    "calibration_applied": deliberation.applied_calibration,
                    "deliberation_conclusion": deliberation.conclusion,
                    "needs_human_review": decision.needs_human_review
                    or bool(judge_failures),
                    "judge_results": [result.to_dict() for result in judge_results],
                    "judge_failures": [failure_to_dict(failure) for failure in judge_failures],
                    "objective_data": objective_data,
                    "notes": str(item.get("notes", "")),
                }
            )
        except Exception as exc:
            records.append(_error_record(item, sample_id, str(exc), [], item.get("objective_data")))

    return {
        "run_id": make_run_id("tuning"),
        "dataset_name": str(dataset.get("name", "tuning_dataset")),
        "profile": profile.to_dict(),
        "metrics": compute_cefr_metrics(records),
        "records": records,
    }


def compute_cefr_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("human_cefr") in CEFR_LEVELS
        and record.get("predicted_cefr") in CEFR_LEVELS
    ]
    matrix = [[0 for _ in CEFR_LEVELS] for _ in CEFR_LEVELS]
    indexes = {level: index for index, level in enumerate(CEFR_LEVELS)}
    for record in evaluable:
        matrix[indexes[record["human_cefr"]]][indexes[record["predicted_cefr"]]] += 1

    per_level = {}
    for index, level in enumerate(CEFR_LEVELS):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row_i, row in enumerate(matrix) if row_i != index)
        false_negative = sum(value for col_i, value in enumerate(matrix[index]) if col_i != index)
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_level[level] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[index]),
        }

    total = len(evaluable)
    exact = sum(record["human_cefr"] == record["predicted_cefr"] for record in evaluable)
    adjacent = sum(
        is_adjacent_level(record["human_cefr"], record["predicted_cefr"])
        for record in evaluable
    )
    failures = [record for record in records if record.get("status") != "ok"]
    provider_failures = sum(len(record.get("judge_failures", [])) for record in records)
    return {
        "n_total": len(records),
        "n_evaluable": total,
        "n_failed": len(failures),
        "accuracy": exact / total if total else 0.0,
        "adjacent_accuracy": adjacent / total if total else 0.0,
        "macro_f1": (
            sum(level_metrics["f1"] for level_metrics in per_level.values()) / len(per_level)
            if per_level
            else 0.0
        ),
        "labels": CEFR_LEVELS,
        "confusion_matrix": matrix,
        "per_level": per_level,
        "provider_failure_count": provider_failures,
        "human_review_rate": (
            sum(bool(record.get("needs_human_review")) for record in records) / len(records)
            if records
            else 0.0
        ),
        "mismatched_samples": [
            record["sample_id"]
            for record in evaluable
            if record["human_cefr"] != record["predicted_cefr"]
        ],
        "level_counts": dict(Counter(record.get("human_cefr") for record in evaluable)),
    }


def is_adjacent_level(human_level: str | None, predicted_level: str | None) -> bool:
    if human_level not in CEFR_LEVELS or predicted_level not in CEFR_LEVELS:
        return False
    return abs(CEFR_LEVELS.index(human_level) - CEFR_LEVELS.index(predicted_level)) <= 1


def failure_to_dict(failure: JudgeFailure) -> dict[str, str]:
    return {
        "judge_id": failure.judge_id,
        "provider": failure.provider_spec.provider,
        "model": failure.provider_spec.model,
        "message": failure.message,
    }


def _run_judges(
    roleplay_input: dict[str, Any],
    profile: TuningProfile,
    *,
    timeout_sec: float,
) -> tuple[list[AutoLevelJudgeResult], list[JudgeFailure]]:
    if profile.judge_mode == "mock":
        return judge_auto_cefr_with_mock_panel(roleplay_input), []
    if profile.judge_mode != "live":
        raise ValueError("profile.judge_mode must be 'mock' or 'live'.")
    provider_specs = list(profile.judge_providers)
    if not provider_specs:
        raise ValueError("live tuning profile requires at least one judge provider.")
    if profile.allow_partial_judges:
        return judge_auto_cefr_with_live_panel_partial(
            roleplay_input,
            provider_specs,
            timeout_sec=timeout_sec,
            system_prompt=compose_auto_cefr_system_prompt(profile),
        )
    results = judge_auto_cefr_with_live_panel(
        roleplay_input,
        provider_specs,
        timeout_sec=timeout_sec,
        system_prompt=compose_auto_cefr_system_prompt(profile),
    )
    return results, []


def _build_roleplay_input(item: dict[str, Any], objective_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(item.get("sample_id") or Path(str(item["audio_path"])).stem),
        "target_cefr_level": "auto",
        "roleplay_id": str(item.get("roleplay_id", "rp-1")),
        "roleplay_task": str(item.get("roleplay_task", "不明")),
        "jfs_can_do_criteria": item.get("jfs_can_do_criteria", []),
        "raw_transcript_hiragana": objective_data["raw_transcript_hiragana"],
        "fluency_metrics": objective_data["fluency_metrics"],
        "speaker_metadata": item.get("speaker_metadata", {}),
        "optional_expected_information": item.get("optional_expected_information", []),
    }


def _error_record(
    item: dict[str, Any],
    sample_id: str,
    message: str,
    judge_failures: list[dict[str, Any]],
    objective_data: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "status": "error",
        "audio_path": str(item.get("audio_path", "")),
        "human_cefr": _optional_level(item.get("human_cefr")),
        "human_rating": _optional_rating_value(item.get("human_rating")),
        "error": message,
        "judge_failures": judge_failures,
        "objective_data": objective_data,
        "notes": str(item.get("notes", "")),
    }


def _resolve_path(base_dir: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else base_dir / path
    if not resolved.exists():
        raise FileNotFoundError(f"audio file not found: {resolved}")
    return resolved


def _optional_level(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    level = str(value).strip().upper()
    if level not in CEFR_LEVELS:
        raise ValueError(f"human_cefr must be one of {', '.join(CEFR_LEVELS)}: {level}")
    return level


def _optional_rating_value(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return Rating.parse(str(value).strip()).value


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run J-GRADE tuning benchmark")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    load_env_file(args.env_file)
    dataset = read_json(args.dataset)
    profile = load_profile(args.profile)
    result = run_tuning_dataset(
        dataset,
        profile,
        base_dir=args.base_dir,
        timeout_sec=args.timeout_sec,
    )
    write_json(args.out, result)
    print(f"wrote {args.out}")
    print(
        "metrics: "
        f"accuracy={result['metrics']['accuracy']:.3f}, "
        f"adjacent_accuracy={result['metrics']['adjacent_accuracy']:.3f}, "
        f"macro_f1={result['metrics']['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
