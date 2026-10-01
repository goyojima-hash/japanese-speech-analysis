from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import CEFR_LEVELS
from .tuning_store import read_json, write_json


DEFAULT_REVIEW_DATASET_PATH = Path("data/tuning/review_samples.json")
MAX_PREDICTION_HISTORY = 20


def make_review_sample_id(audio_path: Path, objective_data: dict[str, Any]) -> str:
    """Create a stable sample id for repeated reviews of the same audio."""
    transcript = str(objective_data.get("raw_transcript_hiragana", ""))
    seed = f"{audio_path.expanduser()}|{transcript[:500]}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    stem = _slug(audio_path.stem) or "sample"
    return f"{stem}-{digest}"


def append_judged_sample(
    dataset_path: Path,
    *,
    record: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    dataset = load_review_dataset(dataset_path)
    item = review_record_to_item(record, source=source)
    dataset["items"] = _upsert_item(list(dataset.get("items", [])), item)
    dataset["updated_at"] = _now_iso()
    write_json(dataset_path, dataset)
    return dataset


def merge_run_result(
    dataset_path: Path,
    result: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    dataset = load_review_dataset(dataset_path)
    items = list(dataset.get("items", []))
    for record in result.get("records", []):
        items = _upsert_item(items, review_record_to_item(record, source=source))
    dataset["items"] = items
    dataset["updated_at"] = _now_iso()
    write_json(dataset_path, dataset)
    return dataset


def load_review_dataset(dataset_path: Path) -> dict[str, Any]:
    if dataset_path.exists():
        loaded = read_json(dataset_path)
        loaded.setdefault("name", "jgrade_review_samples")
        loaded.setdefault("items", [])
        return loaded
    return {
        "name": "jgrade_review_samples",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "items": [],
    }


def review_record_to_item(record: dict[str, Any], *, source: str) -> dict[str, Any]:
    objective_data = record.get("objective_data") or {}
    audio_path = str(record.get("audio_path", ""))
    sample_id = str(record.get("sample_id") or Path(audio_path).stem)
    prediction = _prediction_snapshot(record, source=source)
    item = {
        "sample_id": sample_id,
        "audio_path": audio_path,
        "human_cefr": _optional_cefr(record.get("human_cefr")),
        "human_rating": str(record.get("human_rating") or ""),
        "roleplay_id": str(record.get("roleplay_id") or "rp-1"),
        "roleplay_task": str(record.get("roleplay_task") or "不明"),
        "notes": str(record.get("notes") or f"auto-added from {source}"),
        "objective_data": objective_data,
        "last_prediction": prediction,
        "prediction_history": [prediction],
    }
    if record.get("source_sample_id"):
        item["source_sample_id"] = str(record["source_sample_id"])
    return item


def _upsert_item(items: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    for index, existing in enumerate(items):
        if existing.get("sample_id") != item.get("sample_id"):
            continue
        history = [
            item["last_prediction"],
            *list(existing.get("prediction_history", [])),
        ][:MAX_PREDICTION_HISTORY]
        merged = {
            **existing,
            **item,
            "human_cefr": item.get("human_cefr") or existing.get("human_cefr"),
            "human_rating": item.get("human_rating") or existing.get("human_rating", ""),
            "prediction_history": history,
        }
        items[index] = merged
        return items
    return [*items, item]


def _prediction_snapshot(record: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        **({"task_validation": record["task_validation"]} if "task_validation" in record else {}),
        "recorded_at": _now_iso(),
        "source": source,
        "status": str(record.get("status") or "ok"),
        "raw_predicted_cefr": _optional_cefr(
            record.get("raw_predicted_cefr") or record.get("predicted_cefr")
        ),
        "predicted_cefr": _optional_cefr(record.get("predicted_cefr")),
        "predicted_task_rating": str(record.get("predicted_task_rating") or ""),
        "calibration_applied": bool(record.get("calibration_applied")),
        "deliberation": dict(record.get("deliberation", {})),
        "needs_human_review": bool(record.get("needs_human_review")),
        "judge_results": list(record.get("judge_results", [])),
        "judge_failures": list(record.get("judge_failures", [])),
        "error": str(record.get("error") or ""),
    }


def _optional_cefr(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    level = str(value).strip().upper()
    return level if level in CEFR_LEVELS else None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
