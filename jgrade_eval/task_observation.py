"""Structured, model-inferred speech-act observations; never scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .task_context import TaskAnswer


TASK_OBSERVATION_SCHEMA_VERSION = "task-observation.v1"


@dataclass(frozen=True)
class TaskObservation:
    prompt_id: str
    segment_id: str
    act_type: str
    target: str | None
    quote: str
    start_offset: int
    end_offset: int
    features: Mapping[str, str | bool | int]
    judge_id: str
    model: str
    schema_version: str = TASK_OBSERVATION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, answer: TaskAnswer | None) -> "TaskObservation":
        if "rating" in data or "task_rating" in data or "score" in data:
            raise ValueError("TaskObservation must not include a rating.")
        if answer is None or answer.mapping_status != "confirmed" or answer.transcript_text is None:
            raise ValueError("TaskObservation requires a confirmed answer transcript.")
        prompt_id = str(data.get("prompt_id") or "")
        segment_id = str(data.get("segment_id") or "")
        if prompt_id != answer.prompt_id or segment_id != answer.segment_id:
            raise ValueError("TaskObservation prompt or segment does not match the answer.")
        act_type = str(data.get("act_type") or "").strip()
        if not act_type:
            raise ValueError("TaskObservation requires act_type.")
        start = _strict_int(data.get("start_offset"))
        end = _strict_int(data.get("end_offset"))
        quote = str(data.get("quote") or "")
        full_start = answer.start_offset
        full_end = answer.end_offset
        if (full_start is None or full_end is None or not quote or
                not full_start <= start < end <= full_end or
                quote != answer.transcript_text[start - full_start:end - full_start]):
            raise ValueError("TaskObservation quote does not match the confirmed answer transcript.")
        raw_features = data.get("features") or {}
        if not isinstance(raw_features, Mapping) or any(
            not isinstance(key, str) or type(value) not in {str, bool, int}
            for key, value in raw_features.items()
        ):
            raise ValueError("TaskObservation features must be primitive named values.")
        judge_id = str(data.get("judge_id") or "").strip()
        model = str(data.get("model") or "").strip()
        if not judge_id or not model:
            raise ValueError("TaskObservation requires judge_id and model.")
        return cls(
            prompt_id=prompt_id,
            segment_id=segment_id,
            act_type=act_type,
            target=str(data["target"]).strip() if data.get("target") is not None else None,
            quote=quote,
            start_offset=start,
            end_offset=end,
            features=dict(raw_features),
            judge_id=judge_id,
            model=model,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "segment_id": self.segment_id,
            "act_type": self.act_type,
            "target": self.target,
            "quote": self.quote,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "features": dict(self.features),
            "judge_id": self.judge_id,
            "model": self.model,
        }


def _strict_int(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("TaskObservation transcript offsets must be integers.")
    return value
