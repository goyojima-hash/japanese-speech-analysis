"""User-supplied task metadata, independent of the five fact modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TASK_CONTEXT_SCHEMA_VERSION = "task-context.v1"
_UNKNOWN_TASKS = {"", "unknown", "不明", "複数の設問に対する連続回答"}


@dataclass(frozen=True)
class TaskPrompt:
    prompt_id: str
    text: str
    task_type: str | None = None
    rubric_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "prompt_id": self.prompt_id,
            "text": self.text,
            "task_type": self.task_type,
            "rubric_id": self.rubric_id,
        }


@dataclass(frozen=True)
class TaskAnswer:
    prompt_id: str
    segment_id: str
    mapping_status: str
    start_sec: float | None = None
    end_sec: float | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    transcript_text: str | None = None
    confirmed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "segment_id": self.segment_id,
            "mapping_status": self.mapping_status,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "transcript_text": self.transcript_text,
            "confirmed_by": self.confirmed_by,
        }


@dataclass(frozen=True)
class TaskContext:
    prompts: tuple[TaskPrompt, ...]
    answers: tuple[TaskAnswer, ...]
    schema_version: str = TASK_CONTEXT_SCHEMA_VERSION

    @classmethod
    def from_inputs(
        cls,
        *,
        roleplay_task: str | None,
        interaction_context: Mapping[str, Any] | None,
        task_context: Mapping[str, Any] | None,
        transcript: str,
        duration_sec: float,
    ) -> "TaskContext":
        raw_task = dict(task_context or {})
        raw_interaction = dict(interaction_context or {})
        raw_prompts = raw_task.get("prompts")
        if raw_prompts is None:
            raw_prompts = raw_interaction.get("prompts")
        if raw_prompts is None:
            fallback = str(roleplay_task or "").strip()
            raw_prompts = ([{"prompt_id": "rp-1", "text": fallback}]
                           if fallback not in _UNKNOWN_TASKS else [])
        if not isinstance(raw_prompts, list):
            raise ValueError("task_context.prompts must be a list.")
        prompts: list[TaskPrompt] = []
        seen_ids: set[str] = set()
        for item in raw_prompts:
            if not isinstance(item, Mapping):
                raise ValueError("task_context.prompts items must be objects.")
            prompt_id = str(item.get("prompt_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not prompt_id or not text:
                raise ValueError("task_context.prompts requires prompt_id and text.")
            if prompt_id in seen_ids:
                raise ValueError(f"duplicate prompt_id: {prompt_id}")
            seen_ids.add(prompt_id)
            prompts.append(TaskPrompt(
                prompt_id=prompt_id,
                text=text,
                task_type=_optional_text(item.get("task_type")),
                rubric_id=_optional_text(item.get("rubric_id")),
            ))

        raw_answers = raw_task.get("answers")
        if raw_answers is None:
            raw_answers = raw_interaction.get("answer_segments") or []
        if not isinstance(raw_answers, list):
            raise ValueError("task_context.answers must be a list.")
        answers = [
            _parse_answer(item, index, seen_ids, transcript, duration_sec)
            for index, item in enumerate(raw_answers, 1)
        ]
        whole_prompt_id = _optional_text(raw_task.get("whole_recording_answer_prompt_id"))
        if whole_prompt_id is not None:
            if len(prompts) != 1 or whole_prompt_id != prompts[0].prompt_id or answers:
                raise ValueError("whole_recording_answer_prompt_id requires one prompt and no other answers.")
            if not transcript or duration_sec <= 0:
                raise ValueError("whole recording answer requires a nonempty transcript and duration.")
            answers.append(TaskAnswer(
                prompt_id=whole_prompt_id,
                segment_id="whole-recording",
                mapping_status="confirmed",
                start_sec=0.0,
                end_sec=duration_sec,
                start_offset=0,
                end_offset=len(transcript),
                transcript_text=transcript,
                confirmed_by="request",
            ))
        return cls(prompts=tuple(prompts), answers=tuple(answers))

    def confirmed_answer_for(self, prompt_id: str) -> TaskAnswer | None:
        matches = [answer for answer in self.answers
                   if answer.prompt_id == prompt_id and answer.mapping_status == "confirmed"]
        return matches[0] if len(matches) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
            "answers": [answer.to_dict() for answer in self.answers],
        }


def _parse_answer(
    item: Any,
    index: int,
    prompt_ids: set[str],
    transcript: str,
    duration_sec: float,
) -> TaskAnswer:
    if not isinstance(item, Mapping):
        raise ValueError("task_context.answers items must be objects.")
    prompt_id = str(item.get("prompt_id") or "").strip()
    if prompt_id not in prompt_ids:
        raise ValueError("task_context.answers references an unknown prompt_id.")
    start_sec = _optional_float(item.get("start_sec"))
    end_sec = _optional_float(item.get("end_sec"))
    if (start_sec is None) != (end_sec is None):
        raise ValueError("task_context.answers requires both time bounds.")
    if start_sec is not None and not (0 <= start_sec < end_sec <= duration_sec + 0.01):
        raise ValueError("task_context.answers has an invalid time range.")
    start_offset = _optional_int(item.get("start_offset"))
    end_offset = _optional_int(item.get("end_offset"))
    if (start_offset is None) != (end_offset is None):
        raise ValueError("task_context.answers requires both transcript offsets.")
    text = _optional_text(item.get("transcript_text"))
    if start_offset is not None:
        if not (0 <= start_offset < end_offset <= len(transcript)) or text != transcript[start_offset:end_offset]:
            raise ValueError("task_context.answers transcript span does not match the shared transcript.")
    elif text is not None:
        raise ValueError("task_context.answers transcript span requires offsets.")
    confirmed_by = _optional_text(item.get("confirmed_by"))
    status = "confirmed" if confirmed_by and start_offset is not None else (
        "text_candidate" if start_offset is not None else "time_only"
    )
    return TaskAnswer(
        prompt_id=prompt_id,
        segment_id=str(item.get("segment_id") or f"segment-{index}"),
        mapping_status=status,
        start_sec=start_sec,
        end_sec=end_sec,
        start_offset=start_offset,
        end_offset=end_offset,
        transcript_text=text,
        confirmed_by=confirmed_by,
    )


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
