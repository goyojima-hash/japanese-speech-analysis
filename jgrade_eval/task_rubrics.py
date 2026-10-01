"""Versioned deterministic task-specific rules; no model calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .models import Rating
from .task_context import TaskPrompt
from .task_observation import TaskObservation


@dataclass(frozen=True)
class RubricRule:
    rule_id: str
    all_features: Mapping[str, str | bool | int]
    rating: Rating


@dataclass(frozen=True)
class TaskRubric:
    rubric_id: str
    version: str
    task_type: str
    approval_status: str
    rules: tuple[RubricRule, ...]
    criteria_source: str | None = None
    approved_by: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskRubric":
        rubric_id = _required_text(data, "rubric_id")
        version = _required_text(data, "version")
        task_type = _required_text(data, "task_type")
        approval = _required_text(data, "approval_status")
        if approval not in {"experimental", "approved"}:
            raise ValueError("rubric approval_status must be experimental or approved.")
        criteria_source = _optional_text(data.get("criteria_source"))
        approved_by = _optional_text(data.get("approved_by"))
        if approval == "approved" and (not criteria_source or not approved_by):
            raise ValueError("approved rubric requires criteria_source and approved_by.")
        raw_rules = data.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("rubric rules must be a nonempty list.")
        rules: list[RubricRule] = []
        seen_ids: set[str] = set()
        for raw in raw_rules:
            if not isinstance(raw, Mapping):
                raise ValueError("rubric rule must be an object.")
            rule_id = _required_text(raw, "rule_id")
            if rule_id in seen_ids:
                raise ValueError(f"duplicate rubric rule_id: {rule_id}")
            seen_ids.add(rule_id)
            features = raw.get("all_features")
            if not isinstance(features, Mapping) or not features or any(
                not isinstance(key, str) or type(value) not in {str, bool, int}
                for key, value in features.items()
            ):
                raise ValueError("rubric all_features must be nonempty primitive conditions.")
            rules.append(RubricRule(rule_id, dict(features), Rating.parse(raw.get("rating"))))
        return cls(rubric_id, version, task_type, approval, tuple(rules), criteria_source, approved_by)


def load_task_rubrics(directory: Path) -> dict[str, TaskRubric]:
    """Load operator-managed rubric JSON; never accept rules from evaluation input."""

    if not directory.exists():
        return {}
    if not directory.is_dir():
        raise ValueError("task rubric path must be a directory.")
    rubrics: dict[str, TaskRubric] = {}
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"rubric file must contain an object: {path.name}")
        rubric = TaskRubric.from_dict(raw)
        if rubric.rubric_id in rubrics:
            raise ValueError(f"duplicate rubric_id: {rubric.rubric_id}")
        rubrics[rubric.rubric_id] = rubric
    return rubrics


def load_default_task_rubrics() -> dict[str, TaskRubric]:
    configured = os.environ.get("JGRADE_TASK_RUBRICS_DIR")
    directory = Path(configured) if configured else Path(__file__).resolve().parent.parent / "config" / "task_rubrics"
    return load_task_rubrics(directory)


@dataclass(frozen=True)
class TaskAssessment:
    status: str
    prompt_id: str | None = None
    rating: Rating | None = None
    rule_id: str | None = None
    rubric_id: str | None = None
    rubric_version: str | None = None
    observation: TaskObservation | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "prompt_id": self.prompt_id,
            "rating": self.rating.value if self.rating is not None else None,
            "rule_id": self.rule_id,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "observation": self.observation.to_dict() if self.observation else None,
            "reason": self.reason,
        }


def apply_rubric(rubric: TaskRubric, observation: TaskObservation, prompt: TaskPrompt) -> TaskAssessment:
    if prompt.rubric_id != rubric.rubric_id or prompt.task_type != rubric.task_type:
        return TaskAssessment(status="rubric_mismatch", prompt_id=prompt.prompt_id,
                              reason="prompt metadata does not match rubric")
    if observation.prompt_id != prompt.prompt_id:
        return TaskAssessment(status="invalid_observation", prompt_id=prompt.prompt_id,
                              reason="prompt_id mismatch")
    values: dict[str, str | bool | int | None] = {
        **observation.features,
        "act_type": observation.act_type,
        "target": observation.target,
    }
    matches = [rule for rule in rubric.rules if all(
        key in values and type(values[key]) is type(expected) and values[key] == expected
        for key, expected in rule.all_features.items()
    )]
    if not matches:
        return TaskAssessment(status="no_rule_match", prompt_id=prompt.prompt_id,
                              rubric_id=rubric.rubric_id,
                              rubric_version=rubric.version, observation=observation)
    if len(matches) > 1:
        return TaskAssessment(status="conflicted", prompt_id=prompt.prompt_id,
                              rubric_id=rubric.rubric_id,
                              rubric_version=rubric.version, observation=observation,
                              reason="more than one rubric rule matched")
    rule = matches[0]
    return TaskAssessment(status="applied", prompt_id=prompt.prompt_id,
                          rating=rule.rating, rule_id=rule.rule_id,
                          rubric_id=rubric.rubric_id, rubric_version=rubric.version,
                          observation=observation)


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"rubric requires {key}.")
    return value


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
