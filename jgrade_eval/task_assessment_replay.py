"""Offline replay of saved observations; never calls a model or changes CEFR."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Rating
from .task_assessment import assess_shadow_tasks
from .task_context import TaskContext
from .task_observation import TaskObservation
from .task_rubrics import TaskRubric


def _metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    labeled = len(pairs)
    correct = sum(match for _, match in pairs)
    return {"labeled": labeled, "predicted": sum(present for present, _ in pairs),
            "correct": correct, "accuracy": correct / labeled if labeled else None}


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches(actual[key], value) for key, value in expected.items()
        )
    return type(actual) is type(expected) and actual == expected


def _validate_labels(labels: Any, prompt_ids: set[str]) -> None:
    if not isinstance(labels, dict) or set(labels) - prompt_ids:
        raise ValueError("labels must reference known prompt IDs.")
    for label in labels.values():
        if not isinstance(label, dict) or set(label) - {"observation", "rating"}:
            raise ValueError("labels accepts only observation and rating.")
        if "rating" in label:
            Rating.parse(label["rating"])
        if "observation" not in label:
            continue
        expected = label["observation"]
        if not isinstance(expected, dict) or not expected or set(expected) - {"act_type", "target", "features"}:
            raise ValueError("observation label must be a nonempty subset of act_type, target, features.")
        if "act_type" in expected and (not isinstance(expected["act_type"], str) or not expected["act_type"]):
            raise ValueError("act_type label must be nonempty text.")
        if "target" in expected and expected["target"] is not None and not isinstance(expected["target"], str):
            raise ValueError("target label must be text or null.")
        if "features" in expected:
            features = expected["features"]
            if not isinstance(features, dict) or not features or any(
                not isinstance(key, str) or type(value) not in {str, bool, int}
                for key, value in features.items()
            ):
                raise ValueError("feature labels must be nonempty named primitive values.")


def replay_cases(cases: list[dict[str, Any]], rubrics: list[TaskRubric]) -> dict[str, Any]:
    """Measure independently labeled stages, including abstentions in denominators."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list.")
    identities = [(rubric.rubric_id, rubric.version) for rubric in rubrics]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate rubric ID/version.")
    digest = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False).encode("utf-8")).hexdigest()
    runs = [{"rubric": {"rubric_id": r.rubric_id, "version": r.version,
                        "approval_status": r.approval_status, "task_type": r.task_type,
                        "criteria_source": r.criteria_source, "approved_by": r.approved_by},
             "results": [], "pairs": []} for r in rubrics]
    if not runs:
        runs = [{"rubric": None, "results": [], "pairs": []}]
    observations = []
    observation_pairs = []
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
            raise ValueError("each case requires a nonempty case_id.")
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError("duplicate case_id.")
        seen.add(case_id)
        transcript = case.get("transcript")
        duration = case.get("duration_sec")
        if not isinstance(transcript, str) or type(duration) not in {int, float} or duration <= 0:
            raise ValueError("case requires transcript and positive duration_sec.")
        context = TaskContext.from_inputs(roleplay_task=None, interaction_context=None,
                                          task_context=case.get("task_context"),
                                          transcript=transcript, duration_sec=duration)
        if not context.prompts:
            raise ValueError("case requires at least one prompt.")
        prompt_ids = {prompt.prompt_id for prompt in context.prompts}
        saved = case.get("observations", {})
        if not isinstance(saved, dict) or set(saved) - prompt_ids:
            raise ValueError("observations must reference known prompt IDs.")
        labels = case.get("labels", {})
        _validate_labels(labels, prompt_ids)
        valid = {}
        errors = {}
        for prompt in context.prompts:
            try:
                raw = saved.get(prompt.prompt_id)
                if not isinstance(raw, dict):
                    raise ValueError("missing saved observation")
                valid[prompt.prompt_id] = TaskObservation.from_dict(
                    raw, answer=context.confirmed_answer_for(prompt.prompt_id))
            except ValueError as error:
                errors[prompt.prompt_id] = str(error)
            observation = valid.get(prompt.prompt_id)
            observations.append({"case_id": case_id, "prompt_id": prompt.prompt_id,
                                 "observation": observation.to_dict() if observation else None,
                                 "error": errors.get(prompt.prompt_id)})
            expected = labels.get(prompt.prompt_id, {}).get("observation")
            if expected is not None:
                observation_pairs.append((observation is not None,
                                          observation is not None and _matches(observation.to_dict(), expected)))

        def observe(prompt, answer):
            if prompt.prompt_id not in valid:
                raise ValueError(errors[prompt.prompt_id])
            return valid[prompt.prompt_id]

        for index, run in enumerate(runs):
            registry = {rubrics[index].rubric_id: rubrics[index]} if rubrics else {}
            for result in assess_shadow_tasks(context, registry, observe=observe):
                # A rubric version is compared only on prompts that select that rubric.
                prompt = next(p for p in context.prompts if p.prompt_id == result.prompt_id)
                if rubrics and prompt.rubric_id != rubrics[index].rubric_id:
                    continue
                run["results"].append({"case_id": case_id, **result.to_dict()})
                label = labels.get(result.prompt_id, {})
                if "rating" in label:
                    run["pairs"].append((result.rating is not None,
                                         result.rating == Rating.parse(label["rating"])))
    for run in runs:
        run["metrics"] = _metrics(run.pop("pairs"))
    return {"schema_version": "task-replay.v1", "input_sha256": digest,
            "observations": observations, "observation_metrics": _metrics(observation_pairs),
            "rubric_runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--rubric", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = [args.input, *args.rubric]
        if any(args.output.resolve() == path.resolve() or
               (args.output.exists() and args.output.samefile(path)) for path in inputs):
            raise ValueError("output must not overwrite input or rubric files.")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("input must be an object containing cases.")
        rubrics = [TaskRubric.from_dict(json.loads(path.read_text(encoding="utf-8"))) for path in args.rubric]
        report = replay_cases(data.get("cases"), rubrics)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
