"""Optional task assessment side path, isolated from legacy CEFR decisions."""

from __future__ import annotations

from typing import Callable, Mapping

from .task_context import TaskAnswer, TaskContext, TaskPrompt
from .task_observation import TaskObservation
from .task_rubrics import TaskAssessment, TaskRubric, apply_rubric


Observer = Callable[[TaskPrompt, TaskAnswer], TaskObservation]


def assess_shadow_task(
    context: TaskContext,
    rubrics: Mapping[str, TaskRubric],
    *,
    observe: Observer,
) -> TaskAssessment:
    """Assess only one confirmed answer; never mutate or replace the legacy result."""

    if len(context.prompts) != 1:
        return TaskAssessment(status="insufficient_context", reason="one confirmed prompt is required")
    return _assess_one(context.prompts[0], context, rubrics, observe)


def assess_shadow_tasks(
    context: TaskContext,
    rubrics: Mapping[str, TaskRubric],
    *,
    observe: Observer,
) -> tuple[TaskAssessment, ...]:
    """Assess each human-confirmed answer separately, abstaining per prompt."""

    return tuple(_assess_one(prompt, context, rubrics, observe) for prompt in context.prompts)


def _assess_one(
    prompt: TaskPrompt,
    context: TaskContext,
    rubrics: Mapping[str, TaskRubric],
    observe: Observer,
) -> TaskAssessment:
    answer = context.confirmed_answer_for(prompt.prompt_id)
    if answer is None:
        return TaskAssessment(status="insufficient_context", prompt_id=prompt.prompt_id,
                              reason="answer transcript is not confirmed")
    if not prompt.rubric_id or not prompt.task_type:
        return TaskAssessment(status="rubric_unavailable", prompt_id=prompt.prompt_id,
                              reason="approved task metadata is missing")
    rubric = rubrics.get(prompt.rubric_id)
    if rubric is None:
        return TaskAssessment(status="rubric_unavailable", prompt_id=prompt.prompt_id,
                              reason="rubric is not registered")
    if rubric.task_type != prompt.task_type:
        return TaskAssessment(status="rubric_mismatch", prompt_id=prompt.prompt_id,
                              reason="task type does not match rubric")
    try:
        observation = observe(prompt, answer)
        if observation.prompt_id != prompt.prompt_id or observation.segment_id != answer.segment_id:
            return TaskAssessment(status="invalid_observation", prompt_id=prompt.prompt_id,
                                  reason="observation references another answer")
        return apply_rubric(rubric, observation, prompt)
    except Exception as exc:
        return TaskAssessment(status="observation_failed", prompt_id=prompt.prompt_id,
                              reason=type(exc).__name__)
