from __future__ import annotations

import unittest

from jgrade_eval.models import Rating
from jgrade_eval.task_assessment import assess_shadow_task
from jgrade_eval.task_context import TaskContext
from jgrade_eval.task_observation import TaskObservation
from jgrade_eval.task_rubrics import TaskRubric, apply_rubric


def _confirmed_context() -> TaskContext:
    return TaskContext.from_inputs(
        roleplay_task="unknown", interaction_context=None,
        task_context={
            "prompts": [{"prompt_id": "q1", "text": "日付を確認してください",
                         "task_type": "date_confirmation", "rubric_id": "date-test"}],
            "whole_recording_answer_prompt_id": "q1",
        },
        transcript="らいしゅうのきんようびでいいですか",
        duration_sec=4.0,
    )


def _rubric() -> TaskRubric:
    return TaskRubric.from_dict({
        "rubric_id": "date-test", "version": "test-v1", "task_type": "date_confirmation",
        "approval_status": "experimental",
        "rules": [{"rule_id": "request-date", "all_features": {
            "act_type": "confirmation_request", "target": "date", "date_anchor_present": True,
        }, "rating": "○"}],
    })


def _observation() -> TaskObservation:
    return TaskObservation.from_dict({
        "prompt_id": "q1", "segment_id": "whole-recording",
        "act_type": "confirmation_request", "target": "date",
        "quote": "きんようびでいいですか", "start_offset": 6, "end_offset": 17,
        "features": {"date_anchor_present": True},
        "judge_id": "O1", "model": "fake",
    }, answer=_confirmed_context().confirmed_answer_for("q1"))


class TaskAssessmentTests(unittest.TestCase):
    def test_observation_rejects_quote_not_in_answer(self) -> None:
        answer = _confirmed_context().confirmed_answer_for("q1")
        with self.assertRaisesRegex(ValueError, "quote"):
            TaskObservation.from_dict({
                "prompt_id": "q1", "segment_id": "whole-recording",
                "act_type": "confirmation_request", "target": "date",
                "quote": "あしたですか", "start_offset": 0, "end_offset": 6,
                "features": {}, "judge_id": "O1", "model": "fake",
            }, answer=answer)

    def test_observation_rejects_model_rating(self) -> None:
        answer = _confirmed_context().confirmed_answer_for("q1")
        with self.assertRaisesRegex(ValueError, "must not include a rating"):
            TaskObservation.from_dict({
                "prompt_id": "q1", "segment_id": "whole-recording",
                "act_type": "confirmation_request", "target": "date",
                "quote": "らいしゅう", "start_offset": 0, "end_offset": 5,
                "features": {}, "judge_id": "O1", "model": "fake", "rating": "○",
            }, answer=answer)

    def test_observation_rejects_reserved_feature_keys(self) -> None:
        answer = _confirmed_context().confirmed_answer_for("q1")
        with self.assertRaisesRegex(ValueError, "reserved"):
            TaskObservation.from_dict({
                "prompt_id": "q1", "segment_id": "whole-recording",
                "act_type": "statement", "target": "date",
                "quote": "らいしゅう", "start_offset": 0, "end_offset": 5,
                "features": {"act_type": "confirmation_request"},
                "judge_id": "O1", "model": "fake",
            }, answer=answer)

    def test_fixed_rubric_applies_without_llm(self) -> None:
        result = apply_rubric(_rubric(), _observation(), _confirmed_context().prompts[0])
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.rating, Rating.PASS)
        self.assertEqual(result.rule_id, "request-date")
        self.assertEqual(result.rubric_version, "test-v1")

    def test_no_matching_or_conflicting_rule_abstains(self) -> None:
        rubric = TaskRubric.from_dict({
            "rubric_id": "date-test", "version": "test-v1", "task_type": "date_confirmation",
            "approval_status": "experimental",
            "rules": [{"rule_id": "other", "all_features": {"act_type": "statement"}, "rating": "×"}],
        })
        self.assertEqual(apply_rubric(rubric, _observation(), _confirmed_context().prompts[0]).status,
                         "no_rule_match")
        conflicting = TaskRubric.from_dict({
            "rubric_id": "date-test", "version": "test-v1", "task_type": "date_confirmation",
            "approval_status": "experimental",
            "rules": [
                {"rule_id": "r1", "all_features": {"act_type": "confirmation_request"}, "rating": "○"},
                {"rule_id": "r2", "all_features": {"target": "date"}, "rating": "◎"},
            ],
        })
        self.assertEqual(apply_rubric(conflicting, _observation(), _confirmed_context().prompts[0]).status,
                         "conflicted")

    def test_shadow_skips_unconfirmed_and_missing_rubric_without_observing(self) -> None:
        calls: list[str] = []
        def observe(_prompt, _answer):
            calls.append("called")
            return _observation()

        unconfirmed = TaskContext.from_inputs(
            roleplay_task="日付を確認", interaction_context=None, task_context=None,
            transcript="きんようびですか", duration_sec=3.0,
        )
        self.assertEqual(assess_shadow_task(unconfirmed, {"date-test": _rubric()}, observe=observe).status,
                         "insufficient_context")
        self.assertEqual(assess_shadow_task(_confirmed_context(), {}, observe=observe).status,
                         "rubric_unavailable")
        self.assertEqual(calls, [])

    def test_shadow_observer_failure_does_not_raise(self) -> None:
        def observe(_prompt, _answer):
            raise TimeoutError("provider timeout")
        result = assess_shadow_task(_confirmed_context(), {"date-test": _rubric()}, observe=observe)
        self.assertEqual(result.status, "observation_failed")
        self.assertIsNone(result.rating)


if __name__ == "__main__":
    unittest.main()
