from __future__ import annotations

import unittest

from jgrade_eval.task_assessment import assess_shadow_tasks
from jgrade_eval.task_context import TaskContext
from jgrade_eval.task_observation import TaskObservation
from jgrade_eval.task_rubrics import TaskRubric


class MultiTaskAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = "きんようびですかりゆうはこうです"
        self.context = TaskContext.from_inputs(
            roleplay_task="複数の設問に対する連続回答",
            task_context=None,
            interaction_context={
                "prompts": [
                    {"prompt_id": "q1", "text": "日付を確認", "task_type": "date_confirmation", "rubric_id": "r-date"},
                    {"prompt_id": "q2", "text": "理由を説明", "task_type": "reason_explanation", "rubric_id": "r-reason"},
                ],
                "answer_segments": [
                    {"prompt_id": "q1", "start_sec": 0, "end_sec": 3,
                     "start_offset": 0, "end_offset": 8,
                     "transcript_text": self.transcript[:8], "confirmed_by": "teacher-1"},
                    {"prompt_id": "q2", "start_sec": 3, "end_sec": 8,
                     "start_offset": 8, "end_offset": len(self.transcript),
                     "transcript_text": self.transcript[8:], "confirmed_by": "teacher-1"},
                ],
            },
            transcript=self.transcript, duration_sec=8,
        )

    def test_confirmed_multi_question_spans_are_separate(self) -> None:
        self.assertEqual(self.context.confirmed_answer_for("q1").transcript_text, self.transcript[:8])
        self.assertEqual(self.context.confirmed_answer_for("q2").transcript_text, self.transcript[8:])

    def test_shadow_assesses_each_confirmed_prompt_without_mixing(self) -> None:
        rubrics = {
            "r-date": TaskRubric.from_dict({
                "rubric_id": "r-date", "version": "test-v1", "task_type": "date_confirmation",
                "approval_status": "experimental",
                "rules": [{"rule_id": "date", "all_features": {"act_type": "confirmation_request"}, "rating": "○"}],
            }),
            "r-reason": TaskRubric.from_dict({
                "rubric_id": "r-reason", "version": "test-v1", "task_type": "reason_explanation",
                "approval_status": "experimental",
                "rules": [{"rule_id": "reason", "all_features": {"act_type": "explanation"}, "rating": "△"}],
            }),
        }
        def observe(prompt, answer):
            quote = answer.transcript_text
            return TaskObservation.from_dict({
                "prompt_id": prompt.prompt_id, "segment_id": answer.segment_id,
                "act_type": "confirmation_request" if prompt.prompt_id == "q1" else "explanation",
                "target": None, "quote": quote,
                "start_offset": answer.start_offset, "end_offset": answer.end_offset,
                "features": {}, "judge_id": "O1", "model": "fake",
            }, answer=answer)
        results = assess_shadow_tasks(self.context, rubrics, observe=observe)
        self.assertEqual([(r.prompt_id, r.status, r.rating.value) for r in results],
                         [("q1", "applied", "○"), ("q2", "applied", "△")])

    def test_claimed_confirmation_without_human_source_stays_candidate(self) -> None:
        context = TaskContext.from_inputs(
            roleplay_task="unknown", interaction_context=None,
            task_context={
                "prompts": [{"prompt_id": "q1", "text": "日付を確認"}],
                "answers": [{"prompt_id": "q1", "start_offset": 0, "end_offset": 8,
                             "transcript_text": self.transcript[:8], "mapping_status": "confirmed"}],
            },
            transcript=self.transcript, duration_sec=8,
        )
        self.assertIsNone(context.confirmed_answer_for("q1"))


if __name__ == "__main__":
    unittest.main()
