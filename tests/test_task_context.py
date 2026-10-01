from __future__ import annotations

import unittest

from jgrade_eval.task_context import TaskContext


class TaskContextTests(unittest.TestCase):
    def test_legacy_prompt_is_untyped_and_not_confirmed(self) -> None:
        context = TaskContext.from_inputs(
            roleplay_task="日付を確認してください",
            interaction_context=None,
            task_context=None,
            transcript="らいしゅうのきんようびですか",
            duration_sec=10.0,
        )
        self.assertEqual(context.prompts[0].prompt_id, "rp-1")
        self.assertIsNone(context.prompts[0].task_type)
        self.assertIsNone(context.confirmed_answer_for("rp-1"))

    def test_confirmed_whole_recording_answer_has_verifiable_text_span(self) -> None:
        transcript = "らいしゅうのきんようびですか"
        context = TaskContext.from_inputs(
            roleplay_task="unknown",
            interaction_context=None,
            task_context={
                "prompts": [{"prompt_id": "q1", "text": "日付を確認してください",
                             "task_type": "date_confirmation", "rubric_id": "date-confirmation.v1"}],
                "whole_recording_answer_prompt_id": "q1",
            },
            transcript=transcript,
            duration_sec=10.0,
        )
        answer = context.confirmed_answer_for("q1")
        assert answer is not None
        self.assertEqual(answer.transcript_text, transcript)
        self.assertEqual((answer.start_offset, answer.end_offset), (0, len(transcript)))
        self.assertEqual(answer.mapping_status, "confirmed")

    def test_time_only_segment_does_not_confirm_text_mapping(self) -> None:
        context = TaskContext.from_inputs(
            roleplay_task="複数設問",
            interaction_context={
                "prompts": [{"prompt_id": "q1", "text": "日付を確認してください"},
                            {"prompt_id": "q2", "text": "理由を説明してください"}],
                "answer_segments": [{"prompt_id": "q1", "start_sec": 0, "end_sec": 5}],
            },
            task_context=None,
            transcript="きんようびですかりゆうは",
            duration_sec=10.0,
        )
        self.assertIsNone(context.confirmed_answer_for("q1"))
        self.assertEqual(context.answers[0].mapping_status, "time_only")

    def test_rejects_duplicate_prompt_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate prompt_id"):
            TaskContext.from_inputs(
                roleplay_task="unknown", interaction_context=None,
                task_context={"prompts": [
                    {"prompt_id": "q1", "text": "質問1"},
                    {"prompt_id": "q1", "text": "質問2"},
                ]},
                transcript="はい", duration_sec=1.0,
            )

    def test_rejects_unverified_segment_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "transcript span"):
            TaskContext.from_inputs(
                roleplay_task="unknown", interaction_context=None,
                task_context={
                    "prompts": [{"prompt_id": "q1", "text": "質問1"}],
                    "answers": [{"prompt_id": "q1", "start_offset": 0,
                                 "end_offset": 2, "transcript_text": "いいえ"}],
                },
                transcript="はい", duration_sec=1.0,
            )

    def test_rejects_non_integer_transcript_offsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            TaskContext.from_inputs(
                roleplay_task="unknown", interaction_context=None,
                task_context={
                    "prompts": [{"prompt_id": "q1", "text": "質問1"}],
                    "answers": [{"prompt_id": "q1", "start_offset": 0.5,
                                 "end_offset": 2, "transcript_text": "はい",
                                 "confirmed_by": "teacher-1"}],
                },
                transcript="はい", duration_sec=1.0,
            )


if __name__ == "__main__":
    unittest.main()
