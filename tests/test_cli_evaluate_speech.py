from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jgrade_eval.cli import main


class EvaluateSpeechCliTests(unittest.TestCase):
    def test_shadow_uses_same_service_as_api_and_saves_both_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(json.dumps({
                "prompts": [{"prompt_id": "q1", "text": "予定について話してください"}],
                "whole_recording_answer_prompt_id": "q1",
            }, ensure_ascii=False), encoding="utf-8")
            output = root / "comparison.json"
            expected = {
                "task_rating": "○",
                "task_assessment_shadow": {
                    "status": "rubric_unavailable", "rating": None,
                    "observation": {"act_type": "question", "quote": "そうですか"},
                },
            }
            argv = ["jgrade", "evaluate-speech", "--audio", "sample.wav",
                    "--judge-mode", "live", "--judge-providers", "openai:fake",
                    "--assessment-mode", "shadow", "--task-context-file", str(context),
                    "--modules", "fluency,interaction", "--output", str(output)]
            with (patch.object(sys, "argv", argv),
                  patch("jgrade_eval.api_service.evaluate_speech_level", return_value=expected) as evaluate):
                main()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
            self.assertEqual(evaluate.call_args.args, (Path("sample.wav"),))
            self.assertEqual(evaluate.call_args.kwargs["assessment_mode"], "shadow")
            self.assertEqual(evaluate.call_args.kwargs["selected_modules"], ("fluency", "interaction"))
            self.assertEqual(evaluate.call_args.kwargs["task_context"]["prompts"][0]["prompt_id"], "q1")
            self.assertEqual(evaluate.call_args.kwargs["roleplay_task"], "予定について話してください")

    def test_default_mode_remains_off(self) -> None:
        argv = ["jgrade", "evaluate-speech", "--audio", "sample.wav"]
        with (patch.object(sys, "argv", argv),
              patch("jgrade_eval.api_service.evaluate_speech_level", return_value={}) as evaluate):
            main()
        self.assertEqual(evaluate.call_args.kwargs["assessment_mode"], "off")


if __name__ == "__main__":
    unittest.main()
