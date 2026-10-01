from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from jgrade_eval.alignment_benchmark_cli import prepare, evaluate, main, capture
from tests.test_alignment_benchmark import prediction


class BenchmarkCLITests(unittest.TestCase):
    def test_prepare_evaluate_pending_and_preserve_existing_directory(self):
        with TemporaryDirectory() as temp:
            audio = Path(temp) / "sample.wav"
            audio.write_bytes(b"audio")
            output = Path(temp) / "benchmark"
            with patch("jgrade_eval.alignment_benchmark_cli.capture", return_value=prediction()):
                prepare([audio], output)
            self.assertEqual(evaluate(output)["pending_samples"], 1)
            with self.assertRaises(FileExistsError):
                prepare([audio], output)
            with redirect_stdout(StringIO()) as stdout:
                self.assertEqual(main(["evaluate", str(output)]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["pending_samples"], 1)

    def test_failed_capture_has_no_complete_manifest(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "benchmark"
            with patch("jgrade_eval.alignment_benchmark_cli.capture", side_effect=RuntimeError("failed")):
                with self.assertRaises(RuntimeError):
                    prepare([Path("sample.wav")], output)
            self.assertFalse((output / "manifest.json").exists())

    def test_manifest_path_escape_and_incomplete_rejected(self):
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "manifest.json"
            for content in (dict(schema_version="alignment-dataset.v1", status="incomplete"),
                            dict(schema_version="alignment-dataset.v1", status="complete",
                                 samples=[dict(prediction="../escape", reference="gold")])):
                manifest.write_text(json.dumps(content))
                with self.assertRaises(ValueError):
                    evaluate(directory)

    def test_capture_same_logits_for_both_methods(self):
        import numpy as np
        import torch
        from tests.test_alignment_experiment import FakeProcessor, scores
        with TemporaryDirectory() as temp:
            audio = Path(temp) / "sample.wav"
            audio.write_bytes(b"audio")
            with patch("fluency.load_vumichien", return_value=(object(), FakeProcessor(), "cpu")), \
                    patch("fluency.transcribe", return_value=(
                        "あい", [torch.tensor(scores(0, 1, 0, 2, 0))], 0.1,
                        np.zeros(1600, dtype=np.float32))) as transcribe:
                result = capture(audio)
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(result["methods"]["ctc_viterbi"]["status"], "complete")
            self.assertEqual(result["methods"]["ctc_argmax"]["status"], "complete")
            self.assertIsNone(result["source"]["model_revision"])

    def test_duplicates_and_empty_inputs_rejected(self):
        with TemporaryDirectory() as temp:
            for paths in ([], [Path("a.wav"), Path("a.wav")]):
                with self.assertRaises(ValueError):
                    prepare(paths, Path(temp) / "new")
