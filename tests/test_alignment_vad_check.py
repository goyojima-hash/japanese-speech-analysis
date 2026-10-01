import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from jgrade_eval.alignment_vad_check import diagnose_alignment, main, run
from tests import test_alignment_review as fixtures
from tests.test_alignment_benchmark import prediction


class VADCheckTests(unittest.TestCase):
    def diagnose(self, spans, alignment=None):
        return diagnose_alignment(
            alignment or prediction()["methods"]["ctc_viterbi"],
            spans,
            duration_sec=1.0,
            transcript="あい",
        )

    def test_inside_overlap_is_not_accuracy(self):
        result = self.diagnose([{"start": 0.0, "end": 1.0}])
        self.assertEqual(result["outside_unit_count"], 0)
        self.assertAlmostEqual(result["observations"][0]["speech_overlap_ratio"], 1)
        self.assertFalse(result["accuracy_verified"])

    def test_partial_and_outside_durations(self):
        r = self.diagnose([{"start": 0.2, "end": 0.35}])
        self.assertEqual(r["outside_unit_count"], 2)
        self.assertAlmostEqual(r["outside_speech_duration_sec"], 0.3)
        self.assertEqual(r["observations"][0]["relation"], "partially_outside")
        self.assertEqual(r["observations"][1]["relation"], "fully_outside")

    def test_no_vad_speech_is_explicit_not_invalid_gold(self):
        result = self.diagnose([])
        self.assertEqual(result["vad_status"], "vad_detected_no_speech")
        self.assertEqual(result["fully_outside_unit_count"], 2)

    def test_union_of_multiple_spans_and_crossing_pause(self):
        alignment = {
            "units": [
                {
                    "text": "あい",
                    "start_offset": 0,
                    "end_offset": 2,
                    "start_sec": 0.1,
                    "end_sec": 0.9,
                }
            ]
        }
        result = self.diagnose(
            [{"start": 0.0, "end": 0.2}, {"start": 0.8, "end": 1.0}], alignment
        )
        self.assertAlmostEqual(result["observations"][0]["speech_overlap_ratio"], 0.25)
        self.assertEqual(
            len(result["observations"][0]["crossed_long_pause_candidates"]), 1
        )

    def test_invalid_vad_or_duration_rejected(self):
        for spans in (
            [{"start": -1, "end": 0.1}],
            [{"start": 0, "end": 2}],
            [{"start": 0, "end": 0.6}, {"start": 0.5, "end": 0.9}],
        ):
            with self.assertRaises(ValueError):
                self.diagnose(spans)
        for duration in (float("nan"), True, 0):
            with self.assertRaises(ValueError):
                diagnose_alignment(
                    {"units": []}, [], duration_sec=duration, transcript=""
                )

    def test_invalid_alignment_abstains_and_missing_alignment_is_not_zero_error(self):
        alignment = deepcopy(prediction()["methods"]["ctc_viterbi"])
        alignment["units"][0]["end_sec"] = 10
        self.assertEqual(
            self.diagnose([], alignment)["reason"], "invalid_alignment_geometry"
        )
        result = self.diagnose([], {"units": [], "reason": "ctc_path_unavailable"})
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["outside_speech_duration_sec"])

    def test_runner_records_vad_parameters_and_preserves_dataset(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = fixtures.AlignmentReviewTests().dataset(root)
            before = {p.name: p.read_bytes() for p in dataset.iterdir()}
            with (
                patch("silero_vad.load_silero_vad", return_value="model") as load,
                patch(
                    "silero_vad.get_speech_timestamps",
                    return_value=[{"start": 0, "end": 16000}],
                ) as vad,
            ):
                report = run(dataset, root / "vad")
                self.assertEqual(load.call_count, 1)
                self.assertFalse(vad.call_args.kwargs["return_seconds"])
                self.assertEqual(
                    main([str(dataset), "--output-dir", str(root / "cli")]), 0
                )
            self.assertEqual(
                report["samples"][0]["methods"]["ctc_viterbi"]["outside_unit_count"], 0
            )
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in dataset.iterdir()}
            )
            with self.assertRaises(FileExistsError):
                run(dataset, root / "vad")

    def test_changed_audio_or_wrong_decode_length_does_not_complete(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, audio = fixtures.AlignmentReviewTests().dataset(root)
            with (
                patch("silero_vad.load_silero_vad", return_value="model"),
                patch("librosa.load", return_value=(np.zeros(100), 16000)),
                self.assertRaises(ValueError),
            ):
                run(dataset, root / "short")
            self.assertFalse((root / "short" / "report.json").exists())
            audio.write_bytes(b"changed")
            with (
                patch("silero_vad.load_silero_vad", return_value="model"),
                self.assertRaises(ValueError),
            ):
                run(dataset, root / "bad")
            self.assertFalse((root / "bad" / "report.json").exists())
