import csv
import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from jgrade_eval.alignment_benchmark import make_reference_template
from jgrade_eval.alignment_review import export_review, main
from tests.test_alignment_benchmark import prediction


class AlignmentReviewTests(unittest.TestCase):
    def dataset(self, root):
        audio = root / "source.wav"
        sf.write(audio, np.zeros(16000), 16000)
        pred = prediction()
        pred["source"]["audio_sha256"] = sha256(audio.read_bytes()).hexdigest()
        ref = make_reference_template(pred)
        dataset = root / "dataset"
        dataset.mkdir()
        for name, data in [
            ("prediction.json", pred),
            ("reference.json", ref),
            (
                "manifest.json",
                {
                    "schema_version": "alignment-dataset.v1",
                    "status": "complete",
                    "samples": [
                        {
                            "sample_id": "sample",
                            "audio_path": str(audio),
                            "prediction": "prediction.json",
                            "reference": "reference.json",
                        }
                    ],
                },
            ),
        ]:
            (dataset / name).write_text(json.dumps(data), encoding="utf-8")
        return dataset, audio

    def test_export_clips_preserves_gold_and_sample_origin(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _audio = self.dataset(root)
            before = {p.name: p.read_bytes() for p in dataset.iterdir()}
            out = root / "review"
            report = export_review(dataset, out, padding_sec=0.05)
            self.assertEqual(report["clip_count"], 1)
            row = report["items"][0]
            self.assertAlmostEqual(row["clip_origin_sec"], 0.05)
            wave, rate = sf.read(out / row["clip_path"])
            self.assertEqual(rate, 16000)
            self.assertAlmostEqual(len(wave) / rate, 0.6, places=3)
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in dataset.iterdir()}
            )
            with (out / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
                csv_row = next(csv.DictReader(stream))
            self.assertEqual(csv_row["gold_start_sec"], "")
            self.assertEqual(csv_row["reviewed_by"], "")
            with self.assertRaises(FileExistsError):
                export_review(dataset, out)

    def test_changed_audio_fails_without_complete_manifest(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, audio = self.dataset(root)
            audio.write_bytes(b"changed")
            out = root / "review"
            with self.assertRaises(ValueError):
                export_review(dataset, out)
            self.assertFalse((out / "manifest.json").exists())

    def test_missing_suggestions_keeps_row_and_no_fake_clip(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = self.dataset(root)
            refpath = dataset / "reference.json"
            ref = json.loads(refpath.read_text())
            ref["segments"][0].update(suggested_start_sec=None, suggested_end_sec=None)
            refpath.write_text(json.dumps(ref))
            report = export_review(dataset, root / "review")
            self.assertEqual(report["clip_count"], 0)
            self.assertIsNone(report["items"][0]["clip_path"])

    def test_invalid_suggestions_and_padding_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = self.dataset(root)
            for value in (-1, float("nan"), True):
                with self.assertRaises(ValueError):
                    export_review(dataset, root / "review", padding_sec=value)
            path = dataset / "reference.json"
            ref = json.loads(path.read_text())
            ref["segments"][0]["suggested_end_sec"] = 20
            path.write_text(json.dumps(ref))
            with self.assertRaises(ValueError):
                export_review(dataset, root / "review")

    def test_decode_duration_mismatch_and_formula_safe_csv(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = self.dataset(root)
            with patch("librosa.load", return_value=(np.zeros(100), 16000)):
                with self.assertRaises(ValueError):
                    export_review(dataset, root / "bad")
            path = dataset / "reference.json"
            ref = json.loads(path.read_text())
            ref["segments"][0]["segment_id"] = '=HYPERLINK("bad")'
            path.write_text(json.dumps(ref))
            self.assertEqual(
                main([str(dataset), "--output-dir", str(root / "review")]), 0
            )
            with (root / "review" / "review.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertTrue(row["segment_id"].startswith("'="))
