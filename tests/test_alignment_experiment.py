from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from jgrade_eval.alignment_experiment import analyze_audio, main


def scores(*ids: int) -> np.ndarray:
    value = np.full((1, len(ids), 3), -8.0, dtype=np.float32)
    for frame, token_id in enumerate(ids):
        value[0, frame, token_id] = 8.0
    return value


class FakeTokenizer:
    pad_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "あ": 1, "い": 2}


class FakeProcessor:
    tokenizer = FakeTokenizer()


class AlignmentExperimentTests(unittest.TestCase):
    def test_audio_runner_uses_existing_asr_and_records_reproducible_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            audio_path.write_bytes(b"test audio")
            fake_audio = np.zeros(1_100, dtype=np.float32)
            with patch("fluency.load_vumichien", return_value=(object(), FakeProcessor(), "cpu")), \
                    patch("fluency.transcribe", return_value=(
                        "あい", [scores(0, 1, 0, 2, 0)], 1.1, fake_audio,
                    )):
                report = analyze_audio(audio_path, sample_rate=1000)
        self.assertEqual(report["alignment"]["status"], "complete")
        self.assertEqual(report["alignment"]["units"][1]["start_sec"], 0.66)
        self.assertEqual(report["source"]["model_id"],
                         "vumichien/wav2vec2-large-xlsr-japanese-hiragana")
        self.assertEqual(len(report["source"]["audio_sha256"]), 64)
        self.assertEqual(len(report["source"]["transcript_sha256"]), 64)
        self.assertEqual(report["transcript_source"], "asr")

    def test_runner_uses_exact_supplied_transcript_without_replacing_asr(self) -> None:
        with TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            audio_path.write_bytes(b"test audio")
            with patch("fluency.load_vumichien", return_value=(object(), FakeProcessor(), "cpu")), \
                    patch("fluency.transcribe", return_value=(
                        "あ", [scores(0, 1, 0, 2, 0)], 1.1,
                        np.zeros(1_100, dtype=np.float32),
                    )):
                report = analyze_audio(audio_path, transcript_hiragana="あい", sample_rate=1000)
        self.assertEqual(report["asr_transcript_hiragana"], "あ")
        self.assertEqual(report["alignment"]["status"], "complete")
        self.assertEqual(report["transcript_source"], "supplied")

    def test_command_prints_json_without_running_normal_judge_path(self) -> None:
        with TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            transcript_path = Path(directory) / "transcript.txt"
            audio_path.write_bytes(b"test audio")
            transcript_path.write_text("あい\n", encoding="utf-8")
            output = StringIO()
            with patch("jgrade_eval.alignment_experiment.analyze_audio", return_value={"ok": True}) as analyze, \
                    redirect_stdout(output):
                code = main([str(audio_path), "--transcript-file", str(transcript_path)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"ok": True})
        self.assertEqual(analyze.call_args.kwargs["transcript_hiragana"], "あい")


if __name__ == "__main__":
    unittest.main()
