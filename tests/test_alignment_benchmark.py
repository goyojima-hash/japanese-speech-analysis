from copy import deepcopy
import unittest

from jgrade_eval.alignment_benchmark import make_reference_template, evaluate_sample, summarize


def prediction():
    units = [dict(text="あ", start_offset=0, end_offset=1, start_sec=0.1, end_sec=0.3),
             dict(text="い", start_offset=1, end_offset=2, start_sec=0.4, end_sec=0.6)]
    return dict(sample_id="sample", source=dict(audio_sha256="a" * 64, duration_sec=1.0),
                transcript_hiragana="あい", methods={"ctc_argmax": dict(units=units),
                                                       "ctc_viterbi": dict(units=deepcopy(units))})


def reviewed():
    ref = make_reference_template(prediction())
    ref["transcript_review"] = dict(status="reviewed", source="human", reviewed_by="tester")
    ref["segments"][0].update(start_sec=0.2, end_sec=0.5, review_status="reviewed",
                              review_source="human", reviewed_by="tester")
    return ref


class BenchmarkTests(unittest.TestCase):
    def test_pending_templates_never_score_suggestions(self):
        ref = make_reference_template(prediction())
        self.assertIsNone(ref["segments"][0]["start_sec"])
        report = evaluate_sample(ref, prediction())
        self.assertEqual(report["status"], "pending_reference")
        self.assertIsNone(report["methods"]["ctc_viterbi"]["boundaries"]["mae_ms"])

    def test_known_errors_and_tolerances(self):
        report = evaluate_sample(reviewed(), prediction())
        metrics = report["methods"]["ctc_viterbi"]
        self.assertAlmostEqual(metrics["boundaries"]["mae_ms"], 100)
        self.assertEqual(metrics["coverage"], 1)
        self.assertEqual(metrics["tolerance_ms"]["100"]["all_reference_boundary_rate"], 1)
        self.assertEqual(metrics["tolerance_ms"]["50"]["all_reference_boundary_rate"], 0)

    def test_missing_prediction_in_denominator(self):
        pred = prediction()
        pred["methods"]["ctc_argmax"]["units"].pop()
        result = evaluate_sample(reviewed(), pred)
        metrics = result["methods"]["ctc_argmax"]
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(metrics["abstentions"][0]["reason"], "missing_units")
        self.assertEqual(metrics["tolerance_ms"]["500"]["all_reference_boundary_rate"], 0)

    def test_changed_transcript_or_audio_abstains(self):
        for field in ("transcript_hiragana", "audio_sha256"):
            ref = reviewed()
            ref[field] = "different"
            result = evaluate_sample(ref, prediction())
            self.assertEqual(result["status"], "incompatible_source")
            self.assertIsNone(result["methods"]["ctc_argmax"]["boundaries"]["mae_ms"])

    def test_invalid_reviewed_gold_rejected(self):
        for value in (-1, float("nan"), True, 2):
            ref = reviewed()
            ref["segments"][0]["end_sec"] = value
            with self.assertRaises(ValueError):
                evaluate_sample(ref, prediction())

    def test_machine_or_missing_reviewer_cannot_be_gold(self):
        for field, value in (("review_source", "machine"), ("reviewed_by", "")):
            ref = reviewed()
            ref["segments"][0][field] = value
            with self.assertRaises(ValueError):
                evaluate_sample(ref, prediction())

    def test_invalid_prediction_abstains(self):
        pred = prediction()
        pred["methods"]["ctc_argmax"]["units"][0]["end_sec"] = 10
        result = evaluate_sample(reviewed(), pred)
        self.assertEqual(result["methods"]["ctc_argmax"]["coverage"], 0)

    def test_aggregate_pending_and_measured_without_averaging_averages(self):
        results = [evaluate_sample(reviewed(), prediction()),
                   evaluate_sample(make_reference_template(prediction()), prediction())]
        results[1]["sample_id"] = "second"
        summary = summarize(results)
        self.assertEqual(summary["pending_samples"], 1)
        self.assertEqual(summary["methods"]["ctc_argmax"]["eligible_segments"], 1)
        self.assertAlmostEqual(summary["methods"]["ctc_argmax"]["boundaries"]["mae_ms"], 100)

    def test_candidate_gaps_do_not_claim_speaker_or_question_detection(self):
        pred = prediction()
        pred["methods"]["ctc_viterbi"]["units"][1].update(start_sec=0.9, end_sec=1)
        ref = make_reference_template(pred)
        self.assertEqual(len(ref["segments"]), 2)
        self.assertEqual(ref["recording_type"], "unknown")

    def test_duplicate_sample_rejected(self):
        report = evaluate_sample(reviewed(), prediction())
        with self.assertRaises(ValueError):
            summarize([report, report])
