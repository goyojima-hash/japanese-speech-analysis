from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jgrade_eval.alignment_cross_model import compare_candidates, align_audio, run, main
from jgrade_eval.ctc_forced_alignment import align_ctc_chunks, CTCLogitChunk
from tests.test_alignment_benchmark import prediction
from tests.test_alignment_experiment import FakeProcessor, scores
from tests.test_alignment_review import AlignmentReviewTests


def alternate():
    p = prediction()
    units = deepcopy(p['methods']['ctc_viterbi']['units'])
    for unit in units:
        unit['start_sec'] += 0.1
        unit['end_sec'] += 0.1
    return dict(source=deepcopy(p['source']), transcript_hiragana=p['transcript_hiragana'],
                alignment=dict(units=units))


class CrossModelTests(unittest.TestCase):
    def test_deltas_are_not_gold_errors_or_scores(self):
        p = prediction()
        segments = [dict(segment_id='a',start_offset=0,end_offset=2)]
        r = compare_candidates(p, alternate(), segments)
        self.assertAlmostEqual(r['items'][0]['start_delta_ms'], 100)
        self.assertAlmostEqual(r['items'][0]['end_delta_ms'], 100)
        self.assertFalse(r['accuracy_verified'])
        self.assertNotIn('mae_ms', r)

    def test_missing_units_are_explicit_abstentions(self):
        p = prediction()
        alt = alternate()
        alt['alignment']['units'].pop()
        r = compare_candidates(p, alt, [dict(segment_id='a',start_offset=0,end_offset=2)])
        self.assertEqual(r['compared_segments'],0)
        self.assertEqual(r['items'][0]['status'],'unavailable')

    def test_source_and_text_mismatch_rejected(self):
        for key in ('audio_sha256','duration_sec'):
            alt = alternate()
            alt['source'][key] = 'different'
            with self.assertRaises(ValueError):
                compare_candidates(prediction(),alt,[])
        alt = alternate()
        alt['transcript_hiragana'] = 'different'
        with self.assertRaises(ValueError):
            compare_candidates(prediction(),alt,[])

    def test_audio_runner_uses_pinned_processor_without_rewriting_text(self):
        from types import SimpleNamespace
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor(scores(0,1,0,2,0)))
        class Processor(FakeProcessor):
            def __call__(self,*args,**kwargs):
                return {'input_values':torch.zeros(1,1600)}
        with TemporaryDirectory() as temp:
            audio = Path(temp)/'audio.wav'
            audio.write_bytes(b'audio')
            with patch('librosa.load',return_value=(np.zeros(1600),16000)):
                r=align_audio(audio,'あい',(Model(),Processor(),'cpu'))
        self.assertEqual(r['alignment']['status'],'complete')
        self.assertEqual(r['transcript_hiragana'],'あい')
        self.assertEqual(r['source']['sample_rate'],16000)

    def test_tiny_audio_explicitly_unavailable(self):
        with TemporaryDirectory() as temp:
            audio=Path(temp)/'audio.wav'
            audio.write_bytes(b'audio')
            with patch('librosa.load',return_value=(np.zeros(10),16000)):
                result=align_audio(audio,'あ',(object(),FakeProcessor(),'cpu'))
            self.assertEqual(result['alignment']['reason'],'audio_chunk_too_short')

    def test_run_and_cli_preserve_original_dataset_and_do_not_approve_gold(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            dataset,_=AlignmentReviewTests().dataset(root)
            p=json.loads((dataset/'prediction.json').read_text())
            alt=alternate()
            alt['source']=p['source']
            before={p.name:p.read_bytes() for p in dataset.iterdir()}
            with patch('jgrade_eval.alignment_cross_model.load_comparison_model',return_value=('model','proc','cpu')) as load, patch('jgrade_eval.alignment_cross_model.align_audio',return_value=alt):
                report=run(dataset,root/'comparison')
                self.assertEqual(load.call_count,1)
                with redirect_stdout(StringIO()):
                    self.assertEqual(main([str(dataset),'--output-dir',str(root/'cli')]),0)
            self.assertEqual(report['compared_segments'],1)
            self.assertEqual(before,{p.name:p.read_bytes() for p in dataset.iterdir()})
            with self.assertRaises(FileExistsError):
                run(dataset,root/'comparison')

    def test_inference_failure_does_not_create_complete_report(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            dataset,_=AlignmentReviewTests().dataset(root)
            with patch('jgrade_eval.alignment_cross_model.load_comparison_model',side_effect=RuntimeError('fail')):
                with self.assertRaises(RuntimeError):
                    run(dataset,root/'comparison')
            self.assertFalse((root/'comparison'/'report.json').exists())
