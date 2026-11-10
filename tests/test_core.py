import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch
import numpy as np

from basic5 import COMPONENTS
from basic5.audio import prepare,plan_chunks,read_samples,RATE
from basic5.assembly import align_or_fallback,assign_words,assemble,export_srt,restore_text
from basic5.cluster import cluster_embeddings,speaker_windows,windows_to_turns
from basic5.common import read_json,sha256
from basic5.config import DEFAULTS,load_config
from basic5.pipeline import transcribe,run_worker


def wav(path,frames=32000,channels=1):
    data=np.zeros((frames,channels),dtype='<i2')
    if channels>1:
        data[:,0]=1234;data[:,1]=-2345
    with wave.open(str(path),'wb') as f:
        f.setparams((channels,2,RATE,0,'NONE','not compressed'));f.writeframes(data.tobytes())


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg=dict(DEFAULTS)
    def tearDown(self):self.tmp.cleanup()

    def test_five_components(self):
        self.assertEqual(len(COMPONENTS),5);self.assertEqual(len(set(COMPONENTS)),5)

    def test_chunks_cover_every_sample_once(self):
        path=self.root/'input.wav';wav(path,RATE*67+7)
        chunks=plan_chunks(path,self.cfg)
        self.assertEqual(chunks[0]['start_frame'],0)
        self.assertEqual(chunks[-1]['end_frame'],RATE*67+7)
        self.assertTrue(all(a['end_frame']==b['start_frame'] for a,b in zip(chunks,chunks[1:])))
        self.assertTrue(all(0<c['end_frame']-c['start_frame']<=28*RATE for c in chunks))

    def test_ffmpeg_first_channel(self):
        import shutil
        if not shutil.which('ffmpeg'):self.skipTest('FFmpeg unavailable')
        source=self.root/'stereo.wav';output=self.root/'mono.wav';wav(source,channels=2)
        self.assertEqual(prepare(source,output,10),32000)
        self.assertTrue(np.all(read_samples(output)==np.float32(1234/32768)))

    def test_cut_rounding_always_advances(self):
        path=self.root/'short.wav';wav(path,RATE+1)
        chunks=plan_chunks(path,dict(self.cfg,chunk_seconds=1,cut_search_seconds=1-1e-8))
        self.assertEqual(chunks[-1]['end_frame'],RATE+1)
        self.assertTrue(all(c['end_frame']>c['start_frame'] for c in chunks))

    def test_duration_limit_no_silent_truncation(self):
        source=self.root/'audio.wav';wav(source)
        with self.assertRaises(ValueError):prepare(source,self.root/'out.wav',1)

    def test_text_and_punctuation_preserved(self):
        units=[dict(start=0,end=.5,text='你好'),dict(start=.5,end=1,text='世界')]
        result=restore_text(' 你好，世界！',units)
        self.assertEqual(''.join(x['text'] for x in result),' 你好，世界！')

    def test_unicode_expansion_split_falls_back(self):
        chunk=dict(id=0,start=0,end=1,text='ﬁ')
        words,error=align_or_fallback(chunk,[dict(start=0,end=.2,text='f'),dict(start=.2,end=.4,text='i')],1)
        self.assertIsNotNone(error);self.assertEqual(words[0]['text'],'ﬁ')
        self.assertEqual(words[0]['timestamp_source'],'chunk_bounds')

    def test_invalid_alignment_preserves_full_text(self):
        chunk=dict(id=0,start=0,end=2,text='预算十万。')
        for units in ([],[dict(start=0,end=1,text='预算十二万')],[dict(start=-.1,end=1,text='预算十万')]):
            words,error=align_or_fallback(chunk,units,2)
            self.assertIsNotNone(error);self.assertEqual(''.join(x['text'] for x in words),chunk['text'])

    def test_zero_time_point_keeps_zero(self):
        word=dict(start=1,end=1,text='嗯',alignment_status='aligned')
        result=assign_words([word],[dict(start=0,end=2,speaker='S01')],self.cfg)[0]
        self.assertEqual((result['start'],result['end'],result['speaker']),(1,1,'S01'))

    def test_overlap_and_fallback_are_unknown(self):
        turns=[dict(start=0,end=2,speaker='S01'),dict(start=0,end=2,speaker='S02')]
        word=dict(start=.2,end=1,text='好',alignment_status='aligned')
        self.assertEqual(assign_words([word],turns,self.cfg)[0]['speaker'],'unknown')
        word['alignment_status']='unaligned_text_preserved'
        self.assertEqual(assign_words([word],turns[:1],self.cfg)[0]['speaker'],'unknown')

    def test_duplicate_activity_does_not_double_count(self):
        word=dict(start=0,end=2,text='好',alignment_status='aligned')
        turns=[dict(start=0,end=.6,speaker='S01')]*3
        self.assertEqual(assign_words([word],turns,self.cfg)[0]['speaker'],'unknown')

    def test_single_speaker_and_empty_windows(self):
        self.assertEqual(speaker_windows([[0,.1]],self.cfg),[])
        labels,report=cluster_embeddings(np.array([[1.,0.]]),self.cfg)
        self.assertEqual(labels,['S01']);self.assertEqual(report['estimated_speakers'],1)

    def test_two_distinct_speakers(self):
        cfg=dict(self.cfg,num_speakers=2)
        x=np.array([[1.,.01],[1.,-.01],[.01,1.],[-.01,1.]])
        labels,_=cluster_embeddings(x,cfg)
        self.assertEqual(labels[0],labels[1]);self.assertEqual(labels[2],labels[3]);self.assertNotEqual(labels[0],labels[2])
        with self.assertRaises(ValueError):cluster_embeddings(np.array([[0.,0.]]),cfg)

    def test_speaker_turns_do_not_bridge_silence(self):
        vad=[[0,2],[5,7]];windows=speaker_windows(vad,self.cfg)
        turns=windows_to_turns(windows,['S01']*len(windows),vad)
        self.assertEqual([(t['start'],t['end']) for t in turns],[(0,2),(5,7)])

    def test_disconnected_speaker_graph(self):
        labels,report=cluster_embeddings(np.eye(4),self.cfg)
        self.assertEqual(len(set(labels)),4)
        self.assertTrue(report['graph_regularized'])
        labels,_=cluster_embeddings(np.eye(4),dict(self.cfg,num_speakers=3))
        self.assertEqual(len(set(labels)),3)

    def test_opposing_vectors_with_forced_one_speaker(self):
        labels,report=cluster_embeddings(np.array([[1.,0.],[-1.,0.]]),dict(self.cfg,num_speakers=1))
        self.assertEqual(labels,['S01','S01'])
        self.assertEqual(report['cancelled_centers'],1)

    def test_config_rejects_invalid_values(self):
        cfg=dict(python=dict(qwen=sys.executable,speaker=sys.executable),models=dict(asr='a',aligner='b',vad='c',speaker='d'))
        path=self.root/'config.json';path.write_text(json.dumps(dict(cfg,chunk_seconds=0)))
        with self.assertRaises(ValueError):load_config(path,check_assets=False)

    def test_exports_integration_with_fixed_stage_fixtures(self):
        # This exercises serialization/orchestration only; it does NOT emulate model quality.
        source=self.root/'a.wav';wav(source)
        before=sha256(source)
        cfg=dict(self.cfg,python={'qwen':sys.executable,'speaker':sys.executable},models={})
        def stages(stage,request_path,result_path,cfg,log_path):
            req=read_json(request_path)
            if stage=='asr':
                value=dict(chunks=[dict(req['chunks'][0],text='你好。')])
            elif stage=='diarize':
                value=dict(turns=[dict(start=0,end=2,speaker='S01')],clustering={})
            else:
                value=dict(chunks=[dict(id=0,text='你好。',error=None,words=[dict(start=.1,end=.8,text='你好。',chunk_id=0,alignment_status='aligned',timestamp_source='test_fixture')])])
            value.update(seconds=0,status='complete',stage=stage)
            from basic5.common import write_json
            write_json(result_path,value)
            return value,{}
        with patch('basic5.pipeline.run_worker',side_effect=stages):
            result=transcribe(source,self.root/'output',cfg,'unused')
        self.assertEqual(result['segments'],[dict(start=.1,end=.8,text='你好。',speaker='S01',chunk_id=0)])
        self.assertEqual((self.root/'output/transcript.srt').read_text(),'1\n00:00:00,100 --> 00:00:00,800\n[S01] 你好。\n')
        self.assertEqual(sha256(source),before)
        frozen={p.name:sha256(p) for p in (self.root/'output').glob('transcript.*')}
        with self.assertRaises(ValueError):transcribe(source,self.root/'output',cfg,'unused')
        self.assertEqual(frozen,{p.name:sha256(p) for p in (self.root/'output').glob('transcript.*')})

    def test_stage_failure_marks_run_failed(self):
        source=self.root/'a.wav';wav(source)
        with patch('basic5.pipeline.run_worker',side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):transcribe(source,self.root/'failed',self.cfg,'unused')
        self.assertEqual(read_json(self.root/'failed/run.json')['status'],'failed')
        self.assertFalse((self.root/'failed/transcript.json').exists())

    def test_real_owned_process_timeout(self):
        executable=self.root/'sleep-worker'
        executable.write_text('#!'+sys.executable+'\nimport time\ntime.sleep(60)\n')
        executable.chmod(0o755)
        cfg=dict(self.cfg,python=dict(qwen=str(executable)),stage_timeout_seconds=.15)
        with self.assertRaises(subprocess.TimeoutExpired):
            run_worker('asr',self.root/'request',self.root/'result',cfg,self.root/'worker.log')
        record=read_json(self.root/'worker.process.json')
        with self.assertRaises(ProcessLookupError):os.kill(record['pid'],0)


if __name__=='__main__':unittest.main()
