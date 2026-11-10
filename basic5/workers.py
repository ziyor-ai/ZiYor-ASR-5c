"""Isolated real model workers. No mock backend or hidden cloud service."""
import argparse
import importlib.metadata
import os
import sys
import time
from pathlib import Path
from .common import require, read_json, write_json, sha256
from .audio import RATE, read_samples, inspect_wav

REDIM_SHA256 = "4b5a8c00f813ab0d92fc81ea7c504475d8201bed608fb2c6aaba65be723e9c61"


def event(stage, **fields):
    import json
    print(json.dumps(dict(stage=stage,pid=os.getpid(),**fields),ensure_ascii=False),flush=True)


def setup(cfg):
    import torch
    import numpy as np
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["device"].startswith("cuda:"):
        require(torch.cuda.is_available(), "CUDA unavailable in this interpreter")
        torch.cuda.set_device(cfg["device"])
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch, getattr(torch,cfg["dtype"])


def asr(request):
    cfg=request["config"]
    torch,dtype=setup(cfg)
    from qwen_asr import Qwen3ASRModel
    model=Qwen3ASRModel.from_pretrained(cfg["models"]["asr"],dtype=dtype,
        device_map=cfg["device"],attn_implementation="sdpa",local_files_only=True,
        trust_remote_code=False,max_inference_batch_size=1,max_new_tokens=cfg["max_new_tokens"])
    model.model.eval()
    model.model.generation_config.do_sample=False
    if hasattr(model.model,"thinker"):
        model.model.thinker.generation_config.do_sample=False
    original_generate=model.model.generate
    generation_records=[]

    def observed_generate(*args,**kwargs):
        result=original_generate(*args,**kwargs)
        require(hasattr(result,"sequences") and "input_ids" in kwargs,"Unexpected qwen-asr generation API")
        generated=result.sequences[:,kwargs["input_ids"].shape[1]:]
        require(len(generated)==1,"Expected batch size one")
        eos=model.processor.tokenizer.eos_token_id
        eos_ids=eos if isinstance(eos,(list,tuple)) else [eos]
        ended=generated.shape[1]>0 and int(generated[0,-1]) in eos_ids
        count=int(generated.shape[1])
        require(count<cfg["max_new_tokens"] or ended,"ASR reached token limit without EOS; refusing incomplete output")
        generation_records.append(dict(generated_tokens=count,ended_with_eos=ended))
        return result

    # Observer only: passes original arguments/output through unchanged.
    model.model.generate=observed_generate
    results=[]
    for chunk in request["chunks"]:
        wave=read_samples(request["audio"],chunk["start_frame"],chunk["end_frame"])
        before=len(generation_records);started=time.monotonic()
        with torch.inference_mode():
            result=model.transcribe(audio=(wave,RATE),language=cfg["language"],return_time_stamps=False)
        require(len(result)==1 and isinstance(result[0].text,str),"Unexpected ASR return value")
        require(len(generation_records)>before,"ASR generation observer was not called")
        results.append(dict(chunk,text=result[0].text,language=result[0].language,
            seconds=time.monotonic()-started,generation=generation_records[before:]))
        event("asr",completed=len(results),total=len(request["chunks"]))
    return dict(chunks=results)


def diarize(request):
    import numpy as np
    from .cluster import speaker_windows, cluster_embeddings, windows_to_turns
    cfg=request["config"]
    torch,_=setup(cfg)
    from funasr import AutoModel
    vad_model=AutoModel(model=cfg["models"]["vad"],device="cpu",disable_update=True,
        disable_pbar=True,trust_remote_code=False)
    frames=inspect_wav(request["audio"])
    intervals=[]
    for offset in range(0,frames,60*RATE):
        end=min(frames,offset+60*RATE)
        results=vad_model.generate(input=read_samples(request["audio"],offset,end),fs=RATE,cache={},disable_pbar=True)
        require(isinstance(results,list),"Unexpected FSMN-VAD result")
        for item in results:
            require("value" in item,"FSMN result missing value")
            for start_ms,end_ms in item["value"]:
                require(0<=start_ms<end_ms<=(end-offset)/RATE*1000+1,"Invalid VAD timestamps")
                a=offset/RATE+start_ms/1000
                b=min(end/RATE,offset/RATE+end_ms/1000)
                intervals.append([a,b])
    vad=[]
    for a,b in sorted(intervals):
        if vad and a<=vad[-1][1]:
            vad[-1][1]=max(vad[-1][1],b)
        else:
            vad.append([a,b])
    del vad_model
    windows=speaker_windows(vad,cfg)
    if not windows:
        return dict(vad=vad,windows=[],turns=[],clustering=dict(estimated_speakers=0,reason="no_usable_speaker_window"))
    # Load only the known B0 checkpoint, not arbitrary pickled model code.
    require(sha256(cfg["models"]["speaker"])==REDIM_SHA256,"ReDimNet B0 checkpoint SHA256 differs")
    from redimnet2.redimnet2 import ReDimNet2Wrap
    state=torch.load(cfg["models"]["speaker"],map_location="cpu",weights_only=True)
    embedder=ReDimNet2Wrap(**state["model_config"])
    embedder.load_state_dict(state["state_dict"],strict=True)
    del state
    embedder=embedder.float().eval().to(cfg["device"])
    features=[]
    for i,w in enumerate(windows):
        wave=read_samples(request["audio"],round(w["start"]*RATE),round(w["end"]*RATE))
        with torch.inference_mode():
            z=embedder(torch.from_numpy(wave.copy())[None].to(cfg["device"])).detach().cpu().numpy().reshape(-1)
        require(np.isfinite(z).all(),"Nonfinite speaker embedding")
        features.append(z)
        if (i+1)%50==0 or i+1==len(windows):
            event("speaker_embeddings",completed=i+1,total=len(windows))
    labels,report=cluster_embeddings(np.stack(features),cfg)
    return dict(vad=vad,windows=[dict(w,speaker=s) for w,s in zip(windows,labels)],
        turns=windows_to_turns(windows,labels,vad),clustering=report)


def align(request):
    from .assembly import align_or_fallback, fallback
    cfg=request["config"]
    torch,dtype=setup(cfg)
    from qwen_asr import Qwen3ForcedAligner
    model=Qwen3ForcedAligner.from_pretrained(cfg["models"]["aligner"],dtype=dtype,
        device_map=cfg["device"],attn_implementation="sdpa",local_files_only=True,trust_remote_code=False)
    chunks=read_json(request["asr_result"])["chunks"]
    results=[]
    for chunk in chunks:
        reason=None
        if not chunk["text"]:
            words=[]
        else:
            wave=read_samples(request["audio"],chunk["start_frame"],chunk["end_frame"])
            try:
                with torch.inference_mode():
                    raw=model.align(audio=(wave,RATE),text=chunk["text"],language=cfg["language"])[0]
                items=raw.items if hasattr(raw,"items") and not callable(raw.items) else raw
                units=[dict(start=float(u.start_time),end=float(u.end_time),text=u.text) for u in items]
                words,reason=align_or_fallback(chunk,units,request["duration"])
            except (ValueError,IndexError,KeyError,AssertionError) as error:
                # OOM, missing files, model-load/worker failures must propagate.
                reason=f"{type(error).__name__}: {error}"
                words=fallback(chunk,reason)
        results.append(dict(id=chunk["id"],text=chunk["text"],words=words,error=reason))
        event("align",completed=len(results),total=len(chunks))
    return dict(chunks=results)


def probe(role, cfg):
    torch,_=setup(cfg)
    packages=("torch","torchaudio","qwen-asr","transformers") if role=="qwen" else ("torch","torchaudio","funasr","numpy","scipy","scikit-learn")
    if role=="qwen":
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    else:
        from funasr import AutoModel
        from redimnet2.redimnet2 import ReDimNet2Wrap
    return dict(role=role,python=sys.version,executable=sys.executable,
        versions={p:importlib.metadata.version(p) for p in packages},
        device=cfg["device"],cuda_available=torch.cuda.is_available(),model_inference=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",choices=("asr","diarize","align","probe-qwen","probe-speaker"),required=True)
    ap.add_argument("--request",required=True)
    ap.add_argument("--result",required=True)
    args=ap.parse_args()
    request=read_json(args.request)
    started=time.monotonic()
    if args.stage.startswith("probe-"):
        result=probe(args.stage.split("-",1)[1],request["config"])
    else:
        require(sha256(request["audio"])==request["audio_sha256"],"Worker input hash changed")
        result={"asr":asr,"diarize":diarize,"align":align}[args.stage](request)
    import torch
    result.update(status="complete",stage=args.stage,pid=os.getpid(),seconds=time.monotonic()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None)
    write_json(args.result,result)


if __name__=="__main__":
    main()
