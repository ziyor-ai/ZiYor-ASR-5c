#!/usr/bin/env python3
"""Explicit public model acquisition. Inference itself does not download models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from basic5.config import load_config
from basic5.common import write_json, sha256

REPOS={"asr":"Qwen/Qwen3-ASR-1.7B","aligner":"Qwen/Qwen3-ForcedAligner-0.6B"}
REVISIONS={"asr":"7278e1e70fe206f11671096ffdd38061171dd6e5",
    "aligner":"c7cbfc2048c462b0d63a45797104fc9db3ad62b7"}
REDIM_URL="https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/b0-vox2-lm.pt"
REDIM_SHA="4b5a8c00f813ab0d92fc81ea7c504475d8201bed608fb2c6aaba65be723e9c61"


def inventory(path):
    return {str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob("*"))
        if p.is_file() and ".cache" not in p.parts and p.name!="basic5-source.json"}


def worker(component,cfg):
    target=Path(cfg["models"][component])
    if component in REPOS:
        from huggingface_hub import HfApi,snapshot_download
        revision=HfApi().model_info(REPOS[component],revision=REVISIONS[component]).sha
        snapshot_download(REPOS[component],revision=revision,local_dir=target)
        record=dict(source=REPOS[component],revision=revision,provider="huggingface",files=inventory(target))
        write_json(target/"basic5-source.json",record)
    elif component=="vad":
        from modelscope import snapshot_download
        snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",revision="master",local_dir=str(target))
        write_json(target/"basic5-source.json",dict(source="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            requested_revision="master",provider="modelscope",files=inventory(target)))
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target)!=REDIM_SHA:raise ValueError("Existing ReDimNet file differs; refusing overwrite")
            return
        part=target.with_suffix(".download")
        if part.exists():raise FileExistsError(f"Unfinished download exists: {part}")
        with urllib.request.urlopen(REDIM_URL,timeout=60) as response,part.open("xb") as stream:
            while data:=response.read(1024*1024):stream.write(data)
        if sha256(part)!=REDIM_SHA:raise ValueError("ReDimNet checksum mismatch; partial retained")
        part.replace(target)
        write_json(target.with_suffix(".source.json"),dict(url=REDIM_URL,sha256=REDIM_SHA))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--component",choices=("all","asr","aligner","vad","speaker"),default="all")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args();cfg=load_config(args.config,check_assets=False)
    if args.worker:
        worker(args.component,cfg);return
    components=("asr","aligner","vad","speaker") if args.component=="all" else (args.component,)
    env=dict(os.environ)
    for key in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE"):env.pop(key,None)
    for component in components:
        role="qwen" if component in REPOS else "speaker"
        subprocess.run([cfg["python"][role],str(Path(__file__).resolve()),"--config",str(Path(args.config).resolve()),
            "--component",component,"--worker"],env=env,check=True)
    print("Models downloaded. Keep upstream model cards/licenses and basic5-source.json receipts.")


if __name__=="__main__":main()
