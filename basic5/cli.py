import argparse
import json
import sys
from pathlib import Path
from . import COMPONENTS, __version__


def main(argv=None):
    parser=argparse.ArgumentParser(description="Meeting ASR Basic5 — five core components, no ensemble")
    parser.add_argument("--version",action="version",version=__version__)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("components",help="List the five components; no model import")
    check=commands.add_parser("doctor",help="Check configured runtimes; no model inference")
    check.add_argument("--config",required=True)
    run=commands.add_parser("transcribe",help="Transcribe a local audio/video file")
    run.add_argument("input",type=Path)
    run.add_argument("--config",required=True)
    run.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=="components":
            print(json.dumps(dict(version=__version__,components=list(COMPONENTS)),ensure_ascii=False,indent=2))
            return 0
        from .config import load_config
        cfg=load_config(args.config)
        from .pipeline import doctor, transcribe
        if args.command=="doctor":
            print(json.dumps(doctor(cfg),ensure_ascii=False,indent=2))
        else:
            transcribe(args.input,args.output,cfg,args.config)
            print(str(args.output.absolute()/"transcript.json"))
        return 0
    except KeyboardInterrupt:
        print("Cancelled; task-owned workers stopped.",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 1
