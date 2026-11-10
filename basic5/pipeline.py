"""Sequential process isolation. A failed stage cannot become a successful transcript."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import COMPONENTS, __version__
from .audio import prepare, plan_chunks, RATE
from .assembly import assemble, export_text, export_srt
from .common import require, read_json, write_json, sha256


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def child_env():
    env=dict(os.environ)
    root=Path(__file__).resolve().parent.parent
    env["PYTHONPATH"]=os.pathsep.join((str(root),str(root/"vendor")))
    env.update(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
    return env


def stop_owned(process):
    # Kill the task-owned group even if its leader exited and left descendants.
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


def run_worker(stage, request_path, result_path, cfg, log_path):
    role="qwen" if stage in ("asr","align","probe-qwen") else "speaker"
    command=[cfg["python"][role],"-B","-m","basic5.workers","--stage",stage,
        "--request",str(request_path),"--result",str(result_path)]
    record=dict(stage=stage,command=command,started_at=now(),log=str(log_path))
    started=time.monotonic()
    previous=signal.getsignal(signal.SIGTERM)
    def cancelled(signum,frame):
        raise KeyboardInterrupt("Termination requested")
    signal.signal(signal.SIGTERM,cancelled)
    try:
        with Path(log_path).open("xb") as log:
            process=subprocess.Popen(command,env=child_env(),stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            record["pid"]=process.pid
            print(f"[{stage}] PID={process.pid}; log={log_path}",flush=True)
            try:
                rc=process.wait(timeout=cfg["stage_timeout_seconds"])
                record["returncode"]=rc
                require(rc==0,f"{stage} failed with exit {rc}; see {log_path}")
            finally:
                stop_owned(process)
        result=read_json(result_path)
        require(result.get("status")=="complete" and result.get("stage")==stage,"Invalid worker completion receipt")
        return result,record
    finally:
        record.update(ended_at=now(),wall_seconds=time.monotonic()-started)
        write_json(Path(log_path).with_suffix(".process.json"),record)
        signal.signal(signal.SIGTERM,previous)


def transcribe(source, output, cfg, config_path):
    output=Path(output).absolute()
    require(not output.exists(),"Output already exists; choose a new directory")
    source=Path(source).resolve(strict=True)
    output.mkdir(parents=True,exist_ok=False)
    work=output/"work";work.mkdir()
    logs=output/"logs";logs.mkdir()
    report=dict(schema_version=1,status="running",version=__version__,components=list(COMPONENTS),
        started_at=now(),source=str(source),source_sha256=sha256(source),stages=[],
        quality_benchmark="not_established_for_Basic5",configuration=cfg)
    write_json(output/"run.json",report)
    try:
        audio=work/"audio.wav"
        frames=prepare(source,audio,cfg["max_audio_seconds"])
        chunks=plan_chunks(audio,cfg)
        request=dict(config=cfg,audio=str(audio),audio_sha256=sha256(audio),duration=frames/RATE,chunks=chunks)
        report.update(duration=frames/RATE,audio_sha256=request["audio_sha256"],chunk_count=len(chunks))
        request_path=work/"request.json"
        write_json(request_path,request)
        asr_result,_=run_worker("asr",request_path,work/"asr.json",cfg,logs/"asr.log")
        report["stages"].append(dict(name="asr",seconds=asr_result["seconds"]))
        write_json(output/"run.json",report)
        diarization,_=run_worker("diarize",request_path,work/"diarization.json",cfg,logs/"diarize.log")
        report["stages"].append(dict(name="diarize",seconds=diarization["seconds"]))
        request["asr_result"]=str(work/"asr.json")
        write_json(request_path,request)
        write_json(output/"run.json",report)
        alignment,_=run_worker("align",request_path,work/"alignment.json",cfg,logs/"align.log")
        report["stages"].append(dict(name="align",seconds=alignment["seconds"]))
        result=assemble(asr_result["chunks"],alignment["chunks"],diarization["turns"],cfg,frames/RATE)
        result.update(schema_version=1,system="meeting-asr-basic5",version=__version__,status="complete",
            duration=frames/RATE,components=list(COMPONENTS),asr_chunks=asr_result["chunks"],
            clustering=diarization["clustering"],source_sha256=report["source_sha256"],
            audio_sha256=request["audio_sha256"],quality_benchmark="unmeasured")
        require(sha256(source)==report["source_sha256"] and sha256(audio)==request["audio_sha256"],"Input changed during run")
        write_json(output/"transcript.json",result)
        (output/"transcript.txt").write_text(export_text(result),encoding="utf-8")
        (output/"transcript.srt").write_text(export_srt(result),encoding="utf-8")
        report.update(status="complete",artifacts={name:sha256(output/name) for name in ("transcript.json","transcript.txt","transcript.srt")})
        return result
    except BaseException as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report["ended_at"]=now()
        write_json(output/"run.json",report)


def doctor(cfg):
    import tempfile
    with tempfile.TemporaryDirectory(prefix="basic5-doctor-") as folder:
        root=Path(folder)
        write_json(root/"request.json",dict(config=cfg))
        result={}
        for role in ("qwen","speaker"):
            try:
                value,_=run_worker("probe-"+role,root/"request.json",root/(role+".json"),cfg,root/(role+".log"))
                result[role]=value
            except Exception:
                path=root/(role+".log")
                if path.exists():
                    print(path.read_text(errors="replace")[-4000:],file=sys.stderr)
                raise
        return result
