import os
from pathlib import Path
from .common import read_json, require

DEFAULTS = dict(device="cuda:0", dtype="bfloat16", language="Chinese", threads=4,
    max_audio_seconds=14400, stage_timeout_seconds=14400, max_new_tokens=2048,
    chunk_seconds=28, cut_search_seconds=6, speaker_window_seconds=2.0,
    speaker_hop_seconds=1.0, min_speaker_window_seconds=0.6, num_speakers=None,
    max_speakers=8, max_cluster_windows=2000, single_speaker_similarity=0.8,
    min_word_speaker_coverage=0.5, speaker_ambiguity_margin=0.2, seed=42)


def load_config(path, check_assets=True):
    path = Path(path).resolve(strict=True)
    supplied = read_json(path)
    require(isinstance(supplied, dict), "Config must be an object")
    require(not set(supplied) - (set(DEFAULTS) | {"python", "models"}), "Unknown config field")
    cfg = dict(DEFAULTS, **supplied)
    require(set(cfg.get("python", {})) == {"qwen", "speaker"}, "Expected python.qwen and python.speaker")
    require(set(cfg.get("models", {})) == {"asr", "aligner", "vad", "speaker"}, "Expected exactly four model paths")
    for group in ("python", "models"):
        cfg[group] = {k: str((path.parent / Path(v).expanduser()).absolute()) for k, v in cfg[group].items()}
    for k in ("threads", "max_new_tokens", "max_speakers", "max_cluster_windows", "seed"):
        require(type(cfg[k]) is int and cfg[k] >= (0 if k == "seed" else 1), f"Invalid {k}")
    for k in ("max_audio_seconds", "stage_timeout_seconds", "chunk_seconds", "speaker_window_seconds", "speaker_hop_seconds", "min_speaker_window_seconds"):
        require(type(cfg[k]) in (int, float) and 0 < cfg[k] < float("inf"), f"Invalid {k}")
    require(1 <= cfg["chunk_seconds"] <= 30, "chunk_seconds must be in [1,30]")
    require(type(cfg["cut_search_seconds"]) in (int, float) and 0 <= cfg["cut_search_seconds"] < cfg["chunk_seconds"], "Invalid cut search")
    require(0 < cfg["min_speaker_window_seconds"] <= cfg["speaker_hop_seconds"] <= cfg["speaker_window_seconds"] <= 10, "Invalid speaker windows")
    for k in ("single_speaker_similarity", "min_word_speaker_coverage", "speaker_ambiguity_margin"):
        require(type(cfg[k]) in (int, float) and 0 <= cfg[k] <= 1, f"Invalid {k}")
    require(cfg["max_speakers"] <= cfg["max_cluster_windows"] <= 5000, "Cluster matrix bound invalid")
    n = cfg["num_speakers"]
    require(n is None or type(n) is int and 1 <= n <= cfg["max_speakers"], "Invalid num_speakers")
    require(cfg["dtype"] in {"float32", "float16", "bfloat16"}, "Invalid dtype")
    require(cfg["device"] == "cpu" or cfg["device"].startswith("cuda:"), "Supported devices: cpu or cuda:N")
    require(cfg["device"] != "cpu" or cfg["dtype"] == "float32", "CPU requires float32")
    require(cfg["language"] in ("Chinese", "English"), "Basic5 supports Chinese or English")
    if check_assets:
        for key, executable in cfg["python"].items():
            require(Path(executable).is_file() and os.access(executable, os.X_OK), f"Missing {key} interpreter: {executable}")
        for key, model in cfg["models"].items():
            require(Path(model).is_file() if key == "speaker" else Path(model).is_dir(), f"Missing {key} model: {model}")
    return cfg
