"""Original non-overlapping low-energy cuts, with explicit sample coverage."""
from pathlib import Path
import shutil
import subprocess
import wave
import numpy as np
from .common import require

RATE = 16000


def inspect_wav(path):
    with wave.open(str(path), "rb") as f:
        require((f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype()) == (1, 2, RATE, "NONE"), "Expected mono16k PCM16")
        return f.getnframes()


def prepare(source, destination, max_seconds):
    require(Path(source).is_file(), "Input must be a local file")
    try:
        frames = inspect_wav(source)
    except (wave.Error, EOFError, ValueError):
        frames = None
    if frames is not None:
        require(0 < frames <= max_seconds * RATE, "Input duration is empty or exceeds configured maximum")
        shutil.copyfile(source, destination)
    else:
        require(shutil.which("ffmpeg") is not None, "Install FFmpeg to decode this input")
        # First audio stream, first channel. Limit+1 lets us detect rather than silently accept truncation.
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(source),
            "-map", "0:a:0", "-vn", "-af", "pan=mono|c0=c0", "-ar", str(RATE),
            "-t", str(max_seconds + 1), "-c:a", "pcm_s16le", str(destination)], check=True, timeout=max(60, max_seconds))
        frames = inspect_wav(destination)
        require(0 < frames <= max_seconds * RATE, "Decoded duration is empty or exceeds configured maximum")
    # Read the payload to reject a truncated WAV whose header promises more samples.
    with wave.open(str(destination), "rb") as f:
        total = 0
        while block := f.readframes(RATE * 60):
            total += len(block)
    require(total == frames * 2, "Truncated WAV data")
    return frames


def read_samples(path, start=0, end=None):
    with wave.open(str(path), "rb") as f:
        f.setpos(start)
        size = f.getnframes() - start if end is None else end - start
        raw = f.readframes(size)
    require(len(raw) == size * 2, "Truncated sample range")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768)


def plan_chunks(path, cfg):
    frames = inspect_wav(path)
    maximum = round(cfg["chunk_seconds"] * RATE)
    # Fractional seconds can round to the full block length; always advance.
    search = min(maximum - 1, round(cfg["cut_search_seconds"] * RATE))
    start, chunks = 0, []
    while start < frames:
        end = min(frames, start + maximum)
        if end < frames and search:
            low = end - search
            area = read_samples(path, max(start, low - RATE // 20), min(frames, end + RATE // 20))
            origin = max(start, low - RATE // 20)
            candidates = range(low, end + 1, RATE // 10)
            end = min(candidates, key=lambda x: float(np.mean(area[max(0, x-origin-RATE//20):x-origin+RATE//20].astype(np.float64)**2)))
        chunks.append(dict(id=len(chunks), start_frame=start, end_frame=end,
            start=start/RATE, end=end/RATE))
        start = end
    require(sum(x["end_frame"]-x["start_frame"] for x in chunks) == frames, "Chunk coverage failed")
    return chunks
