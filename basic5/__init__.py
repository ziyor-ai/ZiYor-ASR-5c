"""Meeting ASR Basic5: a deliberately small, five-component baseline."""
__version__ = "0.1.0"
COMPONENTS = (
    "Qwen3-ASR-1.7B", "FSMN-VAD", "ReDimNet2-B0",
    "spectral-clustering", "Qwen3-ForcedAligner-0.6B",
)
