# Vision

Basic5 gives meeting transcription work a small, inspectable starting point. The recognizer supplies the text, the aligner supplies word times, and diarization estimates who spoke when. Keeping these results separate makes it possible to investigate a bad subtitle and tell whether recognition, alignment, or speaker assignment caused it.

The transcript should preserve what the recognizer returned even when another stage cannot time it or assign it to a speaker. Chunk-bound fallbacks and `unknown` labels make those gaps visible. Local inference and per-stage records make the same pipeline useful for private recordings and for repeatable experiments.

## Roadmap

1. **Establish a real inference baseline.** Run a representative two-speaker recording on Linux/NVIDIA. Record model revisions, runtime, peak memory, and stage outcomes.
2. **Measure each source of error.** Report CER, cpCER, DER, and word boundary error separately. Include examples of overlapping speech, short turns, distant microphones, and long recordings.
3. **Make runs easier to reproduce.** Record a tested runtime configuration and document known failure modes, supported hardware, and which model snapshots were measured.

Speaker identification across recordings, source separation, and model ensembles are outside the scope of this baseline. See [validation](docs/validation.md) for the current evidence and [contribution guidelines](CONTRIBUTING.md) for ways to help.
