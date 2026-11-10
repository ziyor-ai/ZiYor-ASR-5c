# Validation for Basic5 v0.1.0

The baseline checks recorded on November 10, 2026 covered source code, Python packaging, and behavior that can be tested on a CPU. They ran on macOS arm64 with Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.9.0, and FFmpeg.

| Check | Recorded result |
| --- | --- |
| Source regression suite | 20 tests passed |
| Clean wheel installs | The same 20 tests passed outside the source tree in two separate environments |
| Installed CLI | `components` listed five components and `--help` succeeded |
| Linux runtime dependency resolution | Both Python 3.12 inference environments resolved under the PyTorch 2.10.0 constraint |
| Source integrity | Python sources parsed; hashes for 11 vendored ReDimNet2 files matched the extraction record |
| Packaging | The wheel included the vendored ReDimNet2 modules and MIT license; the source distribution included the provenance record; `bash -n` accepted the setup script |

The tests cover audio conversion and chunk coverage, rejection of oversized input, text and punctuation preservation, alignment fallback, ambiguous speakers, clustering edge cases, output serialization, overwrite protection, and subprocess cleanup on timeout. The export integration test supplies stage results as fixtures; it checks how the pipeline handles those results.

## Open validation work

- Run all four models on a real recording and review the resulting transcript, labels, and timestamps.
- Measure peak GPU and host memory, runtime relative to recording duration, and installation footprint on Linux/NVIDIA.
- Measure character error rate (CER), concatenated minimum-permutation CER (cpCER), diarization error rate (DER), and word boundary error against annotated data.
- Characterize performance on overlapping speech, brief interjections, distant microphones, and long meetings.

To reproduce the CPU checks, follow [Contributing](../CONTRIBUTING.md#run-the-tests). For an inference smoke test, follow the [quick start](../README.md#-quick-start), then transcribe a short recording with two known speakers and a reference transcript. Compare the transcript and speaker changes to the recording, and inspect `quality_flags` and the stage logs before reporting results.
