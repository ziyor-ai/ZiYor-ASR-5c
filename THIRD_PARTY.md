# Third-party sources and licenses

The project code is Apache-2.0. The vendored ReDimNet2 implementation retains its upstream MIT license. Model weights are downloaded separately and remain subject to their upstream terms. Keep the model cards and licenses with the downloaded assets.

| Component | Upstream source | License or attribution |
| --- | --- | --- |
| Qwen3-ASR-1.7B | [Model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [Qwen3-ASR SDK](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 for the model and SDK |
| Qwen3-ForcedAligner-0.6B | [Model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), [Qwen3-ASR SDK](https://github.com/QwenLM/Qwen3-ASR) | Apache-2.0 for the model and SDK |
| FSMN-VAD | [Model card](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch), [FunASR SDK](https://github.com/modelscope/FunASR) | The model card was recorded as Apache-2.0 on November 10, 2026; the SDK is MIT. Retain the card for the downloaded snapshot. |
| ReDimNet2-B0 | [Source](https://github.com/PalabraAI/redimnet2), [v1.0.0 checkpoint release](https://github.com/PalabraAI/redimnet2/releases/tag/v1.0.0) | Upstream MIT; see [the vendored license](vendor/ReDimNet2-LICENSE) and the ID R&D copyright notices in the vendored source. |
| Spectral clustering | [Basic5 implementation](basic5/cluster.py), SciPy, scikit-learn | Basic5 implementation: Apache-2.0; SciPy and scikit-learn: BSD-3-Clause |

The setup script installs NumPy (BSD-3-Clause), SoundFile (BSD-3-Clause), PyTorch and torchaudio (see their respective distribution licenses), Transformers (Apache-2.0), and the ModelScope SDK (Apache-2.0) from package indexes. Those dependencies are not vendored.

FFmpeg is a system dependency and is not bundled. Its license depends on the installed build.

[Vendor provenance](docs/vendor-provenance.json) records SHA-256 hashes for the imported ReDimNet2 source. The [model downloader](scripts/download_models.py) records source revisions or URLs and file hashes alongside the model files.
