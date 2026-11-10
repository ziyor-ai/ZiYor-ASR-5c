# 第三方来源与许可

本项目胶水代码采用 Apache-2.0；`vendor/redimnet2` 保持上游 MIT 许可。模型权重不随源码分发，下载后保留上游模型卡及许可证。源码许可证不会替代模型许可证。

| 组件 | 来源 | 许可／说明 |
|---|---|---|
| Qwen3-ASR-1.7B | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | 模型卡 Apache-2.0；SDK https://github.com/QwenLM/Qwen3-ASR 同为 Apache-2.0 |
| Qwen3-ForcedAligner-0.6B | https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B | 模型卡 Apache-2.0；SDK 同上 |
| FSMN-VAD | https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch | 官方模型卡标注 Apache License 2.0（2026-11-10 核验）；FunASR SDK（https://github.com/modelscope/FunASR）为 MIT；下载后仍应保留对应快照模型卡 |
| ReDimNet2-B0 | https://github.com/PalabraAI/redimnet2 | 上游仓库声明 MIT；保留 vendor/ReDimNet2-LICENSE 及源文件中的 ID R&D 版权说明；权重取官方 v1.0.0 release |
| 谱聚类实现 | 本项目 cluster.py；调用 SciPy、scikit-learn | 本项目代码 Apache-2.0；SciPy/scikit-learn 为 BSD-3-Clause |

NumPy（BSD-3-Clause）、SoundFile（BSD-3-Clause）、PyTorch/torchaudio（见各自分发许可）、Transformers（Apache-2.0）、ModelScope SDK（Apache-2.0）由安装脚本从官方包源获取，不复制进源码包。

FFmpeg 是系统依赖，许可取决于用户安装的构建配置；本包不分发 FFmpeg 二进制。

`docs/vendor-provenance.json` 记录从原项目提取的 ReDimNet2 文件 SHA256。`scripts/download_models.py` 在模型目录记录实际下载版本和文件散列，不把访问凭证写入项目。
