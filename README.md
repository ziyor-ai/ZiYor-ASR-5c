# Meeting ASR Basic5

**五项核心技术的中文会议转写基础版。** 将本地录音或视频转成带文字、说话人和时间的 JSON、TXT、SRT。这里的“基础版”指功能组合精简，不是限制硬件或故意降低识别质量。

| 核心技术 | 作用 |
|---|---|
| Qwen3-ASR-1.7B | 识别文字 |
| FSMN-VAD | 检测语音区间 |
| ReDimNet2-B0 | 提取声纹 |
| 谱聚类 | 估计人数并串联同一说话人 |
| Qwen3-ForcedAligner-0.6B | 给已有文字标词时间 |

这是一套重新组合的基础流程。**尚未测得本版本的 AliMeeting CER/cpCER/DER，不引用 U08 或 E002 的分数。** 不含多模型融合、大模型复核、语音分离和热词检索。详细取舍见 [架构说明](docs/architecture.md)。

## 运行条件

- Linux x86_64、Python 3.12、FFmpeg、NVIDIA 驱动。安装脚本默认 PyTorch 2.10 / CUDA 12.8 构建。
- 没有承诺固定 8GB 显存门槛；这是功能基础版。实际峰值随录音、模型和运行环境变化，运行日志会记录。
- Qwen 与说话人依赖分开安装，避免 Transformers 版本互相覆盖；模型顺序加载。
- 可将配置改成 `device: "cpu"`、`dtype: "float32"` 做 CPU 尝试，但未提供 CPU 速度保证。Apple MPS 不在本版支持范围。
- 模型单独下载，磁盘需要同时容纳模型、两套推理依赖和中间音频；本次未测量完整安装占用。

## 安装

```bash
# Ubuntu/Debian：安装 Python 3.12/venv 和 FFmpeg 后，在项目目录执行
bash scripts/setup.sh

# 执行模型下载；首次下载需要网络。
.venvs/core/bin/python scripts/download_models.py --config config.json

# 检查两个模型环境的实际导入和 CUDA，不执行转写
.venvs/core/bin/python -m basic5 doctor --config config.json
```

安装脚本不会下载或编译 vLLM/FlashInfer。使用已有模型时，跳过下载，编辑 `config.json` 的四个本地模型路径和两个解释器路径即可。相对路径基于配置文件所在目录。ReDimNet 必须使用官方 `b0-vox2-lm.pt`，代码会检查 SHA256。

仅安装源码／查看组件，不需要 GPU 或模型：

```bash
python3.12 -m venv .venvs/core
.venvs/core/bin/python -m pip install -e .
.venvs/core/bin/python -m basic5 components
```

## 转写

```bash
.venvs/core/bin/python -m basic5 transcribe /path/to/meeting.wav \
  --config config.json --output outputs/meeting-001

# MP3、MP4 等由 FFmpeg 读取第一条音轨的第一声道
.venvs/core/bin/python -m basic5 transcribe /path/to/meeting.mp4 \
  --config config.json --output outputs/meeting-002
```

输出目录必须是新目录，重复执行会拒绝覆盖。模型推理阶段只使用本地模型；不会上传录音或调用云 ASR。阶段失败会返回非零退出码，并在 `run.json` 标记失败，不会输出假的成功记录。可按 Ctrl-C 终止，程序会清理自己的模型子进程。

| 输出 | 用途 |
|---|---|
| `transcript.json` | 完整文字、词时间、说话人、分人区间、对齐与归人不确定性 |
| `transcript.txt` | 可读逐字稿，保留所有文字 |
| `transcript.srt` | 字幕；无法表达的零时长条目不写入，完整稿以 JSON/TXT 为准 |
| `run.json` | 实际组件、配置、阶段耗时、输入与输出 SHA256、成功或失败状态 |
| `logs/` | 每个阶段的日志、PID、命令和退出记录 |
| `work/` | 标准化录音及各阶段输出，用于复核；包含私人录音，不要提交到公开仓库 |

几个主要参数：

- `num_speakers`: 默认 `null`，自动估计；已知人数可设为整数。
- `max_speakers`: 默认 8；过大可能放大短窗噪声。
- `max_audio_seconds`: 输入最长时长，超出会明确失败，不悄悄截断。
- `stage_timeout_seconds`: 每个模型阶段的时间上限，默认 4 小时。
- `chunk_seconds`: 默认 28 秒，`cut_search_seconds: 6` 在块尾前 6 秒寻找较低能量切点；所有音频样本连续覆盖。

## 能力边界

这是标准串联方案，自动人数和换人边界属于估计。重叠发言、很短的插话、远场噪声仍可能识别不全或归人错误。没有采用“先分离再识别”，不会声称已经完整恢复多人同时发言。

对齐失败会保留整块文字，并标记 `unaligned_text_preserved`，不会编造词时间。没有可靠归属证据时显示 `unknown`。SDK 给出的时间包含其自身后处理，不能视为人工精确边界。

## 验证与开发

```bash
python3.12 -m venv .venvs/test
.venvs/test/bin/python -m pip install -e . scipy scikit-learn
.venvs/test/bin/python -m unittest discover -s tests -v
```

测试覆盖音频连续切块、Unicode/标点保留、对齐回退、未知说话人、谱聚类退化情况、子进程超时、文件导出和拒绝覆盖。**这些是代码行为检查，不是模型准确率测试。** 本次打包验证情况见 [验证说明](docs/validation.md)。

## 开源与发布

胶水代码 Apache-2.0，ReDimNet 第三方实现 MIT，详见 [THIRD_PARTY.md](THIRD_PARTY.md)、`LICENSE`、`NOTICE`。模型权重和任何用户录音均不打包。下载模型时保留各自上游许可及模型卡。

本项目已准备为独立源码包，尚未由本次任务公开发布。上传仓库前保留 `.gitignore`，不要上传 `models/`、`outputs/`、`config.json`、令牌或私人日志。
