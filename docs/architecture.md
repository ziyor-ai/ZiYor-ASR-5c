# Architecture

Basic5 uses four models and one clustering algorithm. FFmpeg conversion, chunk planning, timestamp matching, and export support those components.

| Component | Input → output |
| --- | --- |
| Qwen3-ASR-1.7B | Consecutive audio chunks → transcript text |
| FSMN-VAD | Normalized audio → speech intervals |
| ReDimNet2-B0 | Short speech windows → speaker embeddings |
| Spectral clustering | Speaker embeddings → recording-local speaker labels and turns |
| Qwen3-ForcedAligner-0.6B | Audio chunks and recognized text → SDK postprocessed word timestamps |

The [pipeline diagram](../README.md#-how-it-works) shows the data dependencies. The coordinator runs ASR, diarization (VAD, embeddings, and clustering), and alignment sequentially. Each stage gets its own process, and the Qwen and speaker workers have separate dependency environments. Exiting a worker releases its models. Inference uses the local model paths from `config.json`.

## Chunking and speaker turns

ASR uses contiguous, non-overlapping chunks, with a default search for a low-energy cut between 22 and 28 seconds. Each sample belongs to one chunk. Cuts can still affect recognition near chunk boundaries.

Automatic speaker counting is heuristic. Set `num_speakers` if the count is known. One usable embedding window yields one speaker; without usable windows the system retains the text and labels the speaker `unknown`. By default, up to 2,000 evenly sampled windows enter the spectral decomposition. The other windows are assigned to their closest cluster center to keep the matrix bounded.

For disconnected affinity graphs, a small edge weight makes the normalization well-defined and the speaker estimate uses the component count, subject to the configured limit. If embeddings cancel out when computing a cluster center, the first valid vector in that cluster represents it and the result records `cancelled_centers`.

Diarization produces one main speaker activity track. It does not separate simultaneous speakers. Labels apply only within the current recording, and the VoxCeleb2 speaker model may behave differently on distant-microphone Chinese meetings.

## Alignment and export

The aligner proposes word boundaries for text already produced by ASR. Basic5 checks the aligned text against the original transcript and preserves the original punctuation and spacing. It uses overlap with diarized turns to attribute words, marking uncertain assignments as `unknown`.

If an alignment is unusable, Basic5 retains the whole chunk's text with chunk bounds and an unknown speaker. Zero-duration words stay in JSON and TXT; any SRT cue whose duration would round to zero is omitted. The SDK's postprocessed word boundaries require separate validation against human annotations.

Basic5 leaves out speech separation, model ensembling, hotword retrieval, and language-model review. The [validation notes](validation.md) describe the checks that apply to this pipeline.
