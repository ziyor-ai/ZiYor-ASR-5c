# Contributing

Bug reports and pull requests are welcome. For a failure, include the command, platform, Python version, failing stage, and a minimal reproduction. Use synthetic or shareable audio. Remove private transcripts, local paths, and credentials from any excerpt before posting it.

## Run the tests

Use Python 3.12 and install FFmpeg on your PATH. The regression suite does not require model weights or a GPU.

```bash
python3.12 -m venv .venvs/test
.venvs/test/bin/python -m pip install -e . -r requirements/base.txt scipy scikit-learn
.venvs/test/bin/python -m unittest discover -s tests -v
bash -n scripts/setup.sh
```

If a change affects model inference, also run `doctor` and a real transcription with the relevant models. Include the versions tested and note any stages that were not exercised. Keep the Chinese and Unicode fixtures: they check preservation of the text this pipeline processes.

## Review and packaging

- Document changes to CLI arguments, configuration, or exported fields alongside the code.
- For changes to vendored code or models, update the relevant provenance and license records.
- Before distributing a source ZIP, refresh `SOURCE_MANIFEST.json` against the Git snapshot. It records relative paths, byte counts, file modes, and SHA-256 hashes, excluding itself. Archive the reviewed commit with `git archive`.
- Keep weights, virtual environments, generated output, and private config out of the repository.
