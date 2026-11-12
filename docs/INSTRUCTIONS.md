# Instructions

## Set up (once)

You need Python 3.11–3.12 and [uv](https://docs.astral.sh/uv/). About 2 GB of disk for the weights and 5 GB of RAM.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

On an Intel Mac this installs PyTorch 2.2.2, the last version built for Intel Macs. Everywhere else it installs a current version. The model code works on both.

## Run the model

```bash
.venv/bin/python -m replicatesota.generate "The capital of France is"
```

The first run downloads Qwen3.5-0.8B (about 1.7 GB) from Hugging Face. The exact revision is pinned in `replicatesota/load.py`.

## Check it matches the official model

```bash
.venv/bin/python -m pytest
```

- `tests/test_units.py` runs in seconds and needs no weights. It checks that the fast and step-by-step DeltaNet forms agree, that cached decoding matches a full forward pass, causality, and RoPE.
- `tests/test_logit_match.py` compares against the reference fixtures in `tests/fixtures/qwen35/`. It is skipped until those exist.

The fixtures and the full every-position comparison are made by running the official `transformers` code. That needs a current PyTorch, which Intel Macs can't install, so it runs in GitHub Actions (`.github/workflows/verify.yml`), or anywhere Linux is available:

```bash
pip install -e ".[dev,reference]"
python reference/make_reference.py      # writes results/verify.json and the fixtures
```

## Run the recall experiment

```bash
.venv/bin/python -m experiments.recall.run          # full grid → results/recall.json + site/assets/recall.svg
.venv/bin/python -m experiments.recall.run --pilot  # quick look, one run per layer type
.venv/bin/python -m experiments.recall.figure       # redraw the figure only
```

## Private files

`replicate any SOTA model.md` (the project spec), `CLAUDE.md` and `Shortcuts.md` are kept only on the author's Mac and are git-ignored.
