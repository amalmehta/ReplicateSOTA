# What's Where

```
replicatesota/
  models/qwen35.py      the model, from scratch: norms, RoPE, Gated DeltaNet, gated attention, MLP, stack
  load.py               download pinned weights + tokenizer, map checkpoint names onto our modules
  generate.py           greedy decoding with cache; `python -m replicatesota.generate "prompt"`
reference/
  prompts.py            the 8 fixed test prompts and why each exists
  make_reference.py     official transformers vs ours at every position (runs in CI)
tests/
  test_units.py         fast checks on random inputs, no weights needed
  test_logit_match.py   our logits vs the reference fixtures
  fixtures/qwen35/      compact reference outputs (top-20 per position, last row, greedy tokens)
experiments/recall/
  run.py                MQAR associative-recall experiment (Gated DeltaNet vs linear vs softmax)
  figure.py             results/recall.json → site/assets/recall.svg (one dot per seed; inconclusive lengths shaded)
results/                every number the README and site show, as JSON
site/                   static explainer site (GitHub Pages)
docs/                   instructions, system design, this file, prompt log
.github/workflows/      CI: unit tests + official-vs-ours comparison
```
