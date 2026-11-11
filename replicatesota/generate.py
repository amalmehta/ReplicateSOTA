"""Greedy generation with a cache, plus a tiny command line.

    python -m replicatesota.generate "The capital of France is"
"""

from __future__ import annotations

import sys
import time

import torch

from .models.qwen35 import Qwen35


@torch.no_grad()
def greedy(model: Qwen35, ids: list[int], n_new: int, eos: int | None = None) -> list[int]:
    """Prefill the prompt once, then feed one token at a time. Attention layers grow a KV cache;
    DeltaNet layers carry a fixed-size state, so their per-token cost does not grow with length."""
    cache = model.new_cache()
    logits = model(torch.tensor([ids]), cache)[:, -1]
    out = []
    for _ in range(n_new):
        nxt = int(logits.argmax(-1))
        out.append(nxt)
        if eos is not None and nxt == eos:
            break
        logits = model(torch.tensor([[nxt]]), cache)[:, -1]
    return out


def main(argv: list[str]) -> None:
    from .load import load_model, load_tokenizer

    prompt = " ".join(argv) or "The three laws of motion are"
    tok, model = load_tokenizer(), load_model()
    ids = tok.encode(prompt).ids
    t0 = time.time()
    out = greedy(model, ids, 40)
    dt = time.time() - t0
    print(prompt + tok.decode(out))
    print(f"\n[{len(out)} tokens in {dt:.1f}s, {len(out) / dt:.2f} tok/s incl. prefill]", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
