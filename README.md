# Replicate SOTA

> **Work in progress.** The rebuild is verified. The recall experiment is running.

Rebuild a current open-weights model from scratch, prove the rebuild matches the official one at every position, test its key new idea in a small experiment, and explain it all.

**v1:** [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache 2.0, Feb 2026). It is a hybrid model: 18 Gated DeltaNet layers (linear attention with a fixed-size memory) and 6 ordinary attention layers, in a repeating 3 : 1 pattern.

## Result so far

**The from-scratch rebuild matches the official model on all 8 test prompts.** The largest difference in any next-token score, at any position, is **5.3×10⁻⁵**, against a threshold of 10⁻³ fixed before running. The top token agrees at 100% of positions, and every 32-token greedy continuation is identical.

| Prompt | Tokens | Largest score gap | Same top token | Greedy 32 |
|---|---:|---:|---:|---|
| one token | 1 | 1.5×10⁻⁵ | 100% | same |
| english | 18 | 1.7×10⁻⁵ | 100% | same |
| code | 27 | 1.8×10⁻⁵ | 100% | same |
| chinese | 14 | 2.1×10⁻⁵ | 100% | same |
| emoji | 32 | 1.6×10⁻⁵ | 100% | same |
| repeated | 121 | 2.1×10⁻⁵ | 100% | same |
| long 1k | 1,042 | 3.8×10⁻⁵ | 100% | same |
| long 4k | 4,534 | 5.3×10⁻⁵ | 100% | same |

The reference is official `transformers` 5.22.2 on PyTorch 2.7.1. Ours runs on PyTorch 2.2.2, so the match holds across versions. Source: [`results/verify.json`](results/verify.json).

- [How to set up and run](docs/INSTRUCTIONS.md)
- [System design](docs/SYSTEM-DESIGN.md)
- [What's where](docs/FILE-STRUCTURE.md)
- [Prompt log](docs/PROMPTS.md)
