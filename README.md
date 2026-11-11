# Replicate SOTA

> **Work in progress.** v1 is being built. Numbers below appear once the verification run finishes.

Rebuild a current open-weights model from scratch, prove the rebuild matches the official one at every position, test its key new idea in a small experiment, and explain it all.

**v1:** [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache 2.0, Feb 2026). It is a hybrid model: 18 Gated DeltaNet layers (linear attention with a fixed-size memory) and 6 ordinary attention layers, in a repeating 3 : 1 pattern.

- [How to set up and run](docs/INSTRUCTIONS.md)
- [System design](docs/SYSTEM-DESIGN.md)
- [What's where](docs/FILE-STRUCTURE.md)
- [Prompt log](docs/PROMPTS.md)
