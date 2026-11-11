"""The 8 fixed prompts the replication is checked on. Deterministic, original text (no copyrighted material).

Each one probes a different way an implementation can go wrong.
"""

import random

_WORDS = ("river stone lantern harbour quiet signal orbit meadow copper engine winter bridge echo "
          "garden falcon compass thread summit ember tide archive glacier violin pocket").split()


def _long_text(n_sentences: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    for i in range(n_sentences):
        a, b, c = rng.sample(_WORDS, 3)
        out.append(f"Entry {i}: the {a} near the {b} kept a {c} for {rng.randint(2, 97)} days.")
    return " ".join(out)


PROMPTS: dict[str, str] = {
    "one_token": "Hello",                                                    # T=1: no history at all
    "english": "The three laws of motion describe how forces change the way objects move. The first law says",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n",
    "chinese": "机器学习是人工智能的一个分支，它研究计算机如何从数据中学习。",  # multi-byte tokens
    "emoji": "Weather today: ☀️🌧️❄️ — packing list: 🧥🧤🥾 and a 📚.",
    "repeated": "red green blue " * 40,                                     # tests long-range copying
    "long_1k": _long_text(55, seed=1),                                       # ~1k tokens
    "long_4k": _long_text(230, seed=2),                                      # ~4k tokens, crosses 64-token chunks many times
}

N_GREEDY = 32
