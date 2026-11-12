"""results/recall.json -> site/assets/recall.svg (mean ± std over seeds, per mixer and sequence length).

    python -m experiments.recall.figure
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"gated_deltanet": "Gated DeltaNet", "linear_attention": "Linear attention",
          "softmax_attention": "Softmax attention (upper baseline)"}
# One muted accent for the layer under test, warm terracotta for the plain baseline, grey for the reference.
COLORS = {"gated_deltanet": "#3d6a8e", "linear_attention": "#b5643c", "softmax_attention": "#8a8580"}


def summarize(runs):
    out = {}
    for r in runs:
        out.setdefault(r["mixer"], {}).setdefault(r["seq_len"], []).append(r["accuracy"])
    return {m: {n: (float(np.mean(v)), float(np.std(v)), len(v)) for n, v in d.items()} for m, d in out.items()}


def main(src: Path = ROOT / "results" / "recall.json", dst: Path = ROOT / "site" / "assets" / "recall.svg"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(src.read_text())
    s = summarize(data["runs"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=100)
    for m in LABELS:
        if m not in s:
            continue
        ns = sorted(s[m])
        mu = [s[m][n][0] * 100 for n in ns]
        sd = [s[m][n][1] * 100 for n in ns]
        ax.errorbar(ns, mu, yerr=sd, label=LABELS[m], color=COLORS[m], marker="o", capsize=3, lw=2,
                    ls="--" if m == "softmax_attention" else "-")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({n for d in s.values() for n in d}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Sequence length (key–value pairs = length / 8)")
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_ylim(0, 102)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    seeds = max(v[2] for d in s.values() for v in d.values())
    ax.set_title(f"Mean ± std over {seeds} seeds", fontsize=10, loc="left", color="#555")
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, transparent=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
