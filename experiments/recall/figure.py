"""results/recall.json -> site/assets/recall.svg: every seed as a dot, the mean as a short bar.

Lengths where the softmax baseline did not learn the task on every seed are shaded and labelled
"inconclusive" (rule in site/build.py: BASELINE_OK).

    python -m experiments.recall.figure
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"gated_deltanet": "Gated DeltaNet", "linear_attention": "Linear attention",
          "softmax_attention": "Softmax attention (upper baseline)"}
# One muted accent for the layer under test, warm terracotta for the plain baseline, grey for the reference.
COLORS = {"gated_deltanet": "#3d6a8e", "linear_attention": "#b5643c", "softmax_attention": "#8a8580"}
BASELINE_OK = 0.90  # keep in sync with site/build.py


def by_cell(runs):
    out: dict = {}
    for r in runs:
        out.setdefault(r["mixer"], {}).setdefault(r["seq_len"], []).append(r["accuracy"])
    return out


def main(src: Path = ROOT / "results" / "recall.json", dst: Path = ROOT / "site" / "assets" / "recall.svg"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = by_cell(json.loads(src.read_text())["runs"])
    ns = sorted({n for d in cells.values() for n in d})
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=100)
    for i, n in enumerate(ns):
        if min(cells.get("softmax_attention", {}).get(n, [1.0])) < BASELINE_OK:
            ax.axvspan(i - 0.45, i + 0.45, color="#8a8580", alpha=0.10, lw=0)
            ax.text(i, 104, "inconclusive: baseline did not learn on every seed", ha="center", fontsize=8, color="#6b665e")
    for j, m in enumerate(LABELS):
        if m not in cells:
            continue
        off = (j - 1) * 0.22
        for i, n in enumerate(ns):
            v = [a * 100 for a in cells[m].get(n, [])]
            if not v:
                continue
            xs = [i + off + (k - (len(v) - 1) / 2) * 0.035 for k in range(len(v))]
            ax.scatter(xs, v, s=34, color=COLORS[m], zorder=3, label=LABELS[m] if i == 0 else None,
                       edgecolor="white", linewidth=0.6)
            mu = sum(v) / len(v)
            ax.hlines(mu, i + off - 0.08, i + off + 0.08, color=COLORS[m], lw=2, zorder=2)
    ax.set_xticks(range(len(ns)), [f"N = {n}" for n in ns])
    ax.set_xlim(-0.6, len(ns) - 0.4)
    ax.set_xlabel("Sequence length (key–value pairs = length / 8)")
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    seeds = max(len(v) for d in cells.values() for v in d.values())
    ax.set_title(f"Each dot is one seed ({seeds} per cell); bars are means", fontsize=10, loc="left", color="#555")
    fig.tight_layout()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, transparent=True)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
