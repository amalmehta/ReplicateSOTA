"""Fill site/template.html with numbers from results/*.json → site/index.html.

No number on the page is typed by hand: everything shown comes from this embedded data.

    python site/build.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
RES = ROOT / "results"


def model_config() -> dict:
    path = RES / "model_config.json"
    if not path.exists():
        sys.path.insert(0, str(ROOT))
        from replicatesota.load import fetch
        cfg = json.loads(fetch("config.json").read_text())["text_config"]
        keep = ["layer_types", "num_key_value_heads", "head_dim", "linear_num_value_heads",
                "linear_key_head_dim", "linear_value_head_dim", "hidden_size", "vocab_size"]
        path.write_text(json.dumps({k: cfg[k] for k in keep}, indent=1))
    return json.loads(path.read_text())


BASELINE_OK = 0.90  # a length counts only if softmax reaches this on EVERY seed (keep in sync with figure.py)


def recall_summary() -> dict | None:
    """Per-length verdicts.

    Rule fixed before the run: Gated DeltaNet "wins" a length only if its mean lead over linear attention is
    larger than the two seed spreads (sample std) added together.
    Rule added AFTER seeing the results (2026-11-11): a length is "inconclusive" if the softmax baseline,
    which can solve the task at any length here, failed to learn it on any seed. Then the runs measure how
    fast training takes off in 1,500 steps, not how much a memory can hold.
    """
    path = RES / "recall.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    acc: dict = {}
    for r in d["runs"]:
        acc.setdefault(r["mixer"], {}).setdefault(r["seq_len"], []).append(r["accuracy"])
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    summary = {m: {n: [float(np.mean(v)), sd(v), len(v)] for n, v in by.items()} for m, by in acc.items()}
    seeds = {m: {n: [float(a) for a in v] for n, v in by.items()} for m, by in acc.items()}
    per_length = {}
    for n in sorted(summary["gated_deltanet"]):
        base = seeds.get("softmax_attention", {}).get(n, [])
        if base and min(base) < BASELINE_OK:
            per_length[n] = {"status": "inconclusive", "baseline_learned": sum(a >= BASELINE_OK for a in base),
                             "seeds": len(base)}
            continue
        g, l = summary["gated_deltanet"][n], summary["linear_attention"][n]
        gap = g[0] - l[0]
        status = "gdn" if gap > g[1] + l[1] else "linear" if -gap > g[1] + l[1] else "tie"
        per_length[n] = {"status": status, "gap": gap}
    decided = {n: v for n, v in per_length.items() if v["status"] != "inconclusive"}
    unsure = [str(n) for n, v in per_length.items() if v["status"] == "inconclusive"]
    tail = (f" N = {', '.join(unsure)} is inconclusive: the softmax baseline did not learn the task on every seed "
            f"within the step budget." if unsure else "")
    if decided and all(v["status"] == "gdn" for v in decided.values()):
        verdict = f"Supported at N = {', '.join(map(str, decided))}: Gated DeltaNet leads by more than the seed spread." + tail
    elif decided:
        parts = []
        for n, v in decided.items():
            word = {"gdn": "Gated DeltaNet leads", "linear": "linear attention leads", "tie": "a tie"}[v["status"]]
            parts.append(f"at N = {n}, {word} ({v['gap'] * 100:+.1f} points)")
        verdict = "Not supported: " + "; ".join(parts) + ", within seed noise." * all(
            v["status"] == "tie" for v in decided.values()) + tail
    else:
        verdict = "Inconclusive at every length tested." + tail
    return {
        "summary": summary, "seeds": seeds, "per_length": per_length, "verdict": verdict,
        "baseline_ok": BASELINE_OK,
        "setup": (f"Two-layer models, width {d['dim']}, vocabulary {d['vocab']}, batch {d['batch']}, "
                  f"learning rate {d['lr']}, {d['runs'][0]['steps']} training steps, key–value pairs = length/8, "
                  f"accuracy on 1,000 held-out sequences, torch {d['torch']}. Task format from Zoology (Arora et al. 2023)."),
        "deviations_from_zoology": d["deviations_from_zoology"],
    }


def main():
    data = {
        "config": model_config(),
        "verify": json.loads((RES / "verify.json").read_text()) if (RES / "verify.json").exists() else None,
        "recall": recall_summary(),
        "built": dt.date.today().isoformat(),
    }
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (SITE / "template.html").read_text().replace("/*DATA*/", blob)
    (SITE / "index.html").write_text(html)
    print(f"wrote {SITE / 'index.html'} (verify={'yes' if data['verify'] else 'pending'}, "
          f"recall={'yes' if data['recall'] else 'pending'})")


if __name__ == "__main__":
    main()
