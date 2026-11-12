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


def recall_summary() -> dict | None:
    path = RES / "recall.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    acc: dict = {}
    for r in d["runs"]:
        acc.setdefault(r["mixer"], {}).setdefault(r["seq_len"], []).append(r["accuracy"])
    summary = {m: {n: [float(np.mean(v)), float(np.std(v)), len(v)] for n, v in by.items()} for m, by in acc.items()}
    ns = sorted(summary["gated_deltanet"])
    wins = [n for n in ns if summary["gated_deltanet"][n][0] > summary["linear_attention"][n][0]]
    # "clear" = gap larger than the two seed spreads combined; otherwise we say it is within noise.
    clear = [n for n in wins if summary["gated_deltanet"][n][0] - summary["linear_attention"][n][0]
             > summary["gated_deltanet"][n][1] + summary["linear_attention"][n][1]]
    if len(clear) == len(ns):
        verdict = f"Hypothesis supported: Gated DeltaNet beats linear attention at every length, by more than the seed spread."
    elif wins:
        verdict = (f"Partly supported: Gated DeltaNet is ahead at N = {', '.join(map(str, wins))}"
                   f"{'' if clear == wins else ' (within seed noise at N = ' + ', '.join(str(n) for n in wins if n not in clear) + ')'}"
                   f", but not at every length.")
    else:
        verdict = "Not supported: Gated DeltaNet does not beat linear attention at any length tested."
    return {
        "summary": summary, "verdict": verdict,
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
