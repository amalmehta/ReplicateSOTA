"""Run the OFFICIAL Qwen3.5 implementation (transformers) and our from-scratch one side by side.

Runs in CI on Linux (see .github/workflows/verify.yml), because current transformers needs a newer
PyTorch than Intel Macs can install. It writes:

  * results/verify.json             full comparison at every position, per prompt and per layer
  * tests/fixtures/qwen35/*.safetensors   compact reference outputs for the local test:
        top-20 logits per position, the full logit row at the last position, the greedy continuation

Exit code 1 if any done criterion fails (see the spec's DEFINITIONS AND DEFERRED).

    python reference/make_reference.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference.prompts import N_GREEDY, PROMPTS  # noqa: E402
from replicatesota.generate import greedy  # noqa: E402
from replicatesota.load import REPO_ID, REVISION, load_model, load_tokenizer  # noqa: E402

MAX_ABS = 1e-4      # largest allowed |our logit - reference logit| (tightened from 1e-3 on 2026-11-11 after measuring 5.3e-5)
TOPK = 20
POS_CHUNK = 256     # compare logits this many positions at a time (a 4k x 248k matrix is ~4 GB)


@torch.no_grad()
def ref_greedy(ref, ids: list[int], n: int) -> list[int]:
    """Greedy decode with the official model and its own cache, never materialising full-length logits."""
    out = ref(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
    past, toks = out.past_key_values, []
    for _ in range(n):
        nxt = int(out.logits[0, -1].argmax())
        toks.append(nxt)
        out = ref(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
    return toks


@torch.no_grad()
def main() -> int:
    import transformers
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    tok = load_tokenizer()
    ref = AutoModelForImageTextToText.from_pretrained(REPO_ID, revision=REVISION, dtype=torch.float32).eval()
    text_model = ref.model.language_model
    lm_head = ref.lm_head.weight
    ours = load_model()

    # Capture each official decoder layer's output with hooks, so per-layer alignment is exact.
    captured: list[torch.Tensor] = []
    hooks = [layer.register_forward_hook(lambda m, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
             for layer in text_model.layers]

    fx_dir = ROOT / "tests" / "fixtures" / "qwen35"
    fx_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": REPO_ID, "revision": REVISION, "dtype": "float32",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version(), "machine": platform.machine()},
        "thresholds": {"max_abs_logit_diff": MAX_ABS, "top1_agreement": 1.0, "greedy_tokens": N_GREEDY},
        "prompts": {},
    }
    ok = True
    for name, text in PROMPTS.items():
        t0 = time.time()
        ids = tok.encode(text).ids
        x = torch.tensor([ids])

        captured.clear()
        h_ref = text_model(input_ids=x).last_hidden_state[0]
        ref_layers = [c[0] for c in captured]
        h_ours, our_layers = ours.hidden(x, collect=True)
        h_ours, our_layers = h_ours[0], [t[0] for t in our_layers]

        max_abs, agree, topv, topi = 0.0, 0, [], []
        for s in range(0, len(ids), POS_CHUNK):
            lr = h_ref[s:s + POS_CHUNK] @ lm_head.T
            lo = ours.logits(h_ours[s:s + POS_CHUNK])
            max_abs = max(max_abs, float((lr - lo).abs().max()))
            agree += int((lr.argmax(-1) == lo.argmax(-1)).sum())
            v, i = lr.topk(TOPK, dim=-1)
            topv.append(v), topi.append(i)
        last_row = h_ref[-1] @ lm_head.T

        g_ref = ref_greedy(ref, ids, N_GREEDY)
        g_ours = greedy(ours, ids, N_GREEDY)

        r = {
            "n_tokens": len(ids),
            "max_abs_logit_diff": max_abs,
            "top1_agreement": agree / len(ids),
            "greedy_equal": g_ref == g_ours,
            "greedy_first_mismatch": next((j for j, (a, b) in enumerate(zip(g_ref, g_ours)) if a != b), None),
            "final_hidden_max_abs_diff": float((h_ref - h_ours).abs().max()),
            "per_layer_max_abs_diff": [float((a - b).abs().max()) for a, b in zip(ref_layers, our_layers)],
            "reference_continuation": tok.decode(g_ref),
            "seconds": round(time.time() - t0, 1),
        }
        r["pass"] = r["max_abs_logit_diff"] <= MAX_ABS and r["top1_agreement"] == 1.0 and r["greedy_equal"]
        ok &= r["pass"]
        report["prompts"][name] = r
        print(f"{name:10s} T={len(ids):5d} max|dlogit|={max_abs:.2e} top1={r['top1_agreement']:.4f} "
              f"greedy={'same' if r['greedy_equal'] else 'DIFF'} {'PASS' if r['pass'] else 'FAIL'} ({r['seconds']}s)",
              flush=True)

        save_file({
            "input_ids": torch.tensor(ids, dtype=torch.int64),
            "top_values": torch.cat(topv).contiguous(),
            "top_indices": torch.cat(topi).to(torch.int64).contiguous(),
            "last_logits": last_row.contiguous(),
            "greedy": torch.tensor(g_ref, dtype=torch.int64),
        }, str(fx_dir / f"{name}.safetensors"))

    for h in hooks:
        h.remove()
    report["all_pass"] = ok
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "verify.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (fx_dir / "meta.json").write_text(json.dumps({k: report[k] for k in ("model", "revision", "dtype", "versions")}, indent=2))
    print("ALL PASS" if ok else "SOME PROMPTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
