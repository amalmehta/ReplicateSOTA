"""Associative recall (MQAR): can a layer store many key->value pairs and fetch them on request?

Question:   Does the gated delta rule (Qwen3.5's linear layer) recall better than plain linear attention?
Hypothesis: Gated DeltaNet accuracy > linear attention accuracy at every sequence length tested.
Baseline:   softmax attention (the upper bound: it can look back at every token exactly).
Metric:     accuracy of the value token predicted at each query, on 1,000 held-out sequences.

Task format follows Zoology (Arora et al. 2023, arXiv 2312.04927, Alg. 1 + App. E.2):
  D key-value pairs at the start, then each key appears again later (power-law distance, alpha=0.1)
  and the model must output its value. Two layers, each = sequence mixer + MLP with norms. AdamW.
Deviations from Zoology, chosen to fit a CPU budget (labelled in results and on the site):
  vocab 512 (not 8,192); model dim 64; one learning rate (3e-3, chosen up front, not swept);
  fresh random sequences each step instead of a fixed 100k set; 1,500 steps; lengths 64 and 128 only.
Change made after the pilot (2026-11-11, user-approved): both linear mixers get Qwen3.5's short causal
  convolution (kernel 4, SiLU) on q/k/v. Without it, neither could see the previous token and both stalled
  near chance while softmax (which has position embeddings) reached 100%.

    python -m experiments.recall.run            # full grid: 3 mixers x 2 lengths x 3 seeds
    python -m experiments.recall.run --pilot    # one quick run per mixer at N=64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from replicatesota.models.qwen35 import gated_delta_rule_chunked, l2norm

ROOT = Path(__file__).resolve().parents[2]
VOCAB = 512
DIM = 64
LR = 3e-3
BATCH = 64
LENGTHS = (64, 128)
SEEDS = (0, 1, 2)
MIXERS = ("gated_deltanet", "linear_attention", "softmax_attention")
IGNORE = -100


# ------------------------------------------------------------------ data


def mqar_batch(n: int, seq_len: int, rng: np.random.Generator, alpha: float = 0.1):
    """Zoology-style MQAR. Keys from the first half of the vocab, values from the second half.
    Returns inputs [n, T] and targets [n, T] (IGNORE except right after each repeated key)."""
    D = seq_len // 8                                  # number of key-value pairs (N/8, as in Zoology Exp. 2)
    half = VOCAB // 2
    x = np.zeros((n, seq_len), dtype=np.int64)        # 0 = filler token
    y = np.full((n, seq_len), IGNORE, dtype=np.int64)
    # positions for the second occurrence of keys: power-law over gaps, in [2D, T-1)
    gaps = np.arange(1, seq_len - 2 * D)
    p = alpha * gaps.astype(float) ** (alpha - 1)
    p /= p.sum()
    for b in range(n):
        keys = rng.choice(np.arange(1, half), D, replace=False)
        vals = rng.choice(np.arange(half, VOCAB), D)
        x[b, 0:2 * D:2], x[b, 1:2 * D:2] = keys, vals
        pos = 2 * D + rng.choice(gaps - 1, D, replace=False, p=p)   # distinct query slots after the pairs
        pos = np.minimum(pos, seq_len - 2)
        for kk, vv, pp in zip(keys, vals, pos):
            x[b, pp] = kk
            y[b, pp] = vv                              # predict the value at the key's second occurrence
    return torch.from_numpy(x), torch.from_numpy(y)


# ------------------------------------------------------------------ mixers


class Mixer(nn.Module):
    def __init__(self, kind: str, dim: int, max_len: int):
        super().__init__()
        self.kind = kind
        self.heads = 1 if kind == "softmax_attention" else 2    # Zoology: 1-head attention; Yang: 2-head DeltaNet
        self.hd = dim // self.heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        if kind != "softmax_attention":
            # Qwen3.5's short causal depthwise convolution over q/k/v (kernel 4), applied to both linear mixers.
            self.conv = nn.Conv1d(3 * dim, 3 * dim, 4, groups=3 * dim, bias=False)
        if kind == "gated_deltanet":
            self.b = nn.Linear(dim, self.heads)               # write strength beta = sigmoid(.)
            self.a = nn.Linear(dim, self.heads)               # forget gate: g = -exp(A) * softplus(.)
            # Initialise the forget gate near "remember" (Mamba-2 / Qwen3.5 recipe): A ~ U(1, 16),
            # dt ~ logU(1e-3, 1e-1) stored as softplus^-1(dt) in the bias, so decay exp(-A*dt) starts close to 1.
            # (Pilot finding: with zero init the memory halves every token and recall stays at chance.)
            self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
            dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1)))
            with torch.no_grad():
                self.a.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        if kind == "softmax_attention":
            self.pos = nn.Embedding(max_len, dim)             # Zoology gives attention learned positions

    def forward(self, x):
        B, T, D = x.shape
        if self.kind == "softmax_attention":
            x = x + self.pos(torch.arange(T))
        h = self.qkv(x)
        if self.kind != "softmax_attention":
            h = F.silu(self.conv(F.pad(h.transpose(1, 2), (3, 0)))).transpose(1, 2)
        q, k, v = h.view(B, T, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        if self.kind == "softmax_attention":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kind == "linear_attention":
            # S_t = S_{t-1} + k_t v_t^T ; o_t = S_t^T q_t  (no forgetting, no correction). Parallel causal form.
            q, k = l2norm(q) / math.sqrt(self.hd), l2norm(k)
            att = (q @ k.transpose(-1, -2)).tril()
            o = att @ v
        else:
            q, k = l2norm(q) / math.sqrt(self.hd), l2norm(k)
            beta = torch.sigmoid(self.b(x)).transpose(1, 2)
            g = (-self.A_log.exp() * F.softplus(self.a(x))).transpose(1, 2)
            o, _ = gated_delta_rule_chunked(q, k, v, g, beta, chunk=64)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class TinyModel(nn.Module):
    """Two layers, each: norm -> mixer -> residual, norm -> MLP -> residual."""

    def __init__(self, kind: str, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.mix = nn.ModuleList(Mixer(kind, DIM, max_len) for _ in range(2))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
                                 for _ in range(2))
        self.n1 = nn.ModuleList(nn.LayerNorm(DIM) for _ in range(2))
        self.n2 = nn.ModuleList(nn.LayerNorm(DIM) for _ in range(2))
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, ids):
        x = self.emb(ids)
        for m, f, a, b in zip(self.mix, self.mlp, self.n1, self.n2):
            x = x + m(a(x))
            x = x + f(b(x))
        return self.head(x)


# ------------------------------------------------------------------ train / eval


def run_one(kind: str, seq_len: int, seed: int, steps: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    model = TinyModel(kind, seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    warm = steps // 10
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    t0 = time.time()
    curve = []
    for step in range(steps):
        x, y = mqar_batch(BATCH, seq_len, rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 20) == 0:
            curve.append((step, round(float(loss), 4)))
    # held-out evaluation: a separate RNG stream that training never touched
    model.eval()
    erng = np.random.default_rng(999_999)
    correct = total = 0
    with torch.no_grad():
        for _ in range(1000 // 100):
            x, y = mqar_batch(100, seq_len, erng)
            pred = model(x).argmax(-1)
            m = y != IGNORE
            correct += int((pred[m] == y[m]).sum())
            total += int(m.sum())
    return {"mixer": kind, "seq_len": seq_len, "pairs": seq_len // 8, "seed": seed, "steps": steps,
            "params": sum(p.numel() for p in model.parameters()), "accuracy": correct / total,
            "train_seconds": round(time.time() - t0, 1), "loss_curve": curve}


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--mixer", choices=MIXERS)
    ap.add_argument("--merge", action="store_true", help="combine recall_part_*.json into recall.json + figure")
    ap.add_argument("--tag", default="")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    BATCH = args.batch
    torch.set_num_threads(args.threads)
    if args.merge:
        return merge()
    name = ("recall_pilot" if args.pilot else (f"recall_part_{args.mixer}" if args.mixer else "recall")) + (f"_{args.tag}" if args.tag else "")
    out_path = ROOT / "results" / f"{name}.json"
    mixers = (args.mixer,) if args.mixer else MIXERS
    grid = [(m, 64, 0) for m in mixers] if args.pilot else [(m, n, s) for n in LENGTHS for s in SEEDS for m in mixers]
    runs = []
    for kind, n, s in grid:
        r = run_one(kind, n, s, args.steps)
        runs.append(r)
        print(f"{kind:18s} N={n:3d} seed={s} acc={r['accuracy']:.3f} ({r['train_seconds']}s)", flush=True)
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps({
            "task": "MQAR (Zoology-style)", "vocab": VOCAB, "dim": DIM, "lr": LR, "batch": BATCH,
            "deviations_from_zoology": [
                "Vocabulary 512 instead of 8,192; model width 64.",
                "One learning rate (3e-3), chosen up front rather than swept.",
                "Fresh random sequences every step instead of a fixed 100k training set.",
                f"{args.steps} training steps (CPU budget); sequence lengths 64 and 128 only (256 deferred).",
                "Changed after a pilot: both linear mixers get Qwen3.5's 4-token causal convolution on q/k/v. "
                "Without it neither could see the previous token, and both stalled near chance while softmax reached 100%."],
            "torch": torch.__version__, "runs": runs}, indent=1))
    if not args.pilot and not args.mixer:
        from experiments.recall.figure import main as fig
        fig()


def merge():
    parts = sorted((ROOT / "results").glob("recall_part_*.json"))
    docs = [json.loads(p.read_text()) for p in parts]
    out = {k: v for k, v in docs[0].items() if k != "runs"}
    out["runs"] = [r for d in docs for r in d["runs"]]
    (ROOT / "results" / "recall.json").write_text(json.dumps(out, indent=1))
    print(f"merged {len(out['runs'])} runs from {len(parts)} files")
    from experiments.recall.figure import main as fig
    fig()


if __name__ == "__main__":
    main()
