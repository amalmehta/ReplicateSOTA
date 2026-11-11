"""Qwen3.5 (text path), written from scratch in plain PyTorch.

The model is a stack of 24 pre-norm residual blocks. Each block mixes tokens with
one of two layers, then applies a SwiGLU MLP:

  * Gated DeltaNet ("linear_attention", 18 of 24 layers): a fixed-size memory
    matrix S per head that is decayed, corrected (delta rule) and read each step.
  * Gated softmax attention ("full_attention", every 4th layer): ordinary causal
    attention with grouped KV heads, partial RoPE and a sigmoid output gate.

Nothing here imports `transformers`. Names of parameters match the official
checkpoint so weights load one-to-one (see replicatesota/load.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Qwen35Config:
    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    layer_types: list[str] = field(default_factory=list)
    # full attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    @classmethod
    def from_hf(cls, cfg: dict) -> "Qwen35Config":
        t = cfg.get("text_config", cfg)
        rope = t.get("rope_parameters", {})
        return cls(
            vocab_size=t["vocab_size"],
            hidden_size=t["hidden_size"],
            intermediate_size=t["intermediate_size"],
            num_hidden_layers=t["num_hidden_layers"],
            layer_types=list(t["layer_types"]),
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            rope_theta=rope.get("rope_theta", 10_000_000.0),
            partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
            linear_num_key_heads=t["linear_num_key_heads"],
            linear_num_value_heads=t["linear_num_value_heads"],
            linear_key_head_dim=t["linear_key_head_dim"],
            linear_value_head_dim=t["linear_value_head_dim"],
            linear_conv_kernel_dim=t["linear_conv_kernel_dim"],
            rms_norm_eps=t["rms_norm_eps"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", t.get("tie_word_embeddings", True)),
        )


# --------------------------------------------------------------------------- norms


class RMSNorm(nn.Module):
    """Zero-centred RMSNorm: y = x / rms(x) * (1 + w). Weight starts at 0, so the layer starts as identity scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class GatedRMSNorm(nn.Module):
    """Output norm of Gated DeltaNet: y = (x / rms(x)) * w * silu(z). Ordinary (not zero-centred) weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * y.to(dtype)
        return (y * F.silu(z.float())).to(dtype)


# --------------------------------------------------------------------------- RoPE


def rope_cos_sin(positions: torch.Tensor, rot_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape [T, rot_dim] for integer positions [T].

    For text-only input Qwen3.5's multimodal RoPE (three position streams) collapses to
    ordinary RoPE, because all three streams carry the same position.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate only the first rot_dim channels of each head; the rest pass through. x: [B, H, T, D]."""
    r = cos.shape[-1]
    x_rot, x_pass = x[..., :r], x[..., r:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_pass], dim=-1)


# --------------------------------------------------------------------------- Gated DeltaNet core


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_delta_rule_recurrent(q, k, v, g, beta, state=None):
    """The definition, one token at a time. Shapes: q,k [B,H,T,Dk]; v [B,H,T,Dv]; g,beta [B,H,T].

    For each step t (per head), with memory S of shape [Dk, Dv]:
        S <- exp(g_t) * S                       forget a little (g_t <= 0)
        e  = v_t - S^T k_t                      what the memory gets wrong about k_t
        S <- S + beta_t * k_t e^T               write only the correction (delta rule)
        o_t = S^T q_t                           read
    q and k are assumed already L2-normalised; q is assumed already scaled by 1/sqrt(Dk).
    Returns (o [B,H,T,Dv], final state [B,H,Dk,Dv]).
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, dtype=torch.float32) if state is None else state.float()
    out = torch.empty(B, H, T, Dv, dtype=torch.float32)
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        kt = k[:, :, t]
        pred = torch.einsum("bhkv,bhk->bhv", S, kt)
        delta = (v[:, :, t] - pred) * beta[:, :, t, None]
        S = S + kt[..., :, None] * delta[..., None, :]
        out[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, :, t])
    return out, S


def gated_delta_rule_chunked(q, k, v, g, beta, state=None, chunk: int = 64):
    """Same result as the recurrent form, computed a chunk of C tokens at a time with matrix products.

    Inside a chunk, the C sequential delta-rule writes are equivalent to solving one small
    lower-triangular system (the "UT transform"), so the work becomes a few C x C matmuls
    instead of C dependent steps. Between chunks the memory S is carried exactly as before.
    """
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    n = (T + pad) // chunk
    q, k, v = (x.float().reshape(B, H, n, chunk, -1) for x in (q, k, v))
    g = g.float().reshape(B, H, n, chunk)
    beta = beta.float().reshape(B, H, n, chunk)

    G = g.cumsum(-1)                                     # log of total decay from chunk start to t
    causal = torch.ones(chunk, chunk, dtype=torch.bool).tril()
    D = (G[..., :, None] - G[..., None, :]).masked_fill(~causal, float("-inf")).exp()  # decay j -> i, i >= j

    kb = k * beta[..., None]
    # A[i, j] = beta_i * (k_i . k_j) * decay(j->i) for j < i; unit diagonal. Solve (I + A) X = RHS.
    A = (kb @ k.transpose(-1, -2)) * D
    A = A.masked_fill(~causal.tril(-1), 0.0) + torch.eye(chunk)
    u = torch.linalg.solve_triangular(A, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(A, kb * G.exp()[..., None], upper=False, unitriangular=True)

    attn = (q @ k.transpose(-1, -2)) * D                 # intra-chunk read, causal incl. diagonal
    q_dec = q * G.exp()[..., None]                       # query reading the state from chunk start
    k_dec = k * (G[..., -1:] - G).exp()[..., None]       # key decayed to chunk end
    chunk_decay = G[..., -1].exp()[..., None, None]

    S = torch.zeros(B, H, Dk, Dv) if state is None else state.float()
    out = torch.empty(B, H, n, chunk, Dv)
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ S              # corrections after accounting for the old memory
        out[:, :, i] = q_dec[:, :, i] @ S + attn[:, :, i] @ v_new
        S = S * chunk_decay[:, :, i] + k_dec[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, n * chunk, Dv)[:, :, :T]
    return out, S


# --------------------------------------------------------------------------- layers


class GatedDeltaNet(nn.Module):
    def __init__(self, c: Qwen35Config):
        super().__init__()
        self.nk, self.nv = c.linear_num_key_heads, c.linear_num_value_heads
        self.dk, self.dv = c.linear_key_head_dim, c.linear_value_head_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        self.kernel = c.linear_conv_kernel_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(c.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.nv, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.nv, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, self.kernel, groups=conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.nv))
        self.A_log = nn.Parameter(torch.zeros(self.nv))
        self.norm = GatedRMSNorm(self.dv, eps=c.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None, mode: str = "chunk") -> torch.Tensor:
        B, T, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)                        # [B, C, T]
        # Short causal depthwise convolution: each channel looks at itself and the previous kernel-1 tokens.
        prev = cache.get("conv") if cache else None
        if prev is None:
            prev = mixed.new_zeros(B, mixed.shape[1], self.kernel - 1)
        padded = torch.cat([prev, mixed], dim=-1)
        if cache is not None:
            cache["conv"] = padded[..., -(self.kernel - 1):].detach()
        mixed = F.silu(F.conv1d(padded, self.conv1d.weight, groups=padded.shape[1])).transpose(1, 2)

        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, self.nk, self.dk).transpose(1, 2).float()
        k = k.reshape(B, T, self.nk, self.dk).transpose(1, 2).float()
        v = v.reshape(B, T, self.nv, self.dv).transpose(1, 2).float()
        if self.nv != self.nk:
            q = q.repeat_interleave(self.nv // self.nk, dim=1)
            k = k.repeat_interleave(self.nv // self.nk, dim=1)
        q = l2norm(q) / math.sqrt(self.dk)
        k = l2norm(k)
        beta = torch.sigmoid(self.in_proj_b(x)).transpose(1, 2).float()                     # write strength in (0,1)
        g = (-self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)).transpose(1, 2)  # log decay <= 0

        state = cache.get("S") if cache else None
        rule = gated_delta_rule_recurrent if (mode == "recurrent" or T == 1) else gated_delta_rule_chunked
        o, S = rule(q, k, v, g, beta, state)
        if cache is not None:
            cache["S"] = S.detach()

        o = o.transpose(1, 2).to(x.dtype)                                    # [B, T, Hv, Dv]
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)
        o = self.norm(o, z).reshape(B, T, self.value_dim)
        return self.out_proj(o)


class GatedAttention(nn.Module):
    def __init__(self, c: Qwen35Config):
        super().__init__()
        self.h, self.kvh, self.d = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.rot_dim = int(c.head_dim * c.partial_rotary_factor)
        self.theta = c.rope_theta
        self.q_proj = nn.Linear(c.hidden_size, self.h * self.d * 2, bias=False)   # query and gate, interleaved per head
        self.k_proj = nn.Linear(c.hidden_size, self.kvh * self.d, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.kvh * self.d, bias=False)
        self.o_proj = nn.Linear(self.h * self.d, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.d, c.rms_norm_eps)
        self.k_norm = RMSNorm(self.d, c.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.h, 2 * self.d).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.kvh, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kvh, self.d).transpose(1, 2)
        cos, sin = rope_cos_sin(positions, self.rot_dim, self.theta)
        q, k = apply_partial_rope(q, cos, sin), apply_partial_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k.detach(), v.detach()
        rep = self.h // self.kvh
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        S = k.shape[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d)
        # causal mask aligned to the end: query i (absolute S-T+i) sees keys <= its position
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        o = torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v
        o = o.transpose(1, 2).reshape(B, T, self.h * self.d)
        o = o * torch.sigmoid(gate.reshape(B, T, self.h * self.d))
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, c: Qwen35Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c: Qwen35Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(c)
        elif layer_type == "full_attention":
            self.self_attn = GatedAttention(c)
        else:
            raise ValueError(f"unknown layer type {layer_type!r}")
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, cache=None, mode="chunk"):
        h = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(h, cache, mode)
        else:
            h = self.self_attn(h, positions, cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen35(nn.Module):
    def __init__(self, c: Qwen35Config):
        super().__init__()
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Block(c, t) for t in c.layer_types)
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)

    def new_cache(self) -> dict:
        """Decode state: tokens seen so far, plus per-layer memory (KV for attention; conv tail + S for DeltaNet)."""
        return {"pos": 0, "layers": [dict() for _ in self.layers]}

    def hidden(self, ids: torch.Tensor, cache: dict | None = None, mode: str = "chunk",
               collect: bool = False):
        """Final normed hidden states [B, T, D]. With collect=True also returns each block's output."""
        start = cache["pos"] if cache is not None else 0
        positions = torch.arange(start, start + ids.shape[1])
        x = self.embed_tokens(ids)
        per_layer = []
        for i, layer in enumerate(self.layers):
            x = layer(x, positions, None if cache is None else cache["layers"][i], mode)
            if collect:
                per_layer.append(x)
        if cache is not None:
            cache["pos"] = start + ids.shape[1]
        return (self.norm(x), per_layer) if collect else self.norm(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        # Tied embeddings: the output layer reuses the input embedding matrix.
        return h @ self.embed_tokens.weight.T

    def forward(self, ids: torch.Tensor, cache=None, mode: str = "chunk") -> torch.Tensor:
        return self.logits(self.hidden(ids, cache, mode))
