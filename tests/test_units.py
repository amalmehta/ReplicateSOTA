"""Fast tests on random inputs: no weights needed."""

import torch

from replicatesota.models.qwen35 import (
    Qwen35,
    Qwen35Config,
    apply_partial_rope,
    gated_delta_rule_chunked,
    gated_delta_rule_recurrent,
    l2norm,
    rope_cos_sin,
)


def _gdn_inputs(B=2, H=3, T=150, Dk=16, Dv=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(B, H, T, Dk, generator=g)) / Dk**0.5
    k = l2norm(torch.randn(B, H, T, Dk, generator=g))
    v = torch.randn(B, H, T, Dv, generator=g)
    gate = -torch.rand(B, H, T, generator=g) * 0.5        # log decay in (-0.5, 0]
    beta = torch.rand(B, H, T, generator=g)
    return q, k, v, gate, beta


def test_chunked_equals_recurrent():
    """The fast chunked form must compute exactly the same thing as the one-step-at-a-time definition."""
    q, k, v, g, b = _gdn_inputs()
    o1, s1 = gated_delta_rule_recurrent(q, k, v, g, b)
    for chunk in (16, 64):                                 # T=150 is not a multiple: exercises padding
        o2, s2 = gated_delta_rule_chunked(q, k, v, g, b, chunk=chunk)
        assert torch.allclose(o1, o2, atol=1e-5), (o1 - o2).abs().max()
        assert torch.allclose(s1, s2, atol=1e-5)


def test_chunked_continues_from_state():
    """Splitting a sequence in two and carrying the memory gives the same result as one pass."""
    q, k, v, g, b = _gdn_inputs(T=100)
    full, _ = gated_delta_rule_chunked(q, k, v, g, b)
    a, s = gated_delta_rule_chunked(q[:, :, :37], k[:, :, :37], v[:, :, :37], g[:, :, :37], b[:, :, :37])
    c, _ = gated_delta_rule_recurrent(q[:, :, 37:], k[:, :, 37:], v[:, :, 37:], g[:, :, 37:], b[:, :, 37:], s)
    assert torch.allclose(full, torch.cat([a, c], dim=2), atol=1e-5)


def test_delta_rule_recalls_a_stored_pair():
    """With beta=1, no decay and orthogonal keys, writing (k, v) then reading with q=k returns v exactly."""
    Dk, Dv = 8, 4
    keys = torch.eye(Dk)[:3]
    vals = torch.randn(3, Dv)
    k = keys[None, None]
    v = vals[None, None]
    q = keys[None, None]
    out, _ = gated_delta_rule_recurrent(q, k, v, torch.zeros(1, 1, 3), torch.ones(1, 1, 3))
    assert torch.allclose(out[0, 0], vals, atol=1e-6)


def test_rope_rotates_only_first_quarter_and_preserves_norm():
    x = torch.randn(1, 2, 5, 256)
    cos, sin = rope_cos_sin(torch.arange(5), 64, 1e7)
    y = apply_partial_rope(x, cos, sin)
    assert torch.equal(y[..., 64:], x[..., 64:])
    assert torch.allclose(y[..., :64].norm(dim=-1), x[..., :64].norm(dim=-1), atol=1e-5)


def _tiny_model():
    torch.manual_seed(0)
    c = Qwen35Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                     layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                     linear_key_head_dim=8, linear_value_head_dim=8)
    m = Qwen35(c)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m.eval()


@torch.no_grad()
def test_cached_decode_matches_full_forward():
    """Feeding tokens one by one through the cache must match one full forward pass."""
    m = _tiny_model()
    ids = torch.randint(0, 50, (1, 23))
    full = m(ids)
    cache = m.new_cache()
    steps = [m(ids[:, :10], cache)] + [m(ids[:, t:t + 1], cache) for t in range(10, 23)]
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


@torch.no_grad()
def test_causality():
    """Changing a future token must not change earlier outputs."""
    m = _tiny_model()
    ids = torch.randint(0, 50, (1, 12))
    ids2 = ids.clone()
    ids2[0, 8] = (ids[0, 8] + 1) % 50
    assert torch.allclose(m(ids)[:, :8], m(ids2)[:, :8], atol=1e-6)
