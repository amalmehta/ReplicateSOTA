"""Our model vs the official one, on this machine, using the compact fixtures CI produced.

Skips (does not fail) if fixtures have not been generated yet. Run `pytest -m slow` style selection
is not needed: these load the real weights (~1.7 GB download once, ~4 GB RAM in fp32).
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from reference.prompts import N_GREEDY, PROMPTS

FX = Path(__file__).parent / "fixtures" / "qwen35"
MAX_ABS = 1e-4  # tightened from 1e-3 after the first run measured a worst gap of 5.3e-5

pytestmark = pytest.mark.skipif(not (FX / "meta.json").exists(), reason="fixtures not generated yet (run CI)")


@pytest.fixture(scope="module")
def model():
    from replicatesota.load import load_model
    return load_model()


@pytest.mark.parametrize("name", list(PROMPTS))
@torch.no_grad()
def test_matches_official(model, name):
    from replicatesota.generate import greedy

    fx = load_file(str(FX / f"{name}.safetensors"))
    ids = fx["input_ids"][None]
    logits = model(ids)[0]
    # 1) every position: our logits at the reference's top-20 tokens, and the same top-1 token
    ours_at_top = logits.gather(-1, fx["top_indices"])
    assert (ours_at_top - fx["top_values"]).abs().max() <= MAX_ABS
    assert torch.equal(logits.argmax(-1), fx["top_indices"][:, 0])
    # 2) the full 248,320-way row at the last position
    assert (logits[-1] - fx["last_logits"]).abs().max() <= MAX_ABS
    # 3) a 32-token greedy continuation, token for token
    assert greedy(model, fx["input_ids"].tolist(), N_GREEDY) == fx["greedy"].tolist()
