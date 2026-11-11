"""Download Qwen3.5 weights from Hugging Face and load them into our from-scratch modules.

Only safetensors are read (no pickle, no remote code). Vision-encoder and multi-token-prediction
weights in the checkpoint are skipped: v1 replicates the text path.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .models.qwen35 import Qwen35, Qwen35Config

REPO_ID = "Qwen/Qwen3.5-0.8B"
# Pin the exact files we verified against; update deliberately, never silently.
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TEXT_PREFIX = "model.language_model."


def fetch(name: str, repo_id: str = REPO_ID, revision: str = REVISION) -> Path:
    return Path(hf_hub_download(repo_id, name, revision=revision))


def load_tokenizer(repo_id: str = REPO_ID, revision: str = REVISION) -> Tokenizer:
    return Tokenizer.from_file(str(fetch("tokenizer.json", repo_id, revision)))


def load_model(repo_id: str = REPO_ID, revision: str = REVISION, dtype=torch.float32) -> Qwen35:
    cfg = json.loads(fetch("config.json", repo_id, revision).read_text())
    index = json.loads(fetch("model.safetensors.index.json", repo_id, revision).read_text())
    files = sorted(set(index["weight_map"].values()))

    model = Qwen35(Qwen35Config.from_hf(cfg))
    expected = set(model.state_dict())
    state = {}
    for f in files:
        for name, t in load_file(str(fetch(f, repo_id, revision))).items():
            if name.startswith(TEXT_PREFIX):
                state[name[len(TEXT_PREFIX):]] = t.to(dtype)
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}")
    model.load_state_dict(state, strict=True)
    return model.to(dtype).eval()
