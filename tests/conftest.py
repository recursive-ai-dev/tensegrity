import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
torch.set_num_threads(min(2, os.cpu_count() or 1))

from tensegrity import TensegrityConfig, TensegrityLM, prepare_data  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_WORDS = ("the a one little big red old happy sad dog cat bird girl boy tree house river sun moon "
          "ran jumped saw found loved made sang slept under over near beside quickly slowly then "
          "and but because day night morning garden story friend").split()


def make_text(n_docs: int = 400, seed: int = 0, sep: str = "<|endoftext|>") -> str:
    rnd = random.Random(seed)
    docs = []
    for _ in range(n_docs):
        sents = []
        for _ in range(rnd.randint(2, 6)):
            sents.append(" ".join(rnd.choice(_WORDS) for _ in range(rnd.randint(4, 9))).capitalize() + ".")
        docs.append(" ".join(sents))
    return sep.join(docs) + sep


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    src = d / "src.txt"
    src.write_text(make_text(), encoding="utf-8")
    out = d / "prepared"
    prepare_data(str(src), str(out), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
    return str(out)


@pytest.fixture(scope="session")
def corpus_dir2(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus2")
    src = d / "src.txt"
    src.write_text(make_text(300, seed=5).replace("the", "THE"), encoding="utf-8")
    out = d / "prepared"
    prepare_data(str(src), str(out), val_fraction=0.1, doc_sep="<|endoftext|>", verbose=False)
    return str(out)


def small_cfg(**over) -> TensegrityConfig:
    kw = dict(n_layer=4, n_embd=48, n_head=3, n_kv_head=1, block_size=48, sliding_window=12, global_every=3,
              ffn_mult=2.0, ffn_align=8, layerdrop_max=0.0, grad_checkpoint=False)
    kw.update(over)
    return TensegrityConfig(**kw)


def randomized(model: TensegrityLM, std: float = 0.05, seed: int = 0) -> TensegrityLM:
    """Perturb every parameter so tests never run on a lucky structured-zero corner."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * std)
    return model


@pytest.fixture
def cfg_factory():
    return small_cfg
