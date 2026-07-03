"""Probe head: forward shapes, masking, save/load round-trip, bundled weights."""
import os

import numpy as np
import torch

from rewriter.routing_probe import (
    D_MODEL,
    DOMAINS,
    AttnPoolProbe,
    iter_packed_batches,
    load_probe,
    save_probe,
)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "rewriter", "routing_probe", "weights")


def test_forward_shapes():
    probe = AttnPoolProbe(d_model=64)
    x = torch.randn(3, 7, 64)
    mask = torch.ones(3, 7, dtype=torch.bool)
    out = probe(x, mask)
    assert out.shape == (3, 2)


def test_forward_respects_mask():
    """Masked (padding) tokens must not change the pooled output."""
    torch.manual_seed(0)
    probe = AttnPoolProbe(d_model=16).eval()
    x = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, True, False]])
    out_a = probe(x, mask)
    # mutate the masked token; output must be identical
    x2 = x.clone()
    x2[0, 3] = torch.randn(16) * 100
    out_b = probe(x2, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_positive_proba_packed():
    probe = AttnPoolProbe(d_model=16).eval()
    # two sequences of length 3 and 5
    tokens = torch.randn(8, 16)
    offsets = np.array([0, 3, 8])
    probs = probe.positive_proba(tokens, offsets, device="cpu")
    assert probs.shape == (2,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_iter_packed_batches_padding():
    tokens = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    offsets = np.array([0, 3, 8])
    batches = list(iter_packed_batches(tokens, offsets, np.array([0, 1]), 8, "cpu"))
    assert len(batches) == 1
    x, mask, bidx = batches[0]
    assert x.shape == (2, 5, 4)  # padded to longest (len 5)
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True, True]


def test_save_load_roundtrip(tmp_path):
    probe = AttnPoolProbe(d_model=32).eval()
    path = os.path.join(tmp_path, "p.pt")
    save_probe(probe, path, head="refusal", layer=18)
    loaded = load_probe(path, d_model=32)
    for k in probe.state_dict():
        assert torch.allclose(probe.state_dict()[k], loaded.state_dict()[k])
    x = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert torch.allclose(probe(x, mask), loaded(x, mask), atol=1e-6)


def test_bundled_weights_load():
    """All nine shipped L18 heads load into AttnPoolProbe with d_model=2560."""
    names = ["refusal"] + [f"domain_{d}" for d in DOMAINS]
    assert len(names) == 9
    for name in names:
        path = os.path.join(WEIGHTS_DIR, f"L18_attn_pool_{name}.pt")
        assert os.path.exists(path), f"missing bundled weight: {path}"
        probe = load_probe(path, d_model=D_MODEL)
        assert isinstance(probe, AttnPoolProbe)
        assert probe.query.shape == (D_MODEL,)
        assert probe.head.weight.shape == (2, D_MODEL)
