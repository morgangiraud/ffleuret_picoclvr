import torch
import numpy as np

import mygpt
from utils import seed_everything


def test_sanity():
    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, 5))

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=4,
        dim_keys=2,
        dim_hidden=2,
        nb_heads=2,
        nb_blocks=2,
        dropout=0.1,
        causal=True,
    )

    model.eval()
    y1 = model(mygpt.BracketedSequence(x)).x
    y2 = torch.randn_like(y1)
    for s in range(x.size(1)):
        z = model(mygpt.BracketedSequence(x, s, 1))
        y2[:, s] = z.slice()

    error = ((y1 - y2).norm() / (y1.norm() + y2.norm())).item()
    print(f"error: {error}")
    assert error < 1e-6


def test_bracketed_sequence():
    B = 3
    S = 4
    x = torch.arange(0, S).repeat(B, 1)

    bs = mygpt.BracketedSequence(x)
    complete_out = bs.complete()
    assert complete_out is True

    bs = mygpt.BracketedSequence(x, 1)
    sliced_x = bs.slice()
    complete_out = bs.complete()
    assert sliced_x.sum() == x[:, 1:].sum()
    assert complete_out is False

    bs = mygpt.BracketedSequence(x, 1, 1)
    sliced_x = bs.slice()
    complete_out = bs.complete()
    assert sliced_x.sum() == x[:, 1:2].sum()
    assert complete_out is False


def test_cache_wrapper():
    B = 3
    S = 4
    D = 5
    x = torch.ones(B, S, D)

    class AddOne(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    bs = mygpt.BracketedSequence(x, 1, 2)
    cache = mygpt.CacheWrapper(AddOne())

    cached_bs = cache(bs)
    assert (x[:, 1:3] + 1).sum() == cached_bs.x.sum()


def test_qkv_attention():
    B = 3
    S = 4
    D = 5
    Q = 6
    V = 7
    x = torch.rand(B, S, D)

    seed = 42

    with torch.no_grad():
        bs = mygpt.BracketedSequence(x)

        seed_everything(seed)
        att = mygpt.QKVAttention(D, Q, V, 1, False, 0.0)

        seed_everything(seed)
        att_fast = mygpt.QKVAttentionFast(D, Q, V, 1, False, 0.0)

        np.testing.assert_equal(att.w_q.numpy(), att_fast.w_q.numpy())
        np.testing.assert_equal(att.w_k.numpy(), att_fast.w_k.numpy())
        np.testing.assert_equal(att.w_v.numpy(), att_fast.w_v.numpy())
        np.testing.assert_equal(att.w_o.numpy(), att_fast.w_o.numpy())

        bs_y = att(bs)
        bs_y_fast = att_fast(bs)

        np.testing.assert_almost_equal(bs_y.x.numpy(), bs_y_fast.x.numpy(), decimal=5)
