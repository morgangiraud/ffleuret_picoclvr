#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math

import torch

from torch import nn
from torch.nn import functional as F

######################################################################


class WithResidual(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        bs.x = bs.x + self.f(bs).x
        return bs


######################################################################

# A BracketedSequence is a BxTx... tensor with a first and a nb time
# steps to compute.

# Modules able to process it expect that they will have to process a
# first bracket starting at t=0, followed by a succession of brackets
# that move forward in time, do not overlap, and cover the axis T with
# no holes.
#
# Although it is more general, for a classical prompt-conditioned
# auto-regressive process it will be a first bracket starting at 0 and
# of arbitrary length for the "prompt", followed by brackets of length
# 1 for the successive tokens.
#
# Modules able to process brackets may implement a cache that is
# resetted when the input bracket starts at t=0


class BracketedSequence:
    def __init__(self, x, first=None, nb=None):
        self.x = x
        self.first = 0 if first is None else first
        self.nb = x.size(1) if nb is None else nb

    def slice(self):
        return self.x[:, self.first : self.first + self.nb]


######################################################################


class CacheWrapper(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        if bs.first == 0:
            y = self.f(bs.slice())
            self.cache_y = y.new(*((y.size(0), bs.x.size(1)) + y.size()[2:]))
            self.cache_y[:, bs.first : bs.first + bs.nb] = y
        else:
            self.cache_y[:, bs.first : bs.first + bs.nb] = self.f(bs.slice())

        bs.x = self.cache_y

        return bs


##############################


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max):
        super().__init__()
        self.len_max = len_max

    # [Vaswani et al 2018] PE_{t,2i} = sin(t/(L^{2i/D})), PE_{t,2i+1} = cos(t/(L^{2i/D}))

    def forward(self, bs):
        if bs.first == 0:
            t = torch.arange(bs.x.size(1), dtype=bs.x.dtype, device=bs.x.device)[
                :, None
            ]
            j = torch.arange(bs.x.size(2), dtype=bs.x.dtype, device=bs.x.device)[
                None, :
            ]
            k = j % 2
            self.pe = torch.sin(
                t / (self.len_max ** ((j - k) / bs.x.size(2))) + math.pi / 2 * k
            )
            self.cache_y = bs.x.new(bs.x.size())

        self.cache_y[:, bs.first : bs.first + bs.nb] = (
            bs.slice() + self.pe[bs.first : bs.first + bs.nb]
        )

        bs.x = self.cache_y

        return bs


##############################


class QKVAttention(nn.Module):
    def __init__(
        self, dim_in, dim_qk, dim_v, nb_heads=1, causal=False, attention_dropout=0.0
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, bs_q):
        x_q = bs_q.x

        if bs_q.first == 0:
            self.cache_k = x_q.new_zeros(
                x_q.size(0), self.w_k.size(0), x_q.size(1), self.w_k.size(1)
            )
            self.cache_v = x_q.new_zeros(
                x_q.size(0), self.w_v.size(0), x_q.size(1), self.w_v.size(1)
            )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        q = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_q
        )
        self.cache_k[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_k
        )
        self.cache_v[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_v
        )

        a = torch.einsum(
            "nhtd,nhsd->nhts", q, self.cache_k[:, :, : bs_q.first + bs_q.nb]
        ) / math.sqrt(self.w_q.size(1))

        if self.causal:
            if bs_q.first == 0:
                self.cache_attzero = (
                    torch.arange(x_q.size(1), device=q.device)[None, None, :, None]
                    < torch.arange(x_q.size(1), device=q.device)[None, None, None, :]
                )
            a = a.masked_fill(
                self.cache_attzero[
                    :, :, bs_q.first : bs_q.first + bs_q.nb, : bs_q.first + bs_q.nb
                ],
                float("-inf"),
            )

        a = a.softmax(dim=3)
        a = F.dropout(a, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhts,nhsd->nthd", a, self.cache_v[:, :, : bs_q.first + bs_q.nb]
        ).flatten(2)

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y @ self.w_o

        bs_q.x = self.cache_y

        return bs_q


##############################


class MyGPT(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        dim_model,
        dim_keys,
        dim_hidden,
        nb_heads,
        nb_blocks,
        causal=False,
        dropout=0.0,
        len_max=1e5,
    ):
        super().__init__()

        assert dim_model % nb_heads == 0

        self.embedding = nn.Sequential(
            CacheWrapper(nn.Embedding(vocabulary_size, dim_model), nn.Dropout(dropout)),
            AddPositionalEncoding(len_max),
        )

        trunk_blocks = []

        for b in range(nb_blocks):
            trunk_blocks += [
                WithResidual(
                    CacheWrapper(nn.LayerNorm((dim_model,))),
                    QKVAttention(
                        dim_in=dim_model,
                        dim_qk=dim_keys,
                        dim_v=dim_model // nb_heads,
                        nb_heads=nb_heads,
                        causal=causal,
                        attention_dropout=dropout,
                    ),
                ),
                WithResidual(
                    CacheWrapper(
                        nn.LayerNorm((dim_model,)),
                        nn.Linear(in_features=dim_model, out_features=dim_hidden),
                        nn.ReLU(),
                        nn.Linear(in_features=dim_hidden, out_features=dim_model),
                        nn.Dropout(dropout),
                    ),
                ),
            ]

        self.trunk = nn.Sequential(*trunk_blocks)

        self.readout = CacheWrapper(
            nn.Linear(in_features=dim_model, out_features=vocabulary_size)
        )

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, bs):
        bs.x = F.pad(bs.x, (1, -1))
        bs = self.embedding(bs)
        bs = self.trunk(bs)
        bs = self.readout(bs)
        return bs


######################################################################

if __name__ == "__main__":
    print("Basic check.")

    vocabulary_size = 10
    x = torch.randint(vocabulary_size, (9, 7))

    model = MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=18,
        dim_keys=50,
        dim_hidden=100,
        nb_heads=2,
        nb_blocks=1,
        dropout=0.1,
    )

    model.eval()

    y1 = model(BracketedSequence(x)).x

    y2 = torch.randn_like(y1)
    for s in range(x.size(1)):
        z = model(BracketedSequence(x, s, 1))
        y2[:, s] = z.x[:, s]

    # print(y1.max(dim = 2).values)
    # print(y2.max(dim = 2).values)
    print(f"error={((y1 - y2).norm() / (y1.norm() + y2.norm())).item()}")

######################################################################
