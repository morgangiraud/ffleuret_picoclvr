#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# This is an implementation from scratch of a "GPT", that is a model
# composed of several causal self-attention blocks. It is equipped
# with a caching mechanism for keys and values to avoid a O(N^3) cost
# for auto-regression.

import math
from typing import Optional

import torch

from torch import nn
from torch.nn import functional as F

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
    def __init__(self, x: torch.Tensor, first: int = 0, nb: Optional[int] = None):
        self.x = x

        assert type(first) == int
        self.first = first

        self.nb = nb if type(nb) == int and nb > 0 else x.size(1)

    def slice(self):
        return self.x[:, self.first : self.first + self.nb]

    def complete(self):
        return self.first == 0 and self.nb == self.x.size(1)


######################################################################


class CacheWrapper(nn.Module):
    def __init__(self, *f: nn.Module):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)
        self.cache_y: Optional[torch.Tensor] = None

    def forward(self, bs: BracketedSequence):
        sliced_x = bs.slice()  # B, T_nb, dims...
        sliced_y = self.f(sliced_x)  # B, T_nb, dims...

        if self.cache_y is None or bs.first == 0:
            y_size: torch.Size = sliced_y.size()
            B = y_size[0]
            T = bs.x.size(1)
            self.cache_y = torch.zeros(*((B, T) + y_size[2:]))  # B, T, dims...

        self.cache_y[:, bs.first : bs.first + bs.nb] = sliced_y

        return BracketedSequence(self.cache_y, bs.first, bs.nb)


##############################


class WithResidual(nn.Module):
    def __init__(self, *f: nn.Module):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs: BracketedSequence):
        return BracketedSequence(bs.x + self.f(bs).x, bs.first, bs.nb)


##############################


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max: int):
        super().__init__()
        self.len_max = len_max

    # [Vaswani et al 2018] PE_{t,2i} = sin(t/(L^{2i/D})), PE_{t,2i+1} = cos(t/(L^{2i/D}))

    def forward(self, bs: BracketedSequence):
        if bs.first == 0:
            t = torch.arange(bs.x.size(1), dtype=bs.x.dtype, device=bs.x.device)[:, None]
            j = torch.arange(bs.x.size(2), dtype=bs.x.dtype, device=bs.x.device)[None, :]
            k = j % 2
            self.pe = torch.sin(t / (self.len_max ** ((j - k) / bs.x.size(2))) + math.pi / 2 * k)
            self.cache_y = bs.x.new(bs.x.size())

        self.cache_y[:, bs.first : bs.first + bs.nb] = bs.slice() + self.pe[bs.first : bs.first + bs.nb]

        return BracketedSequence(self.cache_y, bs.first, bs.nb)


##############################


class QKVAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nb_heads: int = 1,
        causal: bool = False,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        def randw(*d: int):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout
        self.record_attention = False

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, bs_q: BracketedSequence):
        x_q = bs_q.x

        assert self.causal or bs_q.complete(), "Partial evaluation is only possible for causal models"

        if bs_q.first == 0:
            self.cache_k = x_q.new_zeros(x_q.size(0), self.w_k.size(0), x_q.size(1), self.w_k.size(1))
            self.cache_v = x_q.new_zeros(x_q.size(0), self.w_v.size(0), x_q.size(1), self.w_v.size(1))
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        q = torch.einsum("ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_q)

        self.cache_k[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_k
        )
        self.cache_v[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_v
        )

        a = torch.einsum("nhtd,nhsd->nhts", q, self.cache_k[:, :, : bs_q.first + bs_q.nb]) / math.sqrt(self.w_q.size(1))

        if self.causal:
            if bs_q.first == 0:
                self.cache_attzero = (
                    torch.arange(x_q.size(1), device=q.device)[None, None, :, None]
                    < torch.arange(x_q.size(1), device=q.device)[None, None, None, :]
                )
            a = a.masked_fill(
                self.cache_attzero[:, :, bs_q.first : bs_q.first + bs_q.nb, : bs_q.first + bs_q.nb],
                float("-inf"),
            )

        a = a.softmax(dim=3)

        if self.record_attention:
            self.a = a

        a = F.dropout(a, self.attention_dropout, self.training)

        y = torch.einsum("nhts,nhsd->nthd", a, self.cache_v[:, :, : bs_q.first + bs_q.nb]).flatten(2)

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y @ self.w_o

        return BracketedSequence(self.cache_y, bs_q.first, bs_q.nb)


class QKVAttentionFast(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_qk: int,
        dim_v: int,
        nb_heads: int = 1,
        causal: bool = False,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        def randw(*d: int):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.dim_in = dim_in
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.nb_heads = nb_heads

        self.causal = causal
        self.attention_dropout = attention_dropout
        self.record_attention = False

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, bs_q: BracketedSequence):
        assert self.causal or bs_q.complete(), "Partial evaluation is only possible for causal models"

        B, T, _ = bs_q.x.shape
        sliced_x_q = bs_q.slice()

        q = torch.einsum("ntc,hdc->nhtd", sliced_x_q, self.w_q)

        if bs_q.first == 0:
            self.cache_k = torch.zeros(B, self.nb_heads, T, self.dim_qk, device=q.device)
            self.cache_v = torch.zeros(B, self.nb_heads, T, self.dim_v, device=q.device)
            self.cache_y = torch.zeros(B, T, self.dim_in, device=q.device)

        self.cache_k[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum("ntc,hdc->nhtd", sliced_x_q, self.w_k)
        self.cache_v[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum("ntc,hdc->nhtd", sliced_x_q, self.w_v)

        attn_mask = None
        if self.causal:
            if bs_q.first == 0:
                self.cache_attzero = (
                    torch.arange(T, device=q.device)[None, None, :, None]
                    < torch.arange(T, device=q.device)[None, None, None, :]
                )
            attn_mask = self.cache_attzero[:, :, bs_q.first : bs_q.first + bs_q.nb, : bs_q.first + bs_q.nb]

        y = F.scaled_dot_product_attention(
            q, self.cache_k, self.cache_v, attn_mask=attn_mask, dropout_p=self.attention_dropout
        ).squeeze()

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y @ self.w_o

        return BracketedSequence(self.cache_y, bs_q.first, bs_q.nb)


##############################


class MyGPT(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        dim_model: int,
        dim_keys: int,
        dim_hidden: int,
        nb_heads: int,
        nb_blocks: int,
        causal: bool = False,
        dropout: float = 0.0,
        len_max: int = int(1e5),
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
                    CacheWrapper(
                        nn.LayerNorm(
                            [
                                dim_model,
                            ]
                        )
                    ),
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
                        nn.LayerNorm(
                            [
                                dim_model,
                            ]
                        ),
                        nn.Linear(in_features=dim_model, out_features=dim_hidden),
                        nn.ReLU(),
                        nn.Linear(in_features=dim_hidden, out_features=dim_model),
                        nn.Dropout(dropout),
                    ),
                ),
            ]

        self.trunk = nn.Sequential(*trunk_blocks)

        self.readout = CacheWrapper(nn.Linear(in_features=dim_model, out_features=vocabulary_size))

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, bs: BracketedSequence):
        bs = BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb)
        bs = self.embedding(bs)
        bs = self.trunk(bs)
        bs = self.readout(bs)
        return bs

    # ar_mask is a tensor with 0s and 1s, of same shape as input, with
    # 1s where tokens should be generated. The others are kept
    # unchanged.

    def masked_inplace_autoregression(
        self, input: torch.Tensor, ar_mask: torch.Tensor, forbidden_tokens=None, deterministic_synthesis=False
    ):
        to_generate = (ar_mask.sum(0) > 0).nonzero()
        if to_generate.min() > 0:
            self(BracketedSequence(input, 0, int(to_generate.min().item())))  # Needed to initialize the model's cache
        for s in range(to_generate.min(), to_generate.max() + 1):
            output = self(BracketedSequence(input, s, 1)).x
            logits = output[:, s]
            if forbidden_tokens is not None:
                logits = logits.masked_fill(forbidden_tokens, float("-inf"))
            if deterministic_synthesis:
                t_next = logits.argmax(1)
            else:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                t_next = dist.sample()
            input[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * input[:, s]

    def record_attention(self, v=True):
        for m in self.modules():
            if isinstance(m, QKVAttention):
                m.record_attention = v

    def retrieve_attention(self):
        a = []
        for m in self.modules():
            if isinstance(m, QKVAttention):
                a.append(m.a)
        return a
