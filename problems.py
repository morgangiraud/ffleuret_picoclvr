#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


class Problem:
    def generate_sequences(self, nb):
        pass

    def seq2str(self, seq):
        return "[NOT IMPLEMENTED]"

    def compute_nb_correct(self, input, ar_mask, result):
        nb_total = ar_mask.sum().item()
        nb_correct = ((result == input).long() * ar_mask).sum().item()
        return nb_total, nb_correct

####################


class ProblemTwoCuts(Problem):
    def __init__(self, len_total=50, nb_values=100, global_constraint=True):
        self.len_total = len_total
        self.nb_values = nb_values
        self.global_constraint = global_constraint

    def generate_sequences_internal(self, nb):
        return u,v,a,b,c

    def generate_sequences(self,nb):

        u = torch.randint(self.len_total, (nb,))
        v = torch.randint(self.len_total, (nb,))

        a = torch.randint(self.nb_values, (nb,))
        b = torch.randint(self.nb_values, (nb,))
        c = torch.randint(self.nb_values, (nb,))

        while True:
            to_compute = torch.logical_or(u>=v-self.len_total//10,u<v-self.len_total//5)
            to_compute =torch.logical_or(to_compute, u == 0)
            to_compute =torch.logical_or(to_compute, v == self.len_total)
            n = to_compute.long().sum()
            if n == 0:
                break
            else:
                u[to_compute] = torch.randint(self.len_total, (n,))
                v[to_compute] = torch.randint(self.len_total, (n,))

        while True:
            to_compute = a==b
            to_compute = torch.logical_or(to_compute,b==c)
            to_compute = torch.logical_or(to_compute,a==c)

            if self.global_constraint:
                to_compute = torch.logical_or(to_compute,(a*u+b*(v-u)+c*(self.len_total-v)) // self.len_total != self.nb_values//2)

            n = to_compute.long().sum()
            if n == 0:
                break
            else:
                a[to_compute] = torch.randint(self.nb_values, (n,))
                b[to_compute] = torch.randint(self.nb_values, (n,))
                c[to_compute] = torch.randint(self.nb_values, (n,))

        assert (u>=v).long().sum() == 0
        assert (a==b).long().sum() == 0
        assert (a==c).long().sum() == 0
        assert (c==b).long().sum() == 0

        t = torch.arange(self.len_total)
        seq = (t[None,:] < u[:,None]).long() * a[:,None] + \
            (t[None,:] >= u[:,None]).long() * (t[None,:] < v[:,None]).long() * b[:,None] + \
            (t[None,:] >= v[:,None]).long() * c[:,None]

        return seq,seq.new_full(seq.size(), 1, dtype=torch.int64)

    def compute_nb_correct(self, input, ar_mask, result):
        nb_total = result.size(0)
        nb_correct = 0
        i = torch.arange(result.size(1), device=result.device)

        for k in range(nb_total):
            s = result[k]
            a = s[0]
            uu = (s != a).nonzero()
            if uu.size(0) > 0:
                u = uu.min()
                b = s[u]
                vv = torch.logical_and(s != b, i >= u).nonzero()
                if vv.size(0) > 0:
                    v = vv.min()
                    c = s[v]
                    ww = torch.logical_and(s != c, i >= v).nonzero()
                    if ww.size(0) == 0:
                        if not self.global_constraint or (a*u+b*(v-u)+c*(self.len_total-v)) // self.len_total == self.nb_values//2:
                            nb_correct += 1

        return nb_total, nb_correct

    def seq2str(self, seq):
        return " ".join( [ f"{x:02d}" for x in seq ] )

####################


class ProblemTwoTargets(Problem):
    def __init__(self, len_total=10, len_targets=3):
        assert len_targets >= 3
        assert len_total >= 3 * len_targets - 1
        self.len_total = len_total
        self.len_targets = len_targets

    def generate_sequences(self, nb):
        k = torch.arange(self.len_total)[None, :]
        s = torch.randint(10, (nb, self.len_total))
        l = torch.rand(nb, self.len_total)
        l = l * (k <= self.len_total - self.len_targets).long()
        k1 = l.argmax(dim=1, keepdim=True)
        m = (k != k1).long() * (k != k1 + self.len_targets - 1).long()
        s = s * m + 10 * (1 - m)
        l = l * (
            1
            - (k + self.len_targets - 1 >= k1).long()
            * (k < k1 + self.len_targets).long()
        )
        k2 = l.argmax(dim=1, keepdim=True)
        m = (k != k2).long() * (k != k2 + self.len_targets - 1).long()
        s = s * m + 11 * (1 - m)
        a1 = s.gather(dim=1, index=k1 + 1 + torch.arange(self.len_targets - 2)[None, :])
        a2 = s.gather(dim=1, index=k2 + 1 + torch.arange(self.len_targets - 2)[None, :])
        sequences = torch.cat(
            (
                s,
                torch.full((nb, 1), 12),
                a1,
                torch.full((nb, 1), 12),
                a2,
                torch.full((nb, 1), 12),
            ),
            1,
        )
        ar_mask = (sequences == 12).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789-+|"[x.item()] for x in seq)


####################


class ProblemByHeart(Problem):
    def __init__(self, nb_sentences=100, len_prompt=8, len_result=8):
        self.seq = torch.randint(10, (nb_sentences, len_prompt + 1 + len_result))
        self.seq[:, len_prompt] = 10

    def generate_sequences(self, nb):
        sequences = self.seq[torch.randint(self.seq.size(0), (nb,))]
        ar_mask = (sequences == 10).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789|"[x.item()] for x in seq)


####################


class ProblemLearnOperator(Problem):
    def __init__(self, nb_operators=100, len_source=6, len_result=9):
        self.len_source = len_source
        self.len_result = len_result
        self.len_nb_operator = int(math.log(nb_operators) / math.log(10)) + 1
        self.operators = F.one_hot(
            torch.rand(nb_operators, len_result, len_source).argmax(-1),
            num_classes=len_source,
        )

    def generate_sequences(self, nb):
        nb_operators = torch.randint(self.operators.size(0), (nb,))
        operators = self.operators[nb_operators]
        nb_operators = (
            nb_operators[:, None]
            // 10 ** torch.arange(self.len_nb_operator - 1, -1, -1)
        ) % 10
        marker1 = torch.full((nb, 1), 10)
        source = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        marker2 = torch.full((nb, 1), 11)
        result = operators.bmm(source[:, :, None]).squeeze(-1)
        sequences = torch.cat((nb_operators, marker1, source, marker2, result), 1)
        ar_mask = (sequences == 11).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789|>"[x.item()] for x in seq)


####################


class ProblemGuessOperator(Problem):
    def __init__(self, len_source=5, len_result=8):
        self.len_source = len_source
        self.len_result = len_result

    def generate_sequences(self, nb):
        operators = F.one_hot(
            torch.rand(nb, self.len_result, self.len_source).argmax(-1),
            num_classes=self.len_source,
        )
        source1 = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        marker1 = torch.full((nb, 1), 10)
        result1 = operators.bmm(source1[:, :, None]).squeeze(-1)
        marker2 = torch.full((nb, 1), 11)
        source2 = torch.randint(10, (nb, self.len_source))
        marker3 = torch.full((nb, 1), 12)
        result2 = operators.bmm(source2[:, :, None]).squeeze(-1)

        sequences = torch.cat(
            (source1, marker1, result1, marker2, source2, marker3, result2), 1
        )
        ar_mask = (sequences == 12).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789>|~"[x.item()] for x in seq)


####################


class ProblemAddition(Problem):
    def __init__(self, nb_digits=10, zero_padded=False, inverted_result=False):
        self.nb_digits = nb_digits
        self.zero_padded = zero_padded
        self.inverted_result = inverted_result
        self.char2id = dict([(c, n) for n, c in enumerate("0123456789+=$")])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

    def tensorize(self, strings):
        len_max = max([len(x) for x in strings])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s + "$" * (len_max - len(s))]
                        for s in strings
                    ]
                )
            ],
            0,
        )

    def generate_sequences(self, nb):
        sequences = []
        for k in range(nb):
            a, b = torch.randint(10**self.nb_digits, (2,))
            c = a + b
            a, b, c = str(a.item()), str(b.item()), str(c.item())
            if self.zero_padded:
                a = "0" * (self.nb_digits - len(a)) + a
                b = "0" * (self.nb_digits - len(b)) + b
                c = "0" * (self.nb_digits + 1 - len(c)) + c
            if self.inverted_result:
                c = c[::-1]
            sequences.append(f"{a}+{b}={c}$")

        sequences = self.tensorize(sequences)
        ar_mask = (sequences == self.char2id["="]).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join(self.id2char[x.item()] for x in seq)


if __name__ == "__main__":
    p = ProblemTwoCuts(12)
    s, m = p.generate_sequences(10000)
    print(p.compute_nb_correct(None, None, s))
