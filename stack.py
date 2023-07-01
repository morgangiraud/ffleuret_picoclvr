#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision

######################################################################

# CODE_OP=[0 for push, 1 for pop] + 2 * n_stack
# CODE_VAL=val + 2 * nb_stacks


def generate(nb, seq_len, nb_stacks, nb_values):
    stack = torch.empty(nb, nb_stacks, seq_len, dtype=torch.int64)
    stack_pointers = torch.zeros(nb, nb_stacks, dtype=torch.int64)
    k = torch.arange(nb)
    result = torch.empty(nb, 2 * seq_len, dtype=torch.int64)

    for t in range(seq_len):
        op = torch.randint(2, (nb,))
        st = torch.randint(nb_stacks, (nb,))
        op = op * (stack_pointers[k, st] > 0)
        val_push = torch.randint(nb_values, (nb,))
        # top_val[n,s]=stack[n,stack_pointers[n,s]]
        top_values = stack[
            k,
            st,
            (stack_pointers[k, st] - 1).clamp(min=0),
        ]
        stack[
            k[:, None].expand_as(stack_pointers),
            st[:, None].expand_as(stack_pointers),
            stack_pointers,
        ] = val_push[:, None].expand_as(stack_pointers)
        stack_pointers[k[op == 0], st[op == 0]] += 1
        stack_pointers[k[op == 1], st[op == 1]] -= 1
        result[:, 2 * t] = st * 2 + op
        result[:, 2 * t + 1] = (op * top_values + (1 - op) * val_push) + 2 * nb_stacks

    return result


def seq_to_str(seq):
    assert seq.size(0) % 2 == 0
    s = ""
    for t in range(0, seq.size(0), 2):
        op = seq[t]
        op = f"POP_{op//2}" if op % 2 == 1 else f"PUSH_{op//2}"
        val = seq[t + 1] - 2 * nb_stacks
        if t > 0:
            s += " "
        s += f"{op} {val}"
    return s


######################################################################

if __name__ == "__main__":
    nb, seq_len, nb_stacks, nb_values = 3, 10, 1, 5
    result = generate(nb=nb, seq_len=seq_len, nb_stacks=nb_stacks, nb_values=nb_values)
    for n in range(result.size(0)):
        print(seq_to_str(result[n]))
