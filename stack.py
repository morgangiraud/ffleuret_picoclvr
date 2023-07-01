#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision

######################################################################

# CODE_OP=[0 for push, 1 for pop] + 2 * n_stack
# CODE_VAL=val + 2 * nb_stacks


def generate(nb, nb_steps, nb_stacks, nb_values):
    stack = torch.empty(nb, nb_stacks, nb_steps, dtype=torch.int64)
    stack_pointers = torch.zeros(nb, nb_stacks, dtype=torch.int64)
    k = torch.arange(nb)
    result = torch.empty(nb, 2 * nb_steps, dtype=torch.int64)
    depth_counts = torch.zeros(nb, 2 * nb_steps, dtype=torch.int64)

    for t in range(nb_steps):
        op = torch.randint(2, (nb,))
        st = torch.randint(nb_stacks, (nb,))
        op = op * (stack_pointers[k, st] > 0)
        val_push = torch.randint(nb_values, (nb,))
        val_pop = stack[
            k,
            st,
            (stack_pointers[k, st] - 1).clamp(min=0),
        ]
        stack[k, st, stack_pointers[k, st]] = val_push
        depth_counts[:, 2 * t + 1] = stack_pointers[k, st]
        stack_pointers[k[op == 0], st[op == 0]] += 1
        stack_pointers[k[op == 1], st[op == 1]] -= 1
        result[:, 2 * t] = st * 2 + op
        result[:, 2 * t + 1] = (op * val_pop + (1 - op) * val_push) + 2 * nb_stacks

    return result, depth_counts


def seq_to_str(seq, depth_counts=None):
    assert seq.size(0) % 2 == 0
    s = ""
    for t in range(seq.size(0) // 2):
        op = seq[2 * t]
        op = f"POP_{op//2}" if op % 2 == 1 else f"PUSH_{op//2}"
        val = seq[2 * t + 1] - 2 * nb_stacks
        if t > 0:
            s += " "
        if depth_counts is not None:
            s += f"[{depth_counts[2*t+1]}] "
        s += f"{op} {val}"
    return s


######################################################################

if __name__ == "__main__":
    nb, nb_steps, nb_stacks, nb_values = 150000, 10, 1, 5
    seq, depth_counts = generate(
        nb=nb, nb_steps=nb_steps, nb_stacks=nb_stacks, nb_values=nb_values
    )

    for n in range(min(10, seq.size(0))):
        print(seq_to_str(seq[n], depth_counts[n]))
