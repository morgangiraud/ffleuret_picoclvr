#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


def rpl_exec(program, stack):
    for op in program:
        if op == "add":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(a + b)
        elif op == "min":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(min(a, b))
        elif op == "max":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(max(a, b))
        elif op == "swp":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack.append(a)
                stack.append(b)
        elif op == "rep":
            if len(stack) > 1:
                a, b = stack.pop(), stack.pop()
                stack += [b] * a
        elif op == "dup":
            if len(stack) > 0:
                a = stack.pop()
                stack.append(a)
                stack.append(a)
        elif op == "del":
            if len(stack) > 0:
                a = stack.pop()
        else:
            raise ValueError(f"Unknown instruction {op}")


rpl_ops = ["add", "min", "max", "swp", "rep", "dup", "del"]

######################################################################


def generate(nb_values=3, max_input=9, prog_len=6, nb_runs=5):
    prog_len = 1 + torch.randint(prog_len - 1, (1,)).item()
    prog = [rpl_ops[k] for k in torch.randint(len(rpl_ops), (prog_len,))]

    result = []
    for _ in range(nb_runs):
        stack = [x.item() for x in torch.randint(max_input + 1, (nb_values,))]
        result = result + ["<input>"] + stack
        rpl_exec(prog, stack)
        result = result + ["<output>"] + stack

    result = result + ["<prog>"] + prog
    result = result + ["<end>"]
    return result


def next_marker(seq, tokens, start=0):
    pos = None
    for t in tokens:
        try:
            i = seq.index(t, start)
            if pos is None or i < pos:
                pos = i
        except ValueError:
            pass
    return pos


def check(seq):
    io = []
    k = 0
    while seq[k] == "<input>":
        o = next_marker(seq, ["<output>"], start=k + 1)
        e = next_marker(seq, ["<input>", "<prog>"], start=o)
        if o is None or e is None:
            raise ValueError("Invalid input/output")
        io.append((seq[k + 1 : o], seq[o + 1 : e]))
        k = e

    if seq[k] == "<prog>":
        e = next_marker(seq, ["<end>"], start=k)
        if e is None:
            prog = []
        else:
            prog = seq[k + 1 : e]

    nb_total, nb_errors = 0, 0

    if len(set(prog) - set(rpl_ops)) > 0:
        for stack, target_stack in io:
            nb_total += len(target_stack)
            nb_errors += len(target_stack)

    else:
        for stack, target_stack in io:
            # print(f"INIT {stack} PROG {prog}")
            rpl_exec(prog, stack)
            # print(f"CHECK {stack} REF {target_stack} NB_ERROR {abs(len(stack) - len(target_stack))+sum([0 if x == y else 1 for x, y in zip(stack, target_stack)])}")
            nb_total += len(target_stack)
            nb_errors += abs(len(stack) - len(target_stack))
            nb_errors += sum([0 if x == y else 1 for x, y in zip(stack, target_stack)])

    return nb_total, nb_errors


######################################################################

if __name__ == "__main__":
    seq = generate()
    print(seq)
    seq[3] = 7
    print(seq)
    print(check(seq))
