#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


def rpl_exec(program, stack):
    stack = stack.copy()
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

    return stack


rpl_ops = ["add", "min", "max", "swp", "rep", "dup", "del"]

######################################################################


def generate(nb_values=3, max_input=9, prog_len=6, nb_runs=5):
    prog_len = 1 + torch.randint(prog_len - 1, (1,)).item()
    prog = [rpl_ops[k] for k in torch.randint(len(rpl_ops), (prog_len,))]

    result = []
    for _ in range(nb_runs):
        stack = [x.item() for x in torch.randint(max_input + 1, (nb_values,))]
        result_stack = rpl_exec(prog, stack)
        result = result + ["<input>"] + stack + ["<output>"] + result_stack

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


def decompose(seq):
    io = []
    k = 0
    while seq[k] == "<input>":
        o = next_marker(seq, ["<output>"], start=k + 1)
        e = next_marker(seq, ["<input>", "<prog>"], start=o)
        if o is None or e is None:
            raise ValueError("Invalid input/output")
        try:
            io.append(
                ([int(x) for x in seq[k + 1 : o]], [int(x) for x in seq[o + 1 : e]])
            )
        except ValueError:
            raise ValueError("Invalid input/output")

        k = e

    if seq[k] == "<prog>":
        e = next_marker(seq, ["<end>"], start=k)
        if e is None:
            prog = []
        else:
            prog = seq[k + 1 : e]
    return prog, io


def compute_nb_errors(seq):
    prog, io = decompose(seq)

    nb_total, nb_errors = 0, 0

    stacks = []

    if len(set(prog) - set(rpl_ops)) > 0:
        # Program is not valid, we count 100% error
        for start_stack, target_stack in io:
            stacks.append((start_stack, target_stack, "N/A", False))
            nb_total += len(target_stack)
            nb_errors += len(target_stack)

    else:
        # Program is valid
        for start_stack, target_stack in io:
            result_stack = rpl_exec(prog, start_stack)
            nb_total += len(target_stack)
            e = abs(len(result_stack) - len(target_stack)) + sum(
                [0 if x == y else 1 for x, y in zip(result_stack, target_stack)]
            )
            nb_errors += e
            stacks.append((start_stack, target_stack, result_stack, e == 0))

    return nb_total, nb_errors, prog, stacks


######################################################################

if __name__ == "__main__":
    seq = generate()
    print(seq)
    seq[3] = 7
    print(seq)
    print(compute_nb_errors(seq))
