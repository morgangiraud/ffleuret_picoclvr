#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

from torch import Tensor

import sys


def exception_hook(exc_type, exc_value, tb):
    r"""Hacks the call stack message to show all the local variables in
    case of RuntimeError or ValueError, and prints tensors as shape,
    dtype and device.

    """

    repr_orig = Tensor.__repr__
    Tensor.__repr__ = lambda x: f"{x.size()}:{x.dtype}:{x.device}"

    while tb:
        print("--------------------------------------------------\n")
        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        line_no = tb.tb_lineno
        print(f'  File "{filename}", line {line_no}, in {name}')
        print(open(filename, "r").readlines()[line_no - 1])

        if exc_type in {RuntimeError, ValueError}:
            for n, v in tb.tb_frame.f_locals.items():
                print(f"  {n} -> {v}")

        print()
        tb = tb.tb_next

    Tensor.__repr__ = repr_orig

    print(f"{exc_type.__name__}: {exc_value}")


sys.excepthook = exception_hook

######################################################################

if __name__ == "__main__":

    import torch

    def dummy(a, b):
        print(a @ b)

    def blah(a, b):
        c = b + b
        dummy(a, c)

    mmm = torch.randn(2, 3)
    xxx = torch.randn(3)
    # print(xxx@mmm)
    blah(mmm, xxx)
    blah(xxx, mmm)
