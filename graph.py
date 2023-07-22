#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

import cairo


######################################################################
def save_attention_image(
    filename,
    tokens_input,
    tokens_output,
    attention,
    n_sample=0,
    n_head=0,
    pixel_scale=8,
    token_gap=10,
    layer_gap=25,
    y_eps=0.5,
    padding=10,
    min_att=0,
    k_top=None,
):
    attention = torch.cat(
        [x[n_sample : n_sample + 1, n_head] for x in attention], dim=0
    )

    if k_top is not None:
        attention = attention * (
            attention.sort(dim=-1, descending=True).indices < k_top
        )

    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)

    ctx = cairo.Context(surface)
    ctx.scale(pixel_scale, pixel_scale)

    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_font_size(4.0)
    # ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    x, y = 0, 0

    for d in range(attention.size(0)):
        if d > 0:
            for n in range(attention.size(-1)):
                xc, yc = n * token_gap, -d * layer_gap
                ctx.arc(xc, yc, token_gap / 10, 0, 2 * math.pi)
                ctx.fill()

        at = attention[d]
        ni = torch.arange(at.size(0))[:, None].expand_as(at)
        nj = torch.arange(at.size(1))[None, :].expand_as(at)
        at = at.flatten()
        o = at.sort().indices
        at = at[o]
        ni = ni.flatten()[o]
        nj = nj.flatten()[o]
        for i, j, a in zip(ni, nj, at):
            if a > 0 and a >= min_att:
                c = 1 - a.item()
                ctx.set_source_rgb(c, c, c)
                ctx.set_line_width(0.5)
                ctx.move_to(j * token_gap, y - y_eps)
                ctx.line_to(i * token_gap, y - layer_gap + y_eps)
                ctx.stroke()
        y -= layer_gap

    for d in range(1, attention.size(0)):
        for n in range(attention.size(-1)):
            xc, yc = n * token_gap, -d * layer_gap
            ctx.set_source_rgb(1.0, 1.0, 1.0)
            ctx.arc(xc, yc, token_gap / 10 + 0.5, 0, 2 * math.pi)
            ctx.fill()
            ctx.set_source_rgb(0.0, 0.0, 0.0)
            ctx.arc(xc, yc, token_gap / 10, 0, 2 * math.pi)
            ctx.fill()

    ctx.set_source_rgb(0.0, 0.0, 0.0)

    for k, t in enumerate(tokens_input):
        s = str(t)
        (
            x_bearing,
            y_bearing,
            width_t,
            height_t,
            x_advance,
            y_advance,
        ) = ctx.text_extents(s)
        ctx.move_to(k * token_gap - width_t / 2, -y_bearing)
        ctx.show_text(s)

    for k, t in enumerate(tokens_output):
        s = str(t)
        (
            x_bearing,
            y_bearing,
            width_t,
            height_t,
            x_advance,
            y_advance,
        ) = ctx.text_extents(s)
        ctx.move_to(k * token_gap - width_t / 2, -attention.size(0) * layer_gap)
        ctx.show_text(s)

    x, y, width, height = surface.ink_extents()
    x -= padding
    y -= padding
    width += 2 * padding
    height += 2 * padding
    pdf_surface = cairo.PDFSurface(filename, width, height)
    ctx_pdf = cairo.Context(pdf_surface)
    ctx_pdf.set_source_surface(surface, -x, -y)
    ctx_pdf.paint()
    pdf_surface.finish()


######################################################################

if __name__ == "__main__":
    import mygpt

    tokens_output = ["bluh", 2, 3, 4, "blih"]
    tokens_input = ["n/a"] + tokens_output[:-1]

    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, len(tokens_input)))

    model = mygpt.MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=4,
        dim_keys=2,
        dim_hidden=2,
        nb_heads=2,
        nb_blocks=3,
        dropout=0.1,
        causal=True,
    )

    model.eval()
    model.record_attention()

    y1 = model(mygpt.BracketedSequence(x)).x

    attention = model.retrieve_attention()

    save_attention_image("attention.pdf", tokens_input, tokens_output, attention)
