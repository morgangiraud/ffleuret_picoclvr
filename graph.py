#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

import cairo


######################################################################
def save_attention_image(
    filename,
    tokens,
    attention,
    surface_width=128,
    surface_height=96,
    pixel_scale=8,
    x=10,
    y=10,
    token_gap=15,
    layer_gap=25,
    y_eps=1,
    min_att=1e-2,
):
    # surface = cairo.PDFSurface(
    # filename, surface_width * pixel_scale, surface_height * pixel_scale
    # )

    surface = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)

    ctx = cairo.Context(surface)
    ctx.scale(pixel_scale, pixel_scale)

    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_font_size(4.0)
    # ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    u = []
    for n, t in enumerate(tokens):
        string = str(t)
        (
            x_bearing,
            y_bearing,
            width_t,
            height_t,
            x_advance,
            y_advance,
        ) = ctx.text_extents(string)
        u.append((n, string, x, x + width_t / 2, height_t, y_bearing))
        x += x_advance + token_gap
    tokens = u

    for d in range(attention.size(0) + 1):
        for n, s, x, xc, h, yb in tokens:
            # ctx.set_source_rgb(0.0, 0.0, 0.0)
            # ctx.rectangle(x+x_bearing,y+y_bearing,width_t,height_t)
            # ctx.stroke()
            ctx.set_source_rgb(0.0, 0.0, 0.0)
            ctx.move_to(x, y)
            ctx.show_text(s)
            # x += x_advance + 1
            if d < attention.size(0):
                for m, _, _, x2c, h2, y2b in tokens:
                    if attention[d, n, m] >= min_att:
                        c = 1 - attention[d, n, m]
                        ctx.set_source_rgb(c, c, c)
                        ctx.set_line_width(0.5)
                        ctx.move_to(xc, y + yb + h + y_eps)
                        ctx.line_to(x2c, y + layer_gap + y2b - y_eps)
                        ctx.stroke()
        y += layer_gap

    x, y, width, height = surface.ink_extents()
    pdf_surface = cairo.PDFSurface(filename, width, height)
    ctx_pdf = cairo.Context(pdf_surface)
    ctx_pdf.set_source_surface(surface, -x, -y)
    ctx_pdf.paint()
    pdf_surface.finish()


######################################################################

if __name__ == "__main__":
    import mygpt

    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, 5))

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

    a = model.retrieve_attention()
    print(a)
    attention = torch.cat([x[:0] for x in a], dim=0)

    tokens = ["bluh", 2, 3, 4, "blih"]
    attention = torch.randn(3, len(tokens), len(tokens)).softmax(dim=-1)

    save_attention_image("attention.pdf", tokens, attention)
