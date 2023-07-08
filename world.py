#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F
import cairo


class Box:
    def __init__(self, x, y, w, h, r, g, b):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.r = r
        self.g = g
        self.b = b

    def collision(self, scene):
        for c in scene:
            if (
                self is not c
                and max(self.x, c.x) <= min(self.x + self.w, c.x + c.w)
                and max(self.y, c.y) <= min(self.y + self.h, c.y + c.h)
            ):
                return True
        return False


def scene2tensor(xh, yh, scene, size=512):
    width, height = size, size
    pixel_map = torch.ByteTensor(width, height, 4).fill_(255)
    data = pixel_map.numpy()
    surface = cairo.ImageSurface.create_for_data(
        data, cairo.FORMAT_ARGB32, width, height
    )

    ctx = cairo.Context(surface)
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

    for b in scene:
        ctx.move_to(b.x * size, b.y * size)
        ctx.rel_line_to(b.w * size, 0)
        ctx.rel_line_to(0, b.h * size)
        ctx.rel_line_to(-b.w * size, 0)
        ctx.close_path()
        ctx.set_source_rgba(b.r, b.g, b.b, 1.0)
        ctx.fill_preserve()
        ctx.set_source_rgba(0, 0, 0, 1.0)
        ctx.stroke()

    hs = size * 0.05
    ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
    ctx.move_to(xh * size - hs / 2, yh * size - hs / 2)
    ctx.rel_line_to(hs, 0)
    ctx.rel_line_to(0, hs)
    ctx.rel_line_to(-hs, 0)
    ctx.close_path()
    ctx.fill()

    return pixel_map[None, :, :, :3].flip(-1).permute(0, 3, 1, 2).float() / 255


def random_scene():
    scene = []
    colors = [
        (1.00, 0.00, 0.00),
        (0.00, 1.00, 0.00),
        (0.00, 0.00, 1.00),
        (1.00, 1.00, 0.00),
        (0.75, 0.75, 0.75),
    ]

    for k in range(10):
        wh = torch.rand(2) * 0.2 + 0.2
        xy = torch.rand(2) * (1 - wh)
        c = colors[torch.randint(len(colors), (1,))]
        b = Box(
            xy[0].item(), xy[1].item(), wh[0].item(), wh[1].item(), c[0], c[1], c[2]
        )
        if not b.collision(scene):
            scene.append(b)

    return scene


def sequence(length=10):
    delta = 0.1
    effects = [
        (False, 0, 0),
        (False, delta, 0),
        (False, 0, delta),
        (False, -delta, 0),
        (False, 0, -delta),
        (True, delta, 0),
        (True, 0, delta),
        (True, -delta, 0),
        (True, 0, -delta),
    ]

    while True:
        scene = random_scene()
        xh, yh = tuple(x.item() for x in torch.rand(2))

        frame_start = scene2tensor(xh, yh, scene)

        actions = torch.randint(len(effects), (length,))
        change = False

        for a in actions:
            g, dx, dy = effects[a]
            if g:
                for b in scene:
                    if b.x <= xh and b.x + b.w >= xh and b.y <= yh and b.y + b.h >= yh:
                        x, y = b.x, b.y
                        b.x += dx
                        b.y += dy
                        if (
                            b.x < 0
                            or b.y < 0
                            or b.x + b.w > 1
                            or b.y + b.h > 1
                            or b.collision(scene)
                        ):
                            b.x, b.y = x, y
                        else:
                            xh += dx
                            yh += dy
                            change = True
            else:
                x, y = xh, yh
                xh += dx
                yh += dy
                if xh < 0 or xh > 1 or yh < 0 or yh > 1:
                    xh, yh = x, y

        frame_end = scene2tensor(xh, yh, scene)
        if change:
            break

    return frame_start, frame_end, actions


if __name__ == "__main__":
    frame_start, frame_end, actions = sequence()
    torchvision.utils.save_image(frame_start, "world_start.png")
    torchvision.utils.save_image(frame_end, "world_end.png")
