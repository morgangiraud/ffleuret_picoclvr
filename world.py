#!/usr/bin/env python

import math, sys

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


def scene2tensor(xh, yh, scene, size):
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
        ctx.fill()

    hs = size * 0.1
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
        (0.60, 0.60, 1.00),
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


def generate_sequence(nb_steps=10, all_frames=False, size=64):
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
        frames = []

        scene = random_scene()
        xh, yh = tuple(x.item() for x in torch.rand(2))

        frames.append(scene2tensor(xh, yh, scene, size=size))

        actions = torch.randint(len(effects), (nb_steps,))
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

            if all_frames:
                frames.append(scene2tensor(xh, yh, scene, size=size))

        if not all_frames:
            frames.append(scene2tensor(xh, yh, scene, size=size))

        if change:
            break

    return frames, actions


######################################################################


# ||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2<x_i, c_j>
def sq2matrix(x, c):
    nx = x.pow(2).sum(1)
    nc = c.pow(2).sum(1)
    return nx[:, None] + nc[None, :] - 2 * x @ c.t()


def update_centroids(x, c, nb_min=1):
    _, b = sq2matrix(x, c).min(1)
    b.squeeze_()
    nb_resets = 0

    for k in range(0, c.size(0)):
        i = b.eq(k).nonzero(as_tuple=False).squeeze()
        if i.numel() >= nb_min:
            c[k] = x.index_select(0, i).mean(0)
        else:
            n = torch.randint(x.size(0), (1,))
            nb_resets += 1
            c[k] = x[n]

    return c, b, nb_resets


def kmeans(x, nb_centroids, nb_min=1):
    if x.size(0) < nb_centroids * nb_min:
        print("Not enough points!")
        exit(1)

    c = x[torch.randperm(x.size(0))[:nb_centroids]]
    t = torch.full((x.size(0),), -1)
    n = 0

    while True:
        c, u, nb_resets = update_centroids(x, c, nb_min)
        n = n + 1
        nb_changes = (u - t).sign().abs().sum() + nb_resets
        t = u
        if nb_changes == 0:
            break

    return c, t


######################################################################


def patchify(x, factor, invert_size=None):
    if invert_size is None:
        return (
            x.reshape(
                x.size(0),  # 0
                x.size(1),  # 1
                factor,  # 2
                x.size(2) // factor,  # 3
                factor,  # 4
                x.size(3) // factor,  # 5
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, x.size(1), x.size(2) // factor, x.size(3) // factor)
        )
    else:
        return (
            x.reshape(
                invert_size[0],  # 0
                factor,  # 1
                factor,  # 2
                invert_size[1],  # 3
                invert_size[2] // factor,  # 4
                invert_size[3] // factor,  # 5
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(invert_size)
        )


class Normalizer(nn.Module):
    def __init__(self, mu, std):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.log_var = nn.Parameter(2*torch.log(std))

    def forward(self, x):
        return (x-self.mu)/torch.exp(self.log_var/2.0)

class SignSTE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # torch.sign() takes three values
        s = (x >= 0).float() * 2 - 1
        if self.training:
            u = torch.tanh(x)
            return s + u - u.detach()
        else:
            return s


def train_encoder(
    train_input,
    dim_hidden=64,
    block_size=16,
    nb_bits_per_block=10,
    lr_start=1e-3, lr_end=1e-5,
    nb_epochs=50,
    batch_size=25,
    device=torch.device("cpu"),
):
    mu, std = train_input.mean(), train_input.std()

    encoder = nn.Sequential(
        Normalizer(mu, std),
        nn.Conv2d(3, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(
            dim_hidden,
            nb_bits_per_block,
            kernel_size=block_size,
            stride=block_size,
            padding=0,
        ),
        SignSTE(),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(
            nb_bits_per_block,
            dim_hidden,
            kernel_size=block_size,
            stride=block_size,
            padding=0,
        ),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, dim_hidden, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.Conv2d(dim_hidden, 3, kernel_size=5, stride=1, padding=2),
    )

    model = nn.Sequential(encoder, decoder)

    nb_parameters = sum(p.numel() for p in model.parameters())

    print(f"nb_parameters {nb_parameters}")

    model.to(device)

    for k in range(nb_epochs):
        lr=math.exp(math.log(lr_start) + math.log(lr_end/lr_start)/(nb_epochs-1)*k)
        print(f"lr {lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        acc_loss, nb_samples = 0.0, 0

        for input in train_input.split(batch_size):
            output = model(input)
            loss = F.mse_loss(output, input)
            acc_loss += loss.item() * input.size(0)
            nb_samples += input.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"loss {k} {acc_loss/nb_samples}")
        sys.stdout.flush()

    return encoder, decoder


######################################################################

if __name__ == "__main__":
    import time

    all_frames = []
    nb = 25000
    start_time = time.perf_counter()
    for n in range(nb):
        frames, actions = generate_sequence(nb_steps=31)
        all_frames += frames
    end_time = time.perf_counter()
    print(f"{nb / (end_time - start_time):.02f} samples per second")

    input = torch.cat(all_frames, 0)
    encoder, decoder = train_encoder(input)

    # x = patchify(input, 8)
    # y = x.reshape(x.size(0), -1)
    # print(f"{x.size()=} {y.size()=}")
    # centroids, t = kmeans(y, 4096)
    # results = centroids[t]
    # results = results.reshape(x.size())
    # results = patchify(results, 8, input.size())

    z = encoder(input)
    results = decoder(z)

    print(f"{input.size()=} {z.size()=} {results.size()=}")

    torchvision.utils.save_image(input[:64], "orig.png", nrow=8)

    torchvision.utils.save_image(results[:64], "qtiz.png", nrow=8)

    # frames, actions = generate_sequence(nb_steps=31, all_frames=True)
    # frames = torch.cat(frames, 0)
    # torchvision.utils.save_image(frames, "seq.png", nrow=8)
