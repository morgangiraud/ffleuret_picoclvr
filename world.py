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


def scene2tensor(xh, yh, scene, size=64):
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


def sequence(nb_steps=10, all_frames=False):
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

        frames.append(scene2tensor(xh, yh, scene))

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
                frames.append(scene2tensor(xh, yh, scene))

        if not all_frames:
            frames.append(scene2tensor(xh, yh, scene))

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


def train_encoder(input, device=torch.device("cpu")):
    class SomeLeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
            self.fc1 = nn.Linear(256, 200)
            self.fc2 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    ######################################################################

    model = SomeLeNet()

    nb_parameters = sum(p.numel() for p in model.parameters())

    print(f"nb_parameters {nb_parameters}")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    train_input, train_targets = train_input.to(device), train_targets.to(device)
    test_input, test_targets = test_input.to(device), test_targets.to(device)

    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    start_time = time.perf_counter()

    for k in range(nb_epochs):
        acc_loss = 0.0

        for input, targets in zip(
            train_input.split(batch_size), train_targets.split(batch_size)
        ):
            output = model(input)
            loss = criterion(output, targets)
            acc_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        nb_test_errors = 0
        for input, targets in zip(
            test_input.split(batch_size), test_targets.split(batch_size)
        ):
            wta = model(input).argmax(1)
            nb_test_errors += (wta != targets).long().sum()
        test_error = nb_test_errors / test_input.size(0)
        duration = time.perf_counter() - start_time

        print(f"loss {k} {duration:.02f}s {acc_loss:.02f} {test_error*100:.02f}%")


######################################################################

if __name__ == "__main__":
    import time

    all_frames = []
    nb = 1000
    start_time = time.perf_counter()
    for n in range(nb):
        frames, actions = sequence(nb_steps=31)
        all_frames += frames
    end_time = time.perf_counter()
    print(f"{nb / (end_time - start_time):.02f} samples per second")

    input = torch.cat(all_frames, 0)

    # x = patchify(input, 8)
    # y = x.reshape(x.size(0), -1)
    # print(f"{x.size()=} {y.size()=}")
    # centroids, t = kmeans(y, 4096)
    # results = centroids[t]
    # results = results.reshape(x.size())
    # results = patchify(results, 8, input.size())

    print(f"{input.size()=} {results.size()=}")

    torchvision.utils.save_image(input[:64], "orig.png", nrow=8)
    torchvision.utils.save_image(results[:64], "qtiz.png", nrow=8)

    # frames, actions = sequence(nb_steps=31, all_frames=True)
    # frames = torch.cat(frames, 0)
    # torchvision.utils.save_image(frames, "seq.png", nrow=8)
