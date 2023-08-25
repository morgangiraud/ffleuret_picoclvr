#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math
import torch, torchvision
import torch.nn.functional as F

name_shapes = ["A", "B", "C", "D", "E", "F"]

name_colors = ["red", "yellow", "blue", "green", "white", "purple"]

######################################################################


class GridFactory:
    def __init__(
        self,
        height=4,
        width=4,
        max_nb_items=4,
        max_nb_transformations=4,
        nb_questions=4,
    ):
        self.height = height
        self.width = width
        self.max_nb_items = max_nb_items
        self.nb_questions = nb_questions

    def generate_scene(self):
        nb_items = torch.randint(self.max_nb_items - 1, (1,)).item() + 2
        col = torch.full((self.height * self.width,), -1)
        shp = torch.full((self.height * self.width,), -1)
        a = torch.randperm(len(name_colors) * len(name_shapes))[:nb_items]
        col[:nb_items] = a % len(name_colors)
        shp[:nb_items] = a // len(name_colors)
        i = torch.randperm(self.height * self.width)
        col = col[i]
        shp = shp[i]
        return col.reshape(self.height, self.width), shp.reshape(
            self.height, self.width
        )

    def random_transformations(self):
        nb_transformations = torch.randint(self.max_nb_transformations + 1, (1,)).item()

    def print_scene(self, scene):
        col, shp = scene

        # for i in range(self.height):
        # for j in range(self.width):
        # if col[i,j] >= 0:
        # print(f"at ({i},{j}) {name_colors[col[i,j]]} {name_shapes[shp[i,j]]}")

        for i in range(self.height):
            for j in range(self.width):
                if col[i, j] >= 0:
                    print(f"{name_colors[col[i,j]][0]}{name_shapes[shp[i,j]]}", end="")
                elif j == 0:
                    print(" +", end="")
                else:
                    print("-+", end="")
                if j < self.width - 1:
                    print("--", end="")
                else:
                    print("")
            if i < self.height - 1:
                for j in range(self.width - 1):
                    print(" |  ", end="")
                print(" |")

    def grid_positions(self, scene):
        col, shp = scene

        properties = []

        for i in range(self.height):
            for j in range(self.width):
                if col[i, j] >= 0:
                    n = f"{name_colors[col[i,j]]} {name_shapes[shp[i,j]]}"
                    properties += [f"a {n} at {i} {j}"]

        return properties

    def all_properties(self, scene):
        col, shp = scene

        properties = []

        for i1 in range(self.height):
            for j1 in range(self.width):
                if col[i1, j1] >= 0:
                    n1 = f"{name_colors[col[i1,j1]]} {name_shapes[shp[i1,j1]]}"
                    properties += [f"there is a {n1}"]
                    if i1 < self.height // 2:
                        properties += [f"a {n1} is in the top half"]
                    if i1 >= self.height // 2:
                        properties += [f"a {n1} is in the bottom half"]
                    if j1 < self.width // 2:
                        properties += [f"a {n1} is in the left half"]
                    if j1 >= self.width // 2:
                        properties += [f"a {n1} is in the right half"]
                    for i2 in range(self.height):
                        for j2 in range(self.width):
                            if col[i2, j2] >= 0:
                                n2 = f"{name_colors[col[i2,j2]]} {name_shapes[shp[i2,j2]]}"
                                if i1 > i2:
                                    properties += [f"a {n1} is below a {n2}"]
                                if i1 < i2:
                                    properties += [f"a {n1} is above a {n2}"]
                                if j1 > j2:
                                    properties += [f"a {n1} is right of a {n2}"]
                                if j1 < j2:
                                    properties += [f"a {n1} is left of a {n2}"]

        return properties

    def generate_scene_and_questions(self):
        while True:
            while True:
                scene = self.generate_scene()
                true = self.all_properties(scene)
                if len(true) >= self.nb_questions:
                    break

            start = self.grid_positions(scene)

            for a in range(10):
                col, shp = scene
                col, shp = col.view(-1), shp.view(-1)
                p = torch.randperm(col.size(0))
                col, shp = col[p], shp[p]
                other_scene = (
                    col.view(self.height, self.width),
                    shp.view(self.height, self.width),
                )
                # other_scene = self.generate_scene()
                false = list(set(self.all_properties(other_scene)) - set(true))
                if len(false) >= self.nb_questions:
                    break

            # print(f"{a=}")

            if a < 10:
                break

        true = [true[k] for k in torch.randperm(len(true))[: self.nb_questions]]
        false = [false[k] for k in torch.randperm(len(false))[: self.nb_questions]]
        true = ["<prop> " + q + " <true>" for q in true]
        false = ["<prop> " + q + " <false>" for q in false]

        union = true + false
        questions = [union[k] for k in torch.randperm(len(union))[: self.nb_questions]]

        result = " ".join(
            ["<obj> " + x for x in self.grid_positions(scene)] + questions
        )

        return scene, result

    def generate_samples(self, nb, progress_bar=None):
        result = []

        r = range(nb)
        if progress_bar is not None:
            r = progress_bar(r)

        for _ in r:
            result.append(self.generate_scene_and_questions()[1])

        return result


######################################################################

if __name__ == "__main__":
    import time

    grid_factory = GridFactory()

    start_time = time.perf_counter()
    samples = grid_factory.generate_samples(10000)
    end_time = time.perf_counter()
    print(f"{len(samples) / (end_time - start_time):.02f} samples per second")

    scene, questions = grid_factory.generate_scene_and_questions()
    grid_factory.print_scene(scene)
    print(questions)

######################################################################
