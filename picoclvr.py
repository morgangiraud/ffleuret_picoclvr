#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch, torchvision
import torch.nn.functional as F

colors = [
    [255, 255, 255],
    [255, 0, 0],
    [0, 128, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 0, 0],
    [128, 0, 0],
    [139, 0, 0],
    [165, 42, 42],
    [178, 34, 34],
    [220, 20, 60],
    [255, 99, 71],
    [255, 127, 80],
    [205, 92, 92],
    [240, 128, 128],
    [233, 150, 122],
    [250, 128, 114],
    [255, 160, 122],
    [255, 69, 0],
    [255, 140, 0],
    [255, 165, 0],
    [255, 215, 0],
    [184, 134, 11],
    [218, 165, 32],
    [238, 232, 170],
    [189, 183, 107],
    [240, 230, 140],
    [128, 128, 0],
    [154, 205, 50],
    [85, 107, 47],
    [107, 142, 35],
    [124, 252, 0],
    [127, 255, 0],
    [173, 255, 47],
    [0, 100, 0],
    [34, 139, 34],
    [0, 255, 0],
    [50, 205, 50],
    [144, 238, 144],
    [152, 251, 152],
    [143, 188, 143],
    [0, 250, 154],
    [0, 255, 127],
    [46, 139, 87],
    [102, 205, 170],
    [60, 179, 113],
    [32, 178, 170],
    [47, 79, 79],
    [0, 128, 128],
    [0, 139, 139],
    [0, 255, 255],
    [0, 255, 255],
    [224, 255, 255],
    [0, 206, 209],
    [64, 224, 208],
    [72, 209, 204],
    [175, 238, 238],
    [127, 255, 212],
    [176, 224, 230],
    [95, 158, 160],
    [70, 130, 180],
    [100, 149, 237],
    [0, 191, 255],
    [30, 144, 255],
    [173, 216, 230],
    [135, 206, 235],
    [135, 206, 250],
    [25, 25, 112],
    [0, 0, 128],
    [0, 0, 139],
    [0, 0, 205],
    [65, 105, 225],
    [138, 43, 226],
    [75, 0, 130],
    [72, 61, 139],
    [106, 90, 205],
    [123, 104, 238],
    [147, 112, 219],
    [139, 0, 139],
    [148, 0, 211],
    [153, 50, 204],
    [186, 85, 211],
    [128, 0, 128],
    [216, 191, 216],
    [221, 160, 221],
    [238, 130, 238],
    [255, 0, 255],
    [218, 112, 214],
    [199, 21, 133],
    [219, 112, 147],
    [255, 20, 147],
    [255, 105, 180],
    [255, 182, 193],
    [255, 192, 203],
    [250, 235, 215],
    [245, 245, 220],
    [255, 228, 196],
    [255, 235, 205],
    [245, 222, 179],
    [255, 248, 220],
    [255, 250, 205],
    [250, 250, 210],
    [255, 255, 224],
    [139, 69, 19],
    [160, 82, 45],
    [210, 105, 30],
    [205, 133, 63],
    [244, 164, 96],
    [222, 184, 135],
    [210, 180, 140],
    [188, 143, 143],
    [255, 228, 181],
    [255, 222, 173],
    [255, 218, 185],
    [255, 228, 225],
    [255, 240, 245],
    [250, 240, 230],
    [253, 245, 230],
    [255, 239, 213],
    [255, 245, 238],
    [245, 255, 250],
    [112, 128, 144],
    [119, 136, 153],
    [176, 196, 222],
    [230, 230, 250],
    [255, 250, 240],
    [240, 248, 255],
    [248, 248, 255],
    [240, 255, 240],
    [255, 255, 240],
    [240, 255, 255],
    [255, 250, 250],
    [192, 192, 192],
    [220, 220, 220],
    [245, 245, 245],
]

color_names = [
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "black",
    "maroon",
    "dark_red",
    "brown",
    "firebrick",
    "crimson",
    "tomato",
    "coral",
    "indian_red",
    "light_coral",
    "dark_salmon",
    "salmon",
    "light_salmon",
    "orange_red",
    "dark_orange",
    "orange",
    "gold",
    "dark_golden_rod",
    "golden_rod",
    "pale_golden_rod",
    "dark_khaki",
    "khaki",
    "olive",
    "yellow_green",
    "dark_olive_green",
    "olive_drab",
    "lawn_green",
    "chartreuse",
    "green_yellow",
    "dark_green",
    "forest_green",
    "lime",
    "lime_green",
    "light_green",
    "pale_green",
    "dark_sea_green",
    "medium_spring_green",
    "spring_green",
    "sea_green",
    "medium_aqua_marine",
    "medium_sea_green",
    "light_sea_green",
    "dark_slate_gray",
    "teal",
    "dark_cyan",
    "aqua",
    "cyan",
    "light_cyan",
    "dark_turquoise",
    "turquoise",
    "medium_turquoise",
    "pale_turquoise",
    "aqua_marine",
    "powder_blue",
    "cadet_blue",
    "steel_blue",
    "corn_flower_blue",
    "deep_sky_blue",
    "dodger_blue",
    "light_blue",
    "sky_blue",
    "light_sky_blue",
    "midnight_blue",
    "navy",
    "dark_blue",
    "medium_blue",
    "royal_blue",
    "blue_violet",
    "indigo",
    "dark_slate_blue",
    "slate_blue",
    "medium_slate_blue",
    "medium_purple",
    "dark_magenta",
    "dark_violet",
    "dark_orchid",
    "medium_orchid",
    "purple",
    "thistle",
    "plum",
    "violet",
    "magenta",
    "orchid",
    "medium_violet_red",
    "pale_violet_red",
    "deep_pink",
    "hot_pink",
    "light_pink",
    "pink",
    "antique_white",
    "beige",
    "bisque",
    "blanched_almond",
    "wheat",
    "corn_silk",
    "lemon_chiffon",
    "light_golden_rod_yellow",
    "light_yellow",
    "saddle_brown",
    "sienna",
    "chocolate",
    "peru",
    "sandy_brown",
    "burly_wood",
    "tan",
    "rosy_brown",
    "moccasin",
    "navajo_white",
    "peach_puff",
    "misty_rose",
    "lavender_blush",
    "linen",
    "old_lace",
    "papaya_whip",
    "sea_shell",
    "mint_cream",
    "slate_gray",
    "light_slate_gray",
    "light_steel_blue",
    "lavender",
    "floral_white",
    "alice_blue",
    "ghost_white",
    "honeydew",
    "ivory",
    "azure",
    "snow",
    "silver",
    "gainsboro",
    "white_smoke",
]

color_id = dict([(n, k) for k, n in enumerate(color_names)])
color_tokens = dict([(n, c) for n, c in zip(color_names, colors)])

######################################################################


def all_properties(height, width, nb_squares, square_i, square_j, square_c):
    s = []

    for r, c_r in [(k, color_names[square_c[k]]) for k in range(nb_squares)]:
        s += [f"there is {c_r}"]

        if square_i[r] >= height - height // 3:
            s += [f"{c_r} bottom"]
        if square_i[r] < height // 3:
            s += [f"{c_r} top"]
        if square_j[r] >= width - width // 3:
            s += [f"{c_r} right"]
        if square_j[r] < width // 3:
            s += [f"{c_r} left"]

        for t, c_t in [(k, color_names[square_c[k]]) for k in range(nb_squares)]:
            if square_i[r] > square_i[t]:
                s += [f"{c_r} below {c_t}"]
            if square_i[r] < square_i[t]:
                s += [f"{c_r} above {c_t}"]
            if square_j[r] > square_j[t]:
                s += [f"{c_r} right of {c_t}"]
            if square_j[r] < square_j[t]:
                s += [f"{c_r} left of {c_t}"]

    return s


######################################################################

# Generates sequences


def generate(
    nb,
    height,
    width,
    max_nb_squares=5,
    max_nb_properties=10,
    nb_colors=5,
    pruner=None,
):

    assert nb_colors >= max_nb_squares and nb_colors <= len(color_tokens) - 1

    descr = []

    for n in range(nb):

        nb_squares = torch.randint(max_nb_squares, (1,)) + 1
        square_position = torch.randperm(height * width)[:nb_squares]

        # color 0 is white and reserved for the background
        square_c = torch.randperm(nb_colors)[:nb_squares] + 1
        square_i = square_position.div(width, rounding_mode="floor")
        square_j = square_position % width

        img = [0] * height * width
        for k in range(nb_squares):
            img[square_position[k]] = square_c[k]

        # generates all the true properties

        s = all_properties(height, width, nb_squares, square_i, square_j, square_c)

        if pruner is not None:
            s = list(filter(pruner, s))

        # picks at most max_nb_properties at random

        nb_properties = torch.randint(max_nb_properties, (1,)) + 1
        s = (
            " <sep> ".join([s[k] for k in torch.randperm(len(s))[:nb_properties]])
            + " <img> "
            + " ".join([f"{color_names[n]}" for n in img])
        )

        descr += [s]

    return descr


######################################################################

# Extracts the image after <img> in descr as a 1x3xHxW tensor


def descr2img(descr, n, height, width):

    if type(descr) == list:
        return torch.cat([descr2img(d, n, height, width) for d in descr], 0)

    if type(n) == list:
        return torch.cat([descr2img(descr, k, height, width) for k in n], 0).unsqueeze(
            0
        )

    def token2color(t):
        try:
            return color_tokens[t]
        except KeyError:
            return [128, 128, 128]

    d = descr.split("<img>")
    d = d[n + 1] if len(d) > n + 1 else ""
    d = d.strip().split(" ")[: height * width]
    d = d + ["<unk>"] * (height * width - len(d))
    d = [token2color(t) for t in d]
    img = torch.tensor(d).permute(1, 0)
    img = img.reshape(1, 3, height, width)

    return img


######################################################################

# Returns all the properties of the image after <img> in descr


def descr2properties(descr, height, width):

    if type(descr) == list:
        return [descr2properties(d, height, width) for d in descr]

    d = descr.split("<img>")
    d = d[-1] if len(d) > 1 else ""
    d = d.strip().split(" ")[: height * width]
    if len(d) != height * width:
        return []

    seen = {}
    for k, x in enumerate(d):
        if x != color_names[0]:
            if x in color_tokens:
                if x in seen:
                    return []
            else:
                return []
            seen[x] = (color_id[x], k // width, k % width)

    square_infos = tuple(zip(*seen.values()))

    if square_infos:
        square_c = torch.tensor(square_infos[0])
        square_i = torch.tensor(square_infos[1])
        square_j = torch.tensor(square_infos[2])
    else:
        square_c = torch.tensor([])
        square_i = torch.tensor([])
        square_j = torch.tensor([])

    s = all_properties(height, width, len(seen), square_i, square_j, square_c)

    return s


######################################################################

# Returns a triplet composed of (1) the total number of properties
# before <img> in descr, (2) the total number of properties the image
# after <img> verifies, and (3) the number of properties in (1) not in
# (2)


def nb_properties(descr, height, width, pruner=None):

    if type(descr) == list:
        return [nb_properties(d, height, width, pruner) for d in descr]

    d = descr.split("<img>", 1)
    if len(d) == 0:
        return 0
    d = d[0].strip().split("<sep>")
    d = [x.strip() for x in d]

    all_properties = set(descr2properties(descr, height, width))

    if pruner is None:
        requested_properties = set(d)
    else:
        requested_properties = set(filter(pruner, d))

    missing_properties = requested_properties - all_properties

    return (len(requested_properties), len(all_properties), len(missing_properties))


######################################################################

if __name__ == "__main__":
    for n in range(16):
        descr = generate(nb=1, height=12, width=16)

        print(nb_properties(descr, height=12, width=16))

        with open(f"picoclvr_example_{n:02d}.txt", "w") as f:
            for d in descr:
                f.write(f"{d}\n\n")

        img = descr2img(descr, n=0, height=12, width=16)
        if img.size(0) == 1:
            img = F.pad(img, (1, 1, 1, 1), value=64)

        torchvision.utils.save_image(
            img / 255.0,
            f"picoclvr_example_{n:02d}.png",
            padding=1,
            nrow=4,
            pad_value=0.8,
        )

    import time

    start_time = time.perf_counter()
    descr = generate(nb=1000, height=12, width=16)
    end_time = time.perf_counter()
    print(f"{len(descr) / (end_time - start_time):.02f} samples per second")

######################################################################
