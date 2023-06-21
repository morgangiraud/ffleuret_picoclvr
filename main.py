#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# torch.backends.cuda.matmul.allow_tf23
# torch.autocast(torch.bfloat16)

import math, sys, argparse, time, tqdm, os

import torch, torchvision
from torch import nn
from torch.nn import functional as F

import mygpt, tensorstack

######################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    device = torch.device("cpu")

######################################################################

parser = argparse.ArgumentParser(
    description="An implementation of GPT with cache.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--task", type=str, default="picoclvr")

parser.add_argument("--log_filename", type=str, default="train.log")

parser.add_argument("--result_dir", type=str, default="results_default")

parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--nb_epochs", type=int, default=None)

parser.add_argument("--batch_size", type=int, default=None)

parser.add_argument("--nb_train_samples", type=int, default=250000)

parser.add_argument("--nb_test_samples", type=int, default=10000)

parser.add_argument("--optim", type=str, default="adam")

parser.add_argument("--learning_rate", type=float, default=1e-4)

parser.add_argument("--learning_rate_schedule", type=str, default="10: 2e-5,30: 4e-6")

parser.add_argument("--dim_model", type=int, default=512)

parser.add_argument("--dim_keys", type=int, default=64)

parser.add_argument("--dim_hidden", type=int, default=2048)

parser.add_argument("--nb_heads", type=int, default=8)

parser.add_argument("--nb_blocks", type=int, default=12)

parser.add_argument("--dropout", type=float, default=0.1)

parser.add_argument("--deterministic_synthesis", action="store_true", default=False)

parser.add_argument("--no_checkpoint", action="store_true", default=False)

parser.add_argument("--overwrite_results", action="store_true", default=False)

parser.add_argument("--checkpoint_name", type=str, default="checkpoint.pth")

##############################
# picoclvr options

parser.add_argument("--picoclvr_nb_colors", type=int, default=5)

parser.add_argument("--picoclvr_height", type=int, default=12)

parser.add_argument("--picoclvr_width", type=int, default=16)

parser.add_argument("--picocvlr_prune_properties", type=str, default="none")

##############################
# Maze options

parser.add_argument("--maze_height", type=int, default=13)

parser.add_argument("--maze_width", type=int, default=21)

parser.add_argument("--maze_nb_walls", type=int, default=15)

##############################
# Snake options

parser.add_argument("--snake_height", type=int, default=6)

parser.add_argument("--snake_width", type=int, default=8)

parser.add_argument("--snake_nb_colors", type=int, default=5)

parser.add_argument("--snake_length", type=int, default=400)

######################################################################

args = parser.parse_args()

assert args.picocvlr_prune_properties in {"none", "train+eval", "eval"}

try:
    os.mkdir(args.result_dir)
except FileExistsError:
    if not args.overwrite_results:
        print(f"result directory {args.result_dir} already exists")
        exit(1)

log_file = open(os.path.join(args.result_dir, args.log_filename), "a")

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################

default_args = {
    "picoclvr": {
        "nb_epochs": 25,
        "batch_size": 25,
    },
    "mnist": {
        "nb_epochs": 25,
        "batch_size": 10,
    },
    "maze": {
        "nb_epochs": 25,
        "batch_size": 25,
    },
    "snake": {
        "nb_epochs": 25,
        "batch_size": 20,
    },
}

if args.task in default_args:
    for k, v in default_args[args.task].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

######################################################################


def log_string(s):
    t = time.strftime("%Y%m%d-%H:%M:%S ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + "\n")
        log_file.flush()

    print(t + s)
    sys.stdout.flush()


for n in vars(args):
    log_string(f"args.{n} {getattr(args, n)}")

######################################################################


def masked_inplace_autoregression(
    model, batch_size, input, ar_mask, forbidden_tokens=None, device=torch.device("cpu")
):
    for input, ar_mask in tqdm.tqdm(
        zip(input.split(batch_size), ar_mask.split(batch_size)),
        dynamic_ncols=True,
        desc="autoregression",
        total=input.size(0) // batch_size,
    ):
        i = (ar_mask.sum(0) > 0).nonzero()
        if i.min() > 0:
            model(
                mygpt.BracketedSequence(input, 0, i.min())
            )  # Needed to initialize the model's cache
        for s in range(i.min(), i.max() + 1):
            output = model(mygpt.BracketedSequence(input, s, 1)).x
            logits = output[:, s]
            if forbidden_tokens is not None:
                logits = logits.masked_fill(forbidden_tokens, float("-inf"))
            if args.deterministic_synthesis:
                t_next = logits.argmax(1)
            else:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                t_next = dist.sample()
            input[:, s] = ar_mask[:, s] * t_next + (1 - ar_mask[:, s]) * input[:, s]


######################################################################


class Task:
    def batches(self, split="train"):
        pass

    def vocabulary_size(self):
        pass

    def produce_results(self, n_epoch, model):
        pass


######################################################################

import picoclvr


class TaskPicoCLVR(Task):
    # Make a tensor from a list of strings
    def tensorize(self, descr):
        token_descr = [s.strip().split(" ") for s in descr]
        l = max([len(s) for s in token_descr])
        token_descr = [s + ["<nul>"] * (l - len(s)) for s in token_descr]
        id_descr = [[self.token2id[u] for u in s] for s in token_descr]
        return torch.tensor(id_descr, device=self.device)

    # Make a list of strings from a tensor
    def detensorize(self, x):
        return [" ".join([self.id2token[t.item()] for t in r]) for r in x]

    # trim all the tensors in the tuple z to remove as much token from
    # left and right in the first tensor. If z is a tuple, all its
    # elements are trimed according to the triming for the first
    def trim(self, z, token="<nul>"):
        n = self.token2id[token]
        if type(z) == tuple:
            x = z[0]
            i = (1 - (F.pad(x, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return tuple([t[:, a:b] for t in z])
        else:
            i = (1 - (F.pad(z, (1, 1), value=n) == n).min(0).values.long()).cumsum(0)
            a, b = (i == 0).nonzero().max(), (i == i.max()).nonzero().min()
            return z[:, a:b]

    ######################
    # Not the cleanest part of the code

    # Extract the last image of each sequence, from the last <img>
    # included, and set to <nul> all the tokens from the beginning of
    # that image to the end
    def excise_last_image(self, input):
        t_img, t_nul = self.token2id["<img>"], self.token2id["<nul>"]
        nb_img_tokens = self.height * self.width + 1

        input = input.clone()
        t = (input == t_img).long()
        tail_masks = (t.cumsum(dim=1) == t.sum(dim=1, keepdim=True)).long()
        i = (t * tail_masks).nonzero(as_tuple=True)
        j = (
            i[0][:, None],
            i[1][:, None] + torch.arange(nb_img_tokens, device=input.device)[None, :],
        )
        images = self.trim(input[j])
        input[j] = t_nul
        loss_masks = 1 - tail_masks
        input, loss_masks = self.trim((input, loss_masks))
        return input, loss_masks, images

    def add_true_image(self, input, images, loss_masks):
        t_nul = self.token2id["<nul>"]
        nb_img_tokens = self.height * self.width + 1
        input = F.pad(input, (0, nb_img_tokens), value=t_nul)
        loss_masks = F.pad(loss_masks, (0, nb_img_tokens), value=0)
        t = (input == t_nul).long()
        i = (t.cumsum(dim=1) == 1).nonzero(as_tuple=True)
        j = (
            i[0][:, None],
            i[1][:, None] + torch.arange(nb_img_tokens, device=input.device)[None, :],
        )
        input[j] = images
        loss_masks[j] = 1
        input, loss_masks = self.trim((input, loss_masks))
        return input, loss_masks

    def add_generated_image(self, input, loss_masks, model):
        t_img, t_nul = self.token2id["<img>"], self.token2id["<nul>"]
        nb_img_tokens = self.height * self.width + 1

        input = F.pad(input, (0, nb_img_tokens), value=t_nul)
        loss_masks = F.pad(loss_masks, (0, nb_img_tokens), value=0)
        t = (input == t_nul).long()
        i = (t.cumsum(dim=1) == 1).nonzero(as_tuple=True)
        input[i] = t_img

        j = (
            i[0][:, None],
            i[1][:, None]
            + 1
            + torch.arange(nb_img_tokens - 1, device=input.device)[None, :],
        )
        ar_masks = input.new_zeros(input.size(), dtype=torch.int64)
        ar_masks[j] = 1
        forbidden_tokens = (
            torch.arange(self.vocabulary_size(), device=input.device) == t_nul
        )
        with torch.autograd.no_grad():
            t = model.training
            model.eval()
            masked_inplace_autoregression(
                model,
                self.batch_size,
                input,
                ar_masks,
                forbidden_tokens,
                device=self.device,
            )
            model.train(t)

        input, loss_masks = self.trim((input, loss_masks))

        return input, loss_masks

    ######################

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_colors=5,
        device=torch.device("cpu"),
        pruner_train=None,
        pruner_eval=None,
    ):
        def generate_descr(nb, cache_suffix, pruner):
            return picoclvr.generate(
                nb,
                height=self.height,
                width=self.width,
                nb_colors=nb_colors,
                pruner=pruner,
            )

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device = device
        self.pruner_train = pruner_train
        self.pruner_eval = pruner_eval

        param = {
            "nb_train_samples": nb_train_samples,
            "nb_test_samples": nb_test_samples,
            "height": height,
            "width": width,
            "nb_colors": nb_colors,
            "batch_size": batch_size,
            "rng_state": list(torch.get_rng_state()),
        }

        log_string(
            f"generating {nb_train_samples+nb_test_samples} samples (can take some time)"
        )
        self.train_descr = generate_descr(
            nb_train_samples, "train", pruner=self.pruner_train
        )
        self.test_descr = generate_descr(nb_test_samples, "test", pruner=None)

        # Build the tokenizer
        tokens = {"<nul>", "<img>"}
        for d in [self.train_descr, self.test_descr]:
            for s in d:
                for t in s.strip().split(" "):
                    tokens.add(t)
        # make this set a sorted list to get the same tensors given
        # the same descr
        tokens = list(tokens)
        tokens.sort()
        self.token2id = dict([(t, n) for n, t in enumerate(tokens)])
        self.id2token = dict([(n, t) for n, t in enumerate(tokens)])

        # Tokenize the train and test sets
        self.train_input = self.tensorize(self.train_descr)
        self.test_input = self.tensorize(self.test_descr)

    def batches(self, split="train"):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=f"epoch-{split}"
        ):
            yield self.trim(batch)

    def vocabulary_size(self):
        return len(self.token2id)

    def compute_missing_properties(self, n_epoch, model, pruner=None):
        acc_nb_requested_properties = []
        acc_nb_missing_properties = []
        acc_nb_results = 0

        for input in tqdm.tqdm(
            self.test_input.split(self.batch_size),
            dynamic_ncols=True,
            desc=f"test-properties",
        ):
            tape, loss_masks, _ = self.excise_last_image(input)
            tape, loss_masks = self.add_generated_image(tape, loss_masks, model)
            result_descr = self.detensorize(tape)
            np = picoclvr.nb_properties(
                result_descr,
                height=self.height,
                width=self.width,
                pruner=pruner,
            )
            nb_requested_properties, _, nb_missing_properties = zip(*np)
            acc_nb_requested_properties += nb_requested_properties
            acc_nb_missing_properties += nb_missing_properties
            acc_nb_results += len(result_descr)

        nb_requested_properties = sum(acc_nb_requested_properties)
        nb_missing_properties = sum(acc_nb_missing_properties)

        prefix = "" if pruner is None else "pruned_"
        log_string(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        log_string(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        log_string(
            f"property_{prefix}miss {n_epoch} {100*nb_missing_properties/nb_requested_properties:.02f}%"
        )

    ######################################################################

    def produce_results(self, n_epoch, model):
        self.compute_missing_properties(n_epoch, model)

        if self.pruner_eval is not None:
            self.compute_missing_properties(n_epoch, model, self.pruner_eval)

        nb_tokens_to_generate = self.height * self.width + 3
        result_descr = []
        nb_per_primer = 8
        primer = []

        for primer_descr in [
            "red above green <sep> green top <sep> blue right of red",
            "there is red <sep> there is yellow <sep> there is blue",
            "red below yellow <sep> yellow below green <sep> green below blue <sep> red right <sep> yellow left <sep> green right <sep> blue left",
            "green bottom <sep> yellow bottom <sep> green left of blue <sep> yellow right of blue <sep> blue top",
        ]:
            primer += [primer_descr] * nb_per_primer

        tape = self.tensorize(primer)
        loss_masks = 1 - (tape == self.token2id["<nul>"]).long()
        tape, loss_masks = self.add_generated_image(tape, loss_masks, model)
        result_descr = self.detensorize(tape)

        np = picoclvr.nb_properties(result_descr, height=self.height, width=self.width)

        acc_nb_requested_properties, _, acc_nb_missing_properties = zip(*np)
        acc_nb_results = len(result_descr)

        nb_requested_properties = sum(acc_nb_requested_properties)
        nb_missing_properties = sum(acc_nb_missing_properties)

        prefix = "demo_"
        log_string(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        log_string(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        log_string(
            f"property_{prefix}miss {n_epoch} {100*nb_missing_properties/nb_requested_properties:.02f}%"
        )

        img = picoclvr.descr2img(result_descr, height=self.height, width=self.width)

        if img.dim() == 5:
            if img.size(1) == 1:
                img = F.pad(img.squeeze(1), pad=(1, 1, 1, 1), value=64)
            else:
                img = torch.cat(
                    [
                        torchvision.utils.make_grid(x, padding=1, pad_value=64)[None]
                        for x in img
                    ],
                    0,
                )

        image_name = os.path.join(args.result_dir, f"picoclvr_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            img / 255.0, image_name, nrow=nb_per_primer, padding=1, pad_value=1.0
        )
        log_string(f"wrote {image_name}")


######################################################################


class TaskMNIST(Task):
    def __init__(self, batch_size, device=torch.device("cpu")):
        self.device = device
        self.batch_size = batch_size

    def batches(self, split="train"):
        assert split in {"train", "test"}
        data_set = torchvision.datasets.MNIST(
            root="./data", train=(split == "train"), download=True
        )
        data_input = data_set.data.view(-1, 28 * 28).long()
        if args.nb_train_samples is not None:
            data_input = data_input[: args.nb_train_samples]
        for batch in tqdm.tqdm(
            data_input.split(self.batch_size), desc=f"epoch-{split}"
        ):
            yield batch

    def vocabulary_size(self):
        return 256

    def produce_results(self, n_epoch, model):
        results = torch.empty(64, 28 * 28, device=self.device, dtype=torch.int64)
        ar_mask = torch.full_like(results, 1)
        masked_inplace_autoregression(
            model, self.batch_size, results, ar_mask, device=self.device
        )
        image_name = os.path.join(args.result_dir, f"mnist_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            1 - results.reshape(-1, 1, 28, 28) / 255.0,
            image_name,
            nrow=16,
            pad_value=0.8,
        )
        log_string(f"wrote {image_name}")


######################################################################

import maze


class TaskMaze(Task):
    def map2seq(self, *m):
        return torch.cat([x.flatten(1) for x in m], 1)

    def seq2map(self, s):
        s = s.reshape(s.size(0), -1, self.height, self.width)
        return (s[:, k] for k in range(s.size(1)))

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_walls,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device = device

        train_mazes, train_paths, _ = maze.create_maze_data(
            nb_train_samples,
            height=height,
            width=width,
            nb_walls=nb_walls,
            progress_bar=lambda x: tqdm.tqdm(x, dynamic_ncols=True, desc=f"data-train"),
        )
        self.train_input = self.map2seq(train_mazes.to(device), train_paths.to(device))

        test_mazes, test_paths, _ = maze.create_maze_data(
            nb_test_samples,
            height=height,
            width=width,
            nb_walls=nb_walls,
            progress_bar=lambda x: tqdm.tqdm(x, dynamic_ncols=True, desc=f"data-test"),
        )
        self.test_input = self.map2seq(test_mazes.to(device), test_paths.to(device))

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def compute_error(self, model, split="train", nb_to_use=-1):
        nb_total, nb_correct = 0, 0
        for input in task.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            masked_inplace_autoregression(
                model, self.batch_size, result, ar_mask, device=self.device
            )
            mazes, paths = self.seq2map(result)
            nb_correct += maze.path_correctness(mazes, paths).long().sum()
            nb_total += mazes.size(0)

        return nb_total, nb_correct

    def produce_results(self, n_epoch, model):
        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            train_nb_total, train_nb_correct = self.compute_error(
                model, "train", nb_to_use=1000
            )
            log_string(
                f"accuracy_train nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
            )

            test_nb_total, test_nb_correct = self.compute_error(
                model, "test", nb_to_use=1000
            )
            log_string(
                f"accuracy_test nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            input = self.test_input[:48]
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            masked_inplace_autoregression(
                model, self.batch_size, result, ar_mask, device=self.device
            )

            mazes, paths = self.seq2map(input)
            _, predicted_paths = self.seq2map(result)

            filename = os.path.join(args.result_dir, f"maze_result_{n_epoch:04d}.png")
            maze.save_image(
                filename,
                mazes=mazes,
                target_paths=paths,
                predicted_paths=predicted_paths,
                path_correct=maze.path_correctness(mazes, predicted_paths),
            )
            log_string(f"wrote {filename}")

            model.train(t)


######################################################################


import snake


class TaskSnake(Task):
    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_colors,
        length,
        prompt_length,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.device = device
        self.prompt_length = prompt_length

        self.train_input, self.train_prior_visits = snake.generate_sequences(
            nb_train_samples,
            height,
            width,
            nb_colors,
            length,
            prompt_length,
            self.device,
        )
        self.test_input, self.test_prior_visits = snake.generate_sequences(
            nb_test_samples,
            height,
            width,
            nb_colors,
            length,
            prompt_length,
            self.device,
        )

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

    def batches(self, split="train", nb_to_use=-1, desc=None):
        assert split in {"train", "test"}
        input = self.train_input if split == "train" else self.test_input
        if nb_to_use > 0:
            input = input[:nb_to_use]
        if desc is None:
            desc = f"epoch-{split}"
        for batch in tqdm.tqdm(
            input.split(self.batch_size), dynamic_ncols=True, desc=desc
        ):
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(self, n_epoch, model):
        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            def compute_nb_correct(input, prior_visits):
                result = input.clone()
                i = torch.arange(result.size(1), device=result.device)[None, :]
                ar_mask = (
                    torch.logical_and(i >= self.prompt_length * 2, i % 2 == 0)
                    .long()
                    .expand_as(result)
                )
                result *= 1 - ar_mask

                # snake.solver(result,ar_mask)

                masked_inplace_autoregression(
                    model, self.batch_size, result, ar_mask, device=self.device
                )

                nb_total = ((prior_visits > 0) * ar_mask).sum()

                nb_correct = (
                    (result == input).long() * (prior_visits > 0) * ar_mask
                ).sum()

                # nb_total = result.size(0)
                # nb_correct = ((result - input).abs().sum(1) == 0).sum()

                return nb_total, nb_correct

            # train_nb_total, train_nb_correct = compute_nb_correct(
            # self.train_input, self.train_prior_visits
            # )

            # log_string(
            # f"accuracy_train nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
            # )

            test_nb_total, test_nb_correct = compute_nb_correct(
                self.test_input[:1000], self.test_prior_visits[:1000]
            )

            log_string(
                f"accuracy_test nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            model.train(t)


######################################################################


def picoclvr_pruner_horizontal_green(p):
    return not ("green" in p and ("left" in p or "right" in p))


picoclvr_pruner_train = (
    picoclvr_pruner_horizontal_green
    if args.picocvlr_prune_properties in {"train+eval"}
    else None
)

picoclvr_pruner_eval = (
    (lambda p: not picoclvr_pruner_horizontal_green(p))
    if args.picocvlr_prune_properties in {"train+eval", "eval"}
    else None
)

######################################################################

if args.task == "picoclvr":
    task = TaskPicoCLVR(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.picoclvr_height,
        width=args.picoclvr_width,
        nb_colors=args.picoclvr_nb_colors,
        device=device,
        pruner_train=picoclvr_pruner_train,
        pruner_eval=picoclvr_pruner_eval,
    )

elif args.task == "mnist":
    task = TaskMNIST(
        batch_size=args.batch_size,
        device=device,
    )

elif args.task == "maze":
    task = TaskMaze(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.maze_height,
        width=args.maze_width,
        nb_walls=args.maze_nb_walls,
        device=device,
    )

elif args.task == "snake":
    task = TaskSnake(
        nb_train_samples=args.nb_train_samples,
        nb_test_samples=args.nb_test_samples,
        batch_size=args.batch_size,
        height=args.snake_height,
        width=args.snake_width,
        nb_colors=args.snake_nb_colors,
        length=args.snake_length,
        prompt_length=args.snake_length // 2,
        device=device,
    )

else:
    raise ValueError(f"Unknown task {args.task}")

######################################################################

log_string(f"device {device}")

vocabulary_size = task.vocabulary_size()

log_string(f"vocabulary_size {vocabulary_size}")

##############################

model = mygpt.MyGPT(
    vocabulary_size=vocabulary_size,
    dim_model=args.dim_model,
    dim_keys=args.dim_keys,
    dim_hidden=args.dim_hidden,
    nb_heads=args.nb_heads,
    nb_blocks=args.nb_blocks,
    causal=True,
    dropout=args.dropout,
)

model.to(device)

nb_parameters = sum(p.numel() for p in model.parameters())
log_string(f"nb_parameters {nb_parameters} ({int(nb_parameters/1e6)}M)")

######################################################################

nb_epochs_finished = 0

if args.no_checkpoint:
    log_string(f"not trying to load checkpoint.")

else:
    try:
        checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_name)
        nb_epochs_finished = checkpoint["nb_epochs_finished"]
        model.load_state_dict(checkpoint["model_state"])
        torch.set_rng_state(checkpoint["rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])

        log_string(f"checkpoint loaded with {nb_epochs_finished} epochs finished.")

    except FileNotFoundError:
        log_string("starting from scratch.")

    except:
        log_string("error when loading the checkpoint.")
        exit(1)

######################################################################

nb_epochs = args.nb_epochs if args.nb_epochs > 0 else nb_epochs_default

token_count = 0
for input in task.batches(split="train"):
    token_count += F.one_hot(input, num_classes=task.vocabulary_size()).sum((0, 1))
token_probas = token_count / token_count.sum()
entropy = -torch.xlogy(token_probas, token_probas).sum()
train_set_perplexity = math.exp(entropy)

##############################

if args.learning_rate_schedule == "cos":
    learning_rate_schedule = {}
    for n_epoch in range(args.nb_epochs):
        u = n_epoch / args.nb_epochs * math.pi
        learning_rate_schedule[n_epoch] = args.learning_rate * 0.5 * (1 + math.cos(u))
else:
    u = {
        int(k): float(v)
        for k, v in [
            tuple(x.split(":")) for x in args.learning_rate_schedule.split(",")
        ]
    }

    learning_rate_schedule = {}
    learning_rate = args.learning_rate
    for n_epoch in range(args.nb_epochs):
        if n_epoch in u:
            learning_rate = u[n_epoch]
        learning_rate_schedule[n_epoch] = learning_rate

log_string(f"learning_rate_schedule {learning_rate_schedule}")

##############################

nb_samples_seen = 0

if nb_epochs_finished >= nb_epochs:
    task.produce_results(nb_epochs_finished, model)

for n_epoch in range(nb_epochs_finished, nb_epochs):
    learning_rate = learning_rate_schedule[n_epoch]

    log_string(f"learning_rate {learning_rate}")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {args.optim}.")

    model.train()

    nb_train_samples, acc_train_loss = 0, 0.0

    for input in task.batches(split="train"):
        input = input.to(device)
        output = model(mygpt.BracketedSequence(input)).x
        loss = F.cross_entropy(output.transpose(1, 2), input)
        acc_train_loss += loss.item() * input.size(0)
        nb_train_samples += input.size(0)
        nb_samples_seen += input.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.autograd.no_grad():
        model.eval()

        nb_test_samples, acc_test_loss = 0, 0.0

        for input in task.batches(split="test"):
            input = input.to(device)

            # input, loss_masks, true_images = task.excise_last_image(input)
            # input, loss_masks = task.add_true_image(input, true_images, loss_masks)

            output = model(mygpt.BracketedSequence(input)).x
            loss = F.cross_entropy(output.transpose(1, 2), input)
            acc_test_loss += loss.item() * input.size(0)
            nb_test_samples += input.size(0)

        train_perplexity = math.exp(min(100, acc_train_loss / nb_train_samples))
        test_perplexity = math.exp(min(100, acc_test_loss / nb_test_samples))

        log_string(
            f"perplexity {n_epoch} train_set {train_set_perplexity} train_prediction {train_perplexity} test_prediction {test_perplexity}"
        )

        task.produce_results(n_epoch, model)

    checkpoint = {
        "nb_epochs_finished": n_epoch + 1,
        "model_state": model.state_dict(),
        "rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    checkpoint_name = os.path.join(args.result_dir, args.checkpoint_name)
    torch.save(checkpoint, checkpoint_name)
    log_string(f"saved checkpoint {checkpoint_name}")

######################################################################
