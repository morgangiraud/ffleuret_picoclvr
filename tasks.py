#!/usr/bin/env python

import math, os, tqdm

import torch, torchvision

from torch import nn
from torch.nn import functional as F

######################################################################


def masked_inplace_autoregression(
    model,
    batch_size,
    input,
    ar_mask,
    deterministic_synthesis,
    forbidden_tokens=None,
    progress_bar_desc="autoregression",
    device=torch.device("cpu"),
):
    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            total=input.size(0) // batch_size,
        )

    for input, ar_mask in batches:
        model.masked_inplace_autoregression(
            input, ar_mask, forbidden_tokens, deterministic_synthesis
        )


class Task:
    def batches(self, split="train"):
        pass

    def vocabulary_size(self):
        pass

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        pass


######################################################################

import picoclvr


class PicoCLVR(Task):
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

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        height,
        width,
        nb_colors=5,
        logger=None,
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

        if logger is not None:
            logger(
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
        self.t_img, self.t_nul = self.token2id["<img>"], self.token2id["<nul>"]

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

    def compute_missing_properties(
        self, n_epoch, model, logger, deterministic_synthesis, pruner=None
    ):
        acc_nb_requested_properties = []
        acc_nb_missing_properties = []
        acc_nb_results = 0

        for input in tqdm.tqdm(
            self.test_input.split(self.batch_size),
            dynamic_ncols=True,
            desc=f"test-properties",
        ):
            result = input.clone()
            ar_mask = (result == self.t_img).long().cumsum(dim=1).clamp(max=1)
            result = (1 - ar_mask) * result + ar_mask * self.t_nul
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            result_descr = self.detensorize(result)
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
        logger(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        logger(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        logger(
            f"property_{prefix}miss {n_epoch} {100*nb_missing_properties/nb_requested_properties:.02f}%"
        )

    ######################################################################

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        self.compute_missing_properties(n_epoch, model, logger, deterministic_synthesis)

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
            primer += [primer_descr + " <img>"] * nb_per_primer

        result = self.tensorize(primer)
        fill = result.new_full(
            result.size()[:-1] + (self.height * self.width + 1,), self.t_nul
        )
        result = torch.cat((result, fill), 1)
        ar_mask = (result == self.t_nul).long()
        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )
        result_descr = self.detensorize(result)

        np = picoclvr.nb_properties(result_descr, height=self.height, width=self.width)

        acc_nb_requested_properties, _, acc_nb_missing_properties = zip(*np)
        acc_nb_results = len(result_descr)

        nb_requested_properties = sum(acc_nb_requested_properties)
        nb_missing_properties = sum(acc_nb_missing_properties)

        prefix = "demo_"
        logger(f"nb_{prefix}samples {n_epoch} {acc_nb_results}")
        logger(
            f"property_{prefix}nb {n_epoch} requested {sum(acc_nb_requested_properties)} missing {sum(acc_nb_missing_properties)}"
        )
        logger(
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

        image_name = os.path.join(result_dir, f"picoclvr_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            img / 255.0, image_name, nrow=nb_per_primer, padding=1, pad_value=0.0
        )
        logger(f"wrote {image_name}")


######################################################################


class MNIST(Task):
    def __init__(
        self, nb_train_samples, nb_test_samples, batch_size, device=torch.device("cpu")
    ):
        self.nb_train_samples = (nb_train_samples,)
        self.nb_test_samples = (nb_test_samples,)
        self.batch_size = batch_size
        self.device = device
        data_set = torchvision.datasets.MNIST(root="./data", train=True, download=True)
        self.train_input = data_set.data[:nb_train_samples].view(-1, 28 * 28).long()
        data_set = torchvision.datasets.MNIST(root="./data", train=False, download=True)
        self.test_input = data_set.data[:nb_test_samples].view(-1, 28 * 28).long()

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
        return 256

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        results = torch.empty(64, 28 * 28, device=self.device, dtype=torch.int64)
        ar_mask = torch.full_like(results, 1)
        masked_inplace_autoregression(
            model,
            self.batch_size,
            results,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )
        image_name = os.path.join(result_dir, f"mnist_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            1 - results.reshape(-1, 1, 28, 28) / 255.0,
            image_name,
            nrow=16,
            pad_value=0.8,
        )
        logger(f"wrote {image_name}")


######################################################################

import maze


class Maze(Task):
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

    def compute_error(
        self, model, split="train", nb_to_use=-1, deterministic_synthesis=False
    ):
        nb_total, nb_correct = 0, 0
        count = torch.zeros(
            self.width * self.height,
            self.width * self.height,
            device=self.device,
            dtype=torch.int64,
        )

        for input in self.batches(split, nb_to_use):
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )
            mazes, paths = self.seq2map(result)
            path_correctness = maze.path_correctness(mazes, paths)
            nb_correct += path_correctness.long().sum()
            nb_total += mazes.size(0)

            optimal_path_lengths = (
                (input[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            predicted_path_lengths = (
                (result[:, self.height * self.width :] == maze.v_path).long().sum(1)
            )
            optimal_path_lengths = optimal_path_lengths[path_correctness]
            predicted_path_lengths = predicted_path_lengths[path_correctness]
            count[optimal_path_lengths, predicted_path_lengths] += 1

        if count.max() == 0:
            count = None
        else:
            count = count[
                : count.sum(1).nonzero().max() + 1, : count.sum(0).nonzero().max() + 1
            ]

        return nb_total, nb_correct, count

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            train_nb_total, train_nb_correct, count = self.compute_error(
                model,
                "train",
                nb_to_use=1000,
                deterministic_synthesis=deterministic_synthesis,
            )
            logger(
                f"accuracy_train {n_epoch} nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
            )

            test_nb_total, test_nb_correct, count = self.compute_error(
                model,
                "test",
                nb_to_use=1000,
                deterministic_synthesis=deterministic_synthesis,
            )
            logger(
                f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            if count is not None:
                proportion_optimal = count.diagonal().sum().float() / count.sum()
                logger(f"proportion_optimal_test {proportion_optimal*100:.02f}%")
                with open(
                    os.path.join(result_dir, f"maze_result_{n_epoch:04d}.txt"), "w"
                ) as f:
                    for i in range(count.size(0)):
                        for j in range(count.size(1)):
                            eol = " " if j < count.size(1) - 1 else "\n"
                            f.write(f"{count[i,j]}{eol}")

            input = self.test_input[:48]
            result = input.clone()
            ar_mask = result.new_zeros(result.size())
            ar_mask[:, self.height * self.width :] = 1
            result *= 1 - ar_mask
            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            mazes, paths = self.seq2map(input)
            _, predicted_paths = self.seq2map(result)

            filename = os.path.join(result_dir, f"maze_result_{n_epoch:04d}.png")
            maze.save_image(
                filename,
                mazes=mazes,
                target_paths=paths,
                predicted_paths=predicted_paths,
                path_correct=maze.path_correctness(mazes, predicted_paths),
                path_optimal=maze.path_optimality(paths, predicted_paths),
            )
            logger(f"wrote {filename}")

            model.train(t)


######################################################################


import snake


class Snake(Task):
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

        self.train_input, self.train_prior_visits, _, _ = snake.generate_sequences(
            nb_train_samples,
            height,
            width,
            nb_colors,
            length,
            prompt_length,
            self.device,
        )
        self.test_input, self.test_prior_visits, _, _ = snake.generate_sequences(
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

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
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
                    model,
                    self.batch_size,
                    result,
                    ar_mask,
                    deterministic_synthesis,
                    device=self.device,
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

            # logger(
            # f"accuracy_train nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
            # )

            test_nb_total, test_nb_correct = compute_nb_correct(
                self.test_input[:1000], self.test_prior_visits[:1000]
            )

            logger(
                f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            model.train(t)


######################################################################


import stack


class Stack(Task):
    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        logger,
        nb_steps,
        nb_stacks,
        nb_digits,
        fraction_values_for_train=None,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_stacks = nb_stacks
        self.nb_digits = nb_digits
        self.device = device

        if fraction_values_for_train is None:
            values_for_train = None
            values_for_test = None
        else:
            all = torch.randperm(10**nb_digits)
            nb_for_train = int(all.size(0) * fraction_values_for_train)
            values_for_train = all[:nb_for_train]
            values_for_test = all[nb_for_train:]

        self.train_input, self.train_stack_counts = stack.generate_sequences(
            nb_train_samples,
            nb_steps,
            nb_stacks,
            nb_digits,
            values_for_train,
            self.device,
        )

        self.test_input, self.test_stack_counts = stack.generate_sequences(
            nb_test_samples,
            nb_steps,
            nb_stacks,
            nb_digits,
            values_for_test,
            self.device,
        )

        i = torch.logical_and(self.test_input % 2 == 1, self.test_input < 2 * nb_stacks)
        counts = self.test_stack_counts.flatten()[i.flatten()]
        counts = F.one_hot(counts).sum(0)
        logger(f"test_pop_stack_counts {counts}")

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

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            def compute_nb_correct(input):
                result = input.clone()
                stack.remove_popped_values(result, self.nb_stacks, self.nb_digits)
                ar_mask = (result != input).long()
                masked_inplace_autoregression(
                    model,
                    self.batch_size,
                    result,
                    ar_mask,
                    deterministic_synthesis,
                    device=self.device,
                )

                errors = ((result != input).long() * ar_mask).reshape(
                    -1, 1 + self.nb_digits
                )
                ar_mask = ar_mask.reshape(-1, 1 + self.nb_digits)

                nb_total = ar_mask.max(1).values.sum()
                nb_correct = nb_total - errors.max(1).values.sum()

                return nb_total, nb_correct

            test_nb_total, test_nb_correct = compute_nb_correct(self.test_input[:1000])

            logger(
                f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            ##############################################################
            # Log a few generated sequences
            input = self.test_input[:10, : 12 * (1 + self.nb_digits)]
            result = input.clone()
            stack.remove_popped_values(result, self.nb_stacks, self.nb_digits)
            ar_mask = (result != input).long()

            # for n in range(result.size(0)):
            # logger(
            # f"test_before {stack.seq_to_str(result[n],nb_stacks=self.nb_stacks,nb_digits=self.nb_digits)}"
            # )

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            for n in range(result.size(0)):
                logger(
                    f"test_after  {stack.seq_to_str(result[n],nb_stacks=self.nb_stacks,nb_digits=self.nb_digits)}"
                )
            ##############################################################

            model.train(t)


######################################################################


import expr


class Expr(Task):
    def tensorize(self, sequences):
        len_max = max([len(x) for x in sequences])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s + "#" * (len_max - len(s))]
                        for s in sequences
                    ]
                )
            ],
            0,
        ).to(self.device)

    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        nb_variables,
        sequence_length,
        batch_size,
        device=torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.device = device

        train_sequences = expr.generate_sequences(
            nb_train_samples,
            nb_variables=nb_variables,
            length=sequence_length,
        )

        test_sequences = expr.generate_sequences(
            nb_test_samples,
            nb_variables=nb_variables,
            length=sequence_length,
        )

        symbols = list(set("#" + "".join(train_sequences + test_sequences)))
        symbols.sort()

        self.char2id = dict([(c, n) for n, c in enumerate(symbols)])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

        self.filler, self.space = self.char2id["#"], self.char2id[" "]

        self.train_input = self.tensorize(train_sequences)
        self.test_input = self.tensorize(test_sequences)

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
            last = (batch != self.filler).max(0).values.nonzero().max() + 3
            batch = batch[:, :last]
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def seq2str(self, s):
        return "".join([self.id2char[k.item()] for k in s])

    def produce_results(
        self,
        n_epoch,
        model,
        result_dir,
        logger,
        deterministic_synthesis,
        input_file=None,
    ):
        with torch.autograd.no_grad():
            t = model.training
            model.eval()

            def compute_nb_correct(input):
                result = input.clone()
                s = (result == self.space).long()
                ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
                result = (1 - ar_mask) * result + ar_mask * self.filler
                masked_inplace_autoregression(
                    model,
                    self.batch_size,
                    result,
                    ar_mask,
                    deterministic_synthesis,
                    device=self.device,
                )

                nb_total = input.size(0)
                nb_correct = (input == result).long().min(1).values.sum()

                #######################################################################
                # Comput predicted vs. true variable values

                nb_delta = torch.zeros(5, dtype=torch.int64)
                nb_missed = 0

                values_input = expr.extract_results([self.seq2str(s) for s in input])
                values_result = expr.extract_results([self.seq2str(s) for s in result])

                for i, r in zip(values_input, values_result):
                    for n, vi in i.items():
                        vr = r.get(n)
                        if vr is None or vr < 0:
                            nb_missed += 1
                        else:
                            d = abs(vr - vi)
                            if d >= nb_delta.size(0):
                                nb_missed += 1
                            else:
                                nb_delta[d] += 1

                ######################################################################

                return nb_total, nb_correct, nb_delta, nb_missed

            (
                test_nb_total,
                test_nb_correct,
                test_nb_delta,
                test_nb_missed,
            ) = compute_nb_correct(self.test_input[:10000])

            logger(
                f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
            )

            nb_total = test_nb_delta.sum() + test_nb_missed
            for d in range(test_nb_delta.size(0)):
                logger(
                    f"error_value {n_epoch} delta {d} {test_nb_delta[d]} {test_nb_delta[d]*100/nb_total:.02f}%"
                )
            logger(
                f"error_value {n_epoch} missed {test_nb_missed} {test_nb_missed*100/nb_total:.02f}%"
            )

            ##############################################################
            # Log a few generated sequences
            if input_file is None:
                input = self.test_input[:10]
            else:
                with open(input_file, "r") as f:
                    sequences = [e.strip() for e in f.readlines()]
                    sequences = [s + " " + "#" * 50 for s in sequences]
                    input = self.tensorize(sequences)

            result = input.clone()
            s = (result == self.space).long()
            ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
            result = (1 - ar_mask) * result + ar_mask * self.filler

            for n in range(result.size(0)):
                logger(f"test_before {self.seq2str(result[n])}")

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            correct = (1 - ar_mask) * self.space + ar_mask * input
            for n in range(result.size(0)):
                comment = "GOOD" if (result[n] - input[n]).abs().max() == 0 else ""
                logger(f"test_after  {self.seq2str(result[n])} {comment}")
                logger(f"truth       {self.seq2str(correct[n])}")
            ##############################################################

            model.train(t)


######################################################################
