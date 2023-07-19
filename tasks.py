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
    assert input.size() == ar_mask.size()

    batches = zip(input.split(batch_size), ar_mask.split(batch_size))

    if progress_bar_desc is not None:
        batches = tqdm.tqdm(
            batches,
            dynamic_ncols=True,
            desc=progress_bar_desc,
            # total=input.size(0) // batch_size,
        )

    with torch.autograd.no_grad():
        t = model.training
        model.eval()

        for input, ar_mask in batches:
            model.masked_inplace_autoregression(
                input, ar_mask, forbidden_tokens, deterministic_synthesis
            )

        model.train(t)


######################################################################


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


class Problem:
    def generate_sequences(self, nb):
        pass

    def seq2str(self, seq):
        return "[NOT IMPLEMENTED]"


####################


class ProblemLevel0(Problem):
    def __init__(self, nb_sentences=100, len_prompt=5, len_result=5):
        self.seq = torch.randint(10, (nb_sentences, len_prompt + 1 + len_result))
        self.seq[:, len_prompt] = 10

    def generate_sequences(self, nb):
        sequences = self.seq[torch.randint(self.seq.size(0), (nb,))]
        ar_mask = (sequences == 10).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask


class ProblemLevel1(Problem):
    def __init__(self, nb_operators=100, len_source=5, len_result=8):
        self.len_source = len_source
        self.len_result = len_result
        self.len_nb_operator = int(math.log(nb_operators) / math.log(10)) + 1
        self.operators = F.one_hot(
            torch.rand(nb_operators, len_result, len_source).argmax(-1),
            num_classes=len_source,
        )

    def generate_sequences(self, nb):
        nb_operators = torch.randint(self.operators.size(0), (nb,))
        operators = self.operators[nb_operators]
        nb_operators = (
            nb_operators[:, None]
            // 10 ** torch.arange(self.len_nb_operator - 1, -1, -1)
        ) % 10
        marker1 = torch.full((nb, 1), 10)
        # source = torch.randint(10, (nb, self.len_source))
        source = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        marker2 = torch.full((nb, 1), 11)
        result = operators.bmm(source[:, :, None]).squeeze(-1)
        print(f"{nb_operators.dtype=} {marker1.dtype=}")
        sequences = torch.cat((nb_operators, marker1, source, marker2, result), 1)
        print(f"{sequences.size()=}")
        ar_mask = (sequences == 11).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789|>"[x.item()] for x in seq)


class ProblemLevel2(Problem):
    def __init__(self, len_source=5, len_result=8):
        self.len_source = len_source
        self.len_result = len_result

    def generate_sequences(self, nb):
        operators = F.one_hot(
            torch.rand(nb, self.len_result, self.len_source).argmax(-1),
            num_classes=self.len_source,
        )
        source1 = torch.rand(nb, 10).sort(dim=1).indices[:, : self.len_source]
        # source1 = torch.randint(10, (nb, self.len_source))
        marker1 = torch.full((nb, 1), 10)
        result1 = operators.bmm(source1[:, :, None]).squeeze(-1)
        marker2 = torch.full((nb, 1), 11)
        source2 = torch.randint(10, (nb, self.len_source))
        marker3 = torch.full((nb, 1), 12)
        result2 = operators.bmm(source2[:, :, None]).squeeze(-1)

        sequences = torch.cat(
            (source1, marker1, result1, marker2, source2, marker3, result2), 1
        )
        ar_mask = (sequences == 12).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join("0123456789>|~"[x.item()] for x in seq)


####################


class ProblemAddition(Problem):
    def __init__(self, nb_digits=10, zero_padded=False, inverted_result=False):
        self.nb_digits = nb_digits
        self.zero_padded = zero_padded
        self.inverted_result = inverted_result
        self.char2id = dict([(c, n) for n, c in enumerate("0123456789+=$")])
        self.id2char = dict([(n, c) for c, n in self.char2id.items()])

    def tensorize(self, strings):
        len_max = max([len(x) for x in strings])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [self.char2id[c] for c in s + "$" * (len_max - len(s))]
                        for s in strings
                    ]
                )
            ],
            0,
        )

    def generate_sequences(self, nb):
        sequences = []
        for k in range(nb):
            a, b = torch.randint(10**self.nb_digits, (2,))
            c = a + b
            a, b, c = str(a.item()), str(b.item()), str(c.item())
            if self.zero_padded:
                a = "0" * (self.nb_digits - len(a)) + a
                b = "0" * (self.nb_digits - len(b)) + b
                c = "0" * (self.nb_digits + 1 - len(c)) + c
            if self.inverted_result:
                c = c[::-1]
            sequences.append(f"{a}+{b}={c}$")

        sequences = self.tensorize(sequences)
        ar_mask = (sequences == self.char2id["="]).long()
        ar_mask = (ar_mask.cumsum(1) - ar_mask).clamp(max=1)
        return sequences, ar_mask

    def seq2str(self, seq):
        return "".join(self.id2char[x.item()] for x in seq)


# class ProblemUnion(Problem):
# problems = [ProblemByheart()]
# nb_common_codes = 100

# def generate_sequences(nb_samples):
# problem_indexes = torch.randint(len(problems), (nb_samples,))
# nb_samples_per_problem = torch.one_hot(problem_indexes).sum(0)
# print(f"{nb_samples_per_problem}")
# all_seq = []
# for nb, p in zip(nb_samples_per_problem, problems):
# all_seq.append(p.generate_sequences(nb_samples_per_problem[nb]))
# return all_seq

# for strain, stest in zip(train_seq, test_seq):
# s = torch.cat((strain, stest), 0)

####################


class SandBox(Task):
    def __init__(
        self,
        problem,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        logger=None,
        device=torch.device("cpu"),
        max_nb_codes=1024,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device
        self.problem = problem

        self.train_input, self.train_ar_mask = self.problem.generate_sequences(
            nb_train_samples
        )
        self.test_input, self.test_ar_mask = self.problem.generate_sequences(
            nb_test_samples
        )

        self.train_input, self.train_ar_mask = self.train_input.to(
            device
        ), self.train_ar_mask.to(device)
        self.test_input, self.test_ar_mask = self.test_input.to(
            device
        ), self.test_ar_mask.to(device)

        self.nb_codes = max(self.train_input.max(), self.test_input.max()) + 1

        # A bit of paranoia never hurts
        assert (
            self.nb_codes <= max_nb_codes
            and self.train_input.min() >= 0
            and self.test_input.min() >= 0
            and tuple(self.train_ar_mask.unique()) == (0, 1)
            and tuple(self.test_ar_mask.unique()) == (0, 1)
        )

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
        self, n_epoch, model, result_dir, logger, deterministic_synthesis, nmax=1000
    ):
        def compute_accuracy(input, ar_mask, logger=None):
            input, ar_mask = input[:nmax], ar_mask[:nmax]
            result = input.clone() * (1 - ar_mask)

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                progress_bar_desc=None,
                device=self.device,
            )

            if logger is not None:
                for sp, st in zip(result[:10], input[:10]):
                    logger(
                        f"test_sequences {n_epoch} prediction   {self.problem.seq2str(sp)}"
                    )
                    logger(
                        f"               {n_epoch} ground truth {self.problem.seq2str(st)}"
                    )

            nb_total = ar_mask.sum().item()
            nb_correct = ((result == input).long() * ar_mask).sum().item()

            return nb_total, nb_correct

        train_nb_total, train_nb_correct = compute_accuracy(
            self.train_input, self.train_ar_mask
        )

        logger(
            f"accuracy_train {n_epoch} nb_total {train_nb_total} nb_correct {train_nb_correct} accuracy {(100.0*train_nb_correct)/train_nb_total:.02f}%"
        )

        test_nb_total, test_nb_correct = compute_accuracy(
            self.test_input, self.test_ar_mask, logger
        )

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )


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
        super().__init__()

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
        super().__init__()

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
        super().__init__()

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
        super().__init__()

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
        def compute_nb_correct(input, prior_visits):
            result = input.clone()
            i = torch.arange(result.size(1), device=result.device)[None, :]
            ar_mask = (
                torch.logical_and(i >= self.prompt_length * 2, i % 2 == 0)
                .long()
                .expand_as(result)
            )
            result *= 1 - ar_mask

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            nb_total = ((prior_visits > 0) * ar_mask).sum()

            nb_correct = ((result == input).long() * (prior_visits > 0) * ar_mask).sum()

            return nb_total, nb_correct

        test_nb_total, test_nb_correct = compute_nb_correct(
            self.test_input[:1000], self.test_prior_visits[:1000]
        )

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_correct {test_nb_correct} accuracy {(100.0*test_nb_correct)/test_nb_total:.02f}%"
        )


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
        super().__init__()

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


######################################################################

import rpl


class RPL(Task):
    def tensorize(self, sequences):
        len_max = max([len(x) for x in sequences])
        return torch.cat(
            [
                torch.tensor(
                    [
                        [
                            self.token2id[str(c)]
                            for c in s + ["<nul>"] * (len_max - len(s))
                        ]
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
        batch_size,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device

        train_sequences = [
            rpl.generate()
            for _ in tqdm.tqdm(range(nb_train_samples), desc="train-data")
        ]
        test_sequences = [
            rpl.generate() for _ in tqdm.tqdm(range(nb_test_samples), desc="test-data")
        ]

        symbols = list(
            set(["<nul>"] + [x for l in train_sequences + test_sequences for x in l])
        )
        val_max = max([x if type(x) is int else 0 for x in symbols])
        symbols = list(filter(lambda x: type(x) is str, symbols))
        symbols.sort()
        symbols += [str(n) for n in range(val_max + 1)]
        print(f"{val_max=}")
        self.token2id = dict([(c, n) for n, c in enumerate(symbols)])
        self.id2token = dict([(n, c) for c, n in self.token2id.items()])

        self.t_nul, self.t_prog = self.token2id["<nul>"], self.token2id["<prog>"]

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
            last = (batch != self.t_nul).max(0).values.nonzero().max() + 3
            batch = batch[:, :last]
            yield batch

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        def compute_nb_errors(input, nb_to_log=0):
            result = input.clone()
            s = (result == self.t_prog).long()
            ar_mask = (s.cumsum(dim=1) - s).clamp(min=0, max=1)
            result = (1 - ar_mask) * result + ar_mask * self.t_nul

            masked_inplace_autoregression(
                model,
                self.batch_size,
                result,
                ar_mask,
                deterministic_synthesis,
                device=self.device,
            )

            if nb_to_log > 0:
                for x in result[:nb_to_log]:
                    s = " ".join([self.id2token[i.item()] for i in x])
                    logger(f"check {n_epoch} {s}")
                nb_to_log -= min(nb_to_log, result.size(0))

            sum_nb_total, sum_nb_errors = 0, 0
            for x in result:
                seq = [self.id2token[i.item()] for i in x]
                nb_total, nb_errors = rpl.check(seq)
                sum_nb_total += nb_total
                sum_nb_errors += nb_errors

            return sum_nb_total, sum_nb_errors

        test_nb_total, test_nb_errors = compute_nb_errors(self.test_input, nb_to_log=10)

        logger(
            f"accuracy_test {n_epoch} nb_total {test_nb_total} nb_errors {test_nb_errors} accuracy {100.0*(1-test_nb_errors/test_nb_total):.02f}%"
        )


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
        operand_max,
        result_max,
        batch_size,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device

        train_sequences = expr.generate_sequences(
            nb_train_samples,
            nb_variables=nb_variables,
            length=sequence_length,
            operand_max=operand_max,
            result_max=result_max,
        )

        test_sequences = expr.generate_sequences(
            nb_test_samples,
            nb_variables=nb_variables,
            length=sequence_length,
            operand_max=operand_max,
            result_max=result_max,
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

            filename = os.path.join(result_dir, f"expr_result_{n_epoch:04d}.txt")

            with open(filename, "w") as f:
                for i, r in zip(values_input, values_result):
                    for n, vi in i.items():
                        vr = r.get(n)
                        f.write(f"{vi} {-1 if vr is None else vr}\n")

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


######################################################################

import world


class World(Task):
    def __init__(
        self,
        nb_train_samples,
        nb_test_samples,
        batch_size,
        vqae_nb_epochs,
        logger=None,
        device=torch.device("cpu"),
        device_storage=torch.device("cpu"),
    ):
        super().__init__()

        self.batch_size = batch_size
        self.device = device

        (
            train_frames,
            train_action_seq,
            test_frames,
            test_action_seq,
            self.frame2seq,
            self.seq2frame,
        ) = world.create_data_and_processors(
            nb_train_samples,
            nb_test_samples,
            mode="first_last",
            nb_steps=30,
            nb_epochs=vqae_nb_epochs,
            logger=logger,
            device=device,
            device_storage=device_storage,
        )

        train_frame_seq = self.frame2seq(train_frames).to(device_storage)
        test_frame_seq = self.frame2seq(test_frames).to(device_storage)

        nb_frame_codes = max(train_frame_seq.max(), test_frame_seq.max()) + 1
        nb_action_codes = max(train_action_seq.max(), test_action_seq.max()) + 1

        self.len_frame_seq = train_frame_seq.size(1)
        self.len_action_seq = train_action_seq.size(1)
        self.nb_codes = nb_frame_codes + nb_action_codes

        train_frame_seq = train_frame_seq.reshape(train_frame_seq.size(0) // 2, 2, -1)

        train_action_seq += nb_frame_codes
        self.train_input = torch.cat(
            (train_frame_seq[:, 0, :], train_action_seq, train_frame_seq[:, 1, :]), 1
        )

        test_frame_seq = test_frame_seq.reshape(test_frame_seq.size(0) // 2, 2, -1)
        test_action_seq += nb_frame_codes
        self.test_input = torch.cat(
            (test_frame_seq[:, 0, :], test_action_seq, test_frame_seq[:, 1, :]), 1
        )

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
            yield batch.to(self.device)

    def vocabulary_size(self):
        return self.nb_codes

    def produce_results(
        self, n_epoch, model, result_dir, logger, deterministic_synthesis
    ):
        k = torch.arange(
            2 * self.len_frame_seq + self.len_action_seq, device=self.device
        )[None, :]

        input = self.test_input[:64].to(self.device)
        result = input.clone()

        ar_mask = (
            (k >= self.len_frame_seq + self.len_action_seq).long().expand_as(result)
        )
        result *= 1 - ar_mask

        masked_inplace_autoregression(
            model,
            self.batch_size,
            result,
            ar_mask,
            deterministic_synthesis,
            device=self.device,
        )

        seq_start = input[:, : self.len_frame_seq]
        seq_end = input[:, self.len_frame_seq + self.len_action_seq :]
        seq_predicted = result[:, self.len_frame_seq + self.len_action_seq :]

        result = torch.cat(
            (seq_start[:, None, :], seq_end[:, None, :], seq_predicted[:, None, :]), 1
        )
        result = result.reshape(-1, result.size(-1))

        frames = self.seq2frame(result)
        image_name = os.path.join(result_dir, f"world_result_{n_epoch:04d}.png")
        torchvision.utils.save_image(
            frames.float() / (world.Box.nb_rgb_levels - 1),
            image_name,
            nrow=12,
            padding=1,
            pad_value=0.0,
        )
        logger(f"wrote {image_name}")


######################################################################
