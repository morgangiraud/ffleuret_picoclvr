import pathlib
from itertools import product

import torch
import torch.utils.benchmark as benchmark


abs_path = pathlib.Path(__file__).parent.resolve()

number_of_executions = 500
setup_code = f"""
import sys;sys.path.insert(0, "{abs_path}/../src")
import torch
import mygpt
from utils import seed_everything

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
Q = 60
V = 70
seed = 42

x.to()
bs = mygpt.BracketedSequence(x)

seed_everything(seed)
att = mygpt.QKVAttention(D, Q, V, 1, causal, 0.0).to(device)

seed_everything(seed)
att_fast = mygpt.QKVAttentionFast(D, Q, V, 1, causal, 0.0).to(device)
"""

label = "QKV attention"
causal = False
results = []
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


T = 32
sizes = [4, 64, 256]
for B, D in product(sizes, sizes):
    sub_label = f"{B}, {T}, {D}"
    x = torch.ones((B, T, D), device=device)

    for num_threads in [1, 4, 8]:
        print(f"{label}:{sub_label} - threds:{num_threads}, device: {device}, performing benchmark!")
        ms0 = benchmark.Timer(
            stmt="att(bs)",
            setup=setup_code,
            globals={"causal": causal, "D": D, "x": x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="att",
        ).blocked_autorange(min_run_time=1)
        ms1 = benchmark.Timer(
            stmt="att_fast(bs)",
            setup=setup_code,
            globals={"causal": causal, "D": D, "x": x},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="att_fast",
        ).blocked_autorange(min_run_time=1)
        results.append(ms0)
        results.append(ms1)
compare = benchmark.Compare(results)
compare.colorize()
compare.print()
