import pathlib
import timeit

import torch
import torch.utils.benchmark as benchmark
import matplotlib.pyplot as plt

abs_path = pathlib.Path(__file__).parent.resolve()

number_of_executions = 500
setup_code = f"""
import sys;sys.path.insert(0, "{abs_path}/../src")
from maze import create_maze_old, create_maze
"""

label = "Create maze function (small)"
results = []
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

height, width = 11, 17
nb_walls_values = range(1, 13)

times_create_maze_old = []
times_create_maze = []
for nb_walls in nb_walls_values:
    sub_label = f"{height}, {width}, {nb_walls}"

    for num_threads in [1]:
        print(f"{label}:{sub_label} - threds:{num_threads}, device: {device}, performing benchmark!")

        mes0 = benchmark.Timer(
            stmt="create_maze_old(height, width, nb_walls)",
            setup=setup_code,
            globals={"height": height, "width": width, "nb_walls": nb_walls},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Old",
        ).blocked_autorange(min_run_time=1)
        results.append(mes0)
        times_create_maze_old.append(mes0.mean)

        mes1 = benchmark.Timer(
            stmt="create_maze(height, width, nb_walls)",
            setup=setup_code,
            globals={"height": height, "width": width, "nb_walls": nb_walls},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="New",
        ).blocked_autorange(min_run_time=1)
        results.append(mes1)
        times_create_maze.append(mes1.mean)

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(nb_walls_values, times_create_maze_old, label="create_maze_old")
plt.plot(nb_walls_values, times_create_maze, label="create_maze")
plt.xlabel("Number of Walls")
plt.ylabel("Execution Time (ms)")
plt.title("Performance Comparison of create_maze_old and create_maze")
plt.legend()
plt.grid(True)
filepath = f"{abs_path}/../tmp/maze-perf_{height}_{width}.png"
plt.savefig(filepath)
print(f"Performance graph saved at {filepath}")
# plt.show()

label = "Create maze function (large)"
height, width = 31, 37
nb_walls_values = range(8, 23)

times_create_maze_old = []
times_create_maze = []
for nb_walls in nb_walls_values:
    sub_label = f"{height}, {width}, {nb_walls}"

    for num_threads in [1]:
        print(f"{label}:{sub_label} - threds:{num_threads}, device: {device}, performing benchmark!")

        mes0 = benchmark.Timer(
            stmt="create_maze_old(height, width, nb_walls)",
            setup=setup_code,
            globals={"height": height, "width": width, "nb_walls": nb_walls},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="Old",
        ).blocked_autorange(min_run_time=1)
        results.append(mes0)
        times_create_maze_old.append(mes0.mean)

        mes1 = benchmark.Timer(
            stmt="create_maze(height, width, nb_walls)",
            setup=setup_code,
            globals={"height": height, "width": width, "nb_walls": nb_walls},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description="New",
        ).blocked_autorange(min_run_time=1)
        results.append(mes1)
        times_create_maze.append(mes1.mean)

compare = benchmark.Compare(results)
compare.colorize()
compare.print()

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(nb_walls_values, times_create_maze_old, label="create_maze_old")
plt.plot(nb_walls_values, times_create_maze, label="create_maze")
plt.xlabel("Number of Walls")
plt.ylabel("Execution Time (ms)")
plt.title("Performance Comparison of create_maze_old and create_maze")
plt.legend()
plt.grid(True)
filepath = f"{abs_path}/../tmp/maze-perf_{height}_{width}.png"
plt.savefig(filepath)
print(f"Performance graph saved at {filepath}")
