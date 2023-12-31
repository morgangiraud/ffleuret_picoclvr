
import pathlib
abs_path = pathlib.Path(__file__).parent.resolve()

import timeit
import matplotlib.pyplot as plt

number_of_executions = 2000
height, width = 11, 17
nb_walls_values = range(1, 13)

times_create_maze_old = []
times_create_maze = []
print(f"create_maze perf for:")
for nb_walls in nb_walls_values:
    print(f"    height, width, nb_walls = 11, 17, {nb_walls}")
    setup_code = f"""
import sys;sys.path.insert(0, "{abs_path}/../")
from maze import create_maze_old, create_maze
"""
    t0 = timeit.Timer(
        'create_maze_old(height, width, nb_walls)',
        setup=setup_code,
        globals={'height': height, 'width':width, 'nb_walls':nb_walls}
    )
    
    t1 = timeit.Timer(
        'create_maze(height, width, nb_walls)',
        setup=setup_code,
        globals={'height': height, 'width':width, 'nb_walls':nb_walls}
    )
    
    ms0 = t0.timeit(number_of_executions) / number_of_executions * 1e3
    times_create_maze_old.append(ms0)
    print(f"        create_maze_old:     {ms0} ms")

    ms1 = t1.timeit(number_of_executions) / number_of_executions * 1e3
    times_create_maze.append(ms1)
    print(f"        create_maze:         {ms1} ms")

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(nb_walls_values, times_create_maze_old, label='create_maze_old')
plt.plot(nb_walls_values, times_create_maze, label='create_maze')
plt.xlabel('Number of Walls')
plt.ylabel('Execution Time (ms)')
plt.title('Performance Comparison of create_maze_old and create_maze')
plt.legend()
plt.grid(True)
filepath = f"{abs_path}/../tmp/maze-perf.png"
plt.savefig(filepath)
print(f"Performance graph saved at {filepath}")
# plt.show()
