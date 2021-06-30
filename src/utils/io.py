import os

import matplotlib.pyplot as plt
import numpy as np

from instances import VRPInstance


def read_vrp(filepath: str) -> VRPInstance:
    with open(filepath, "r") as f:
        lines = [ll.strip() for ll in f]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                size = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + size], dtype=int)
                i = i + size
            elif line.startswith('DEMAND_SECTION'):
                demands = np.loadtxt(lines[i + 1:i + 1 + size], dtype=int)
                i = i + size
            i += 1

    def normalize(coord: int, grid_dim: int = capacity) -> float:
        return float(coord) / grid_dim

    return VRPInstance(
        depot=(normalize(locations[0][1]), normalize(locations[0][2])),
        customers=[(normalize(loc[1]), normalize(loc[2])) for loc in locations[1:]],
        demands=[d[1] for d in demands[1:]],
        capacity=capacity
    )


def write_vrp(instance: VRPInstance, filepath: str):
    with open(filepath, 'w') as f:
        f.write("\n".join([
            f"{key} : {value}"
            for key, value in (
                ("NAME", os.path.splitext(filepath)[0].split('/')[-1]),
                ("TYPE", "CVRP"),
                ("DIMENSION", instance.n_customers + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", instance.capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([f"{i + 1}\t{x}\t{y}" for i, (x, y) in enumerate([instance.depot] + instance.customers)]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([f"{i + 1}\t{d}" for i, d in enumerate([0] + instance.demands)]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":
    inst = read_vrp("../../res/A-n32-k5.vrp")
    inst.plot()
    plt.show()
