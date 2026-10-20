#!/usr/bin/env python3
# Plot MPI strong scaling: mpi vs mpi2
# X axis = exactly the tested nprocs values

import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_path>")
    sys.exit(1)

CSV_PATH = sys.argv[1]

OUT_DIR = Path("perfs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# data[variant][N][nprocs] = list of gflops
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        variant = row["variant"]          # mpi or mpi2
        N = int(row["N"])
        nprocs = int(row["nprocs"])
        gflops = float(row["gflops"])
        data[variant][N][nprocs].append(gflops)

for N in sorted({N for v in data for N in data[v]}):
    plt.figure()

    all_x = set()

    for variant in ["mpi", "mpi2"]:
        if variant not in data or N not in data[variant]:
            continue

        x_vals = sorted(data[variant][N].keys())
        y_vals = [
            sum(data[variant][N][p]) / len(data[variant][N][p])
            for p in x_vals
        ]

        all_x.update(x_vals)
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", label=variant)

    plt.xlabel("Number of MPI processes")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title(f"DGETRF MPI strong scaling (N={N})")
    plt.xticks(sorted(all_x))
    plt.grid(True)
    plt.legend()

    out_file = OUT_DIR / f"mpi_strong_scaling_mpi_vs_mpi2_N{N}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()