#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# --- Paths ---
CSV_PATH = "perfs/logs/dgetrf_omp_size_scaling.csv"
OUT_DIR = Path("perfs/plots")
OUT_FILE = OUT_DIR / "dgetrf_omp_size_scaling_12t.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# data[variant][N] = list of gflops
data = defaultdict(lambda: defaultdict(list))
all_Ns = set()

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        variant = row["variant"]
        N = int(row["N"])
        gflops = float(row["gflops"])
        data[variant][N].append(gflops)
        all_Ns.add(N)

plt.figure()

for variant, values in data.items():
    Ns = sorted(values.keys())
    avg_gflops = [
        sum(values[N]) / len(values[N])
        for N in Ns
    ]
    plt.plot(Ns, avg_gflops, marker="o", label=variant)

# 🔒 forcer les ticks exactement sur les tailles testées
plt.xticks(sorted(all_Ns))

plt.xlabel("Matrix size N")
plt.ylabel("Performance (GFLOP/s)")
plt.title("DGETRF OpenMP scaling with problem size (12 threads)")
plt.grid(True)
plt.legend()

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
plt.close()