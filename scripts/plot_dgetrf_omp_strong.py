#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# --- Paths ---
CSV_PATH = "perfs/logs/dgetrf_omp_strong_scaling.csv"
OUT_DIR = Path("perfs/plots")
OUT_FILE = OUT_DIR / "dgetrf_omp_strong_scaling.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# data[variant][threads] = list of gflops
data = defaultdict(lambda: defaultdict(list))
all_threads = set()

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        variant = row["variant"]
        threads = int(row["threads"])
        gflops = float(row["gflops"])
        data[variant][threads].append(gflops)
        all_threads.add(threads)

plt.figure()

for variant, values in data.items():
    threads = sorted(values.keys())
    avg_gflops = [
        sum(values[t]) / len(values[t])
        for t in threads
    ]
    plt.plot(threads, avg_gflops, marker="o", label=variant)

# 🔒 forcer les ticks exactement sur les threads testés
plt.xticks(sorted(all_threads))

plt.xlabel("Number of OpenMP threads")
plt.ylabel("Performance (GFLOP/s)")
plt.title("DGETRF OpenMP strong scaling (fixed matrix size)")
plt.grid(True)
plt.legend()

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
plt.close()