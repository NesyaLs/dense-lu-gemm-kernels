#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

CSV_PATH = "perfs/logs/dgetrf_vecto_impact.csv"
OUT_DIR = Path("perfs/plots")
OUT_FILE = OUT_DIR / "dgetrf_vecto_impact.png"

data = defaultdict(lambda: defaultdict(list))

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        variant = row["variant"]          # seq or nbloc
        N = int(row["N"])
        gflops = float(row["gflops"])
        data[variant][N].append(gflops)

plt.figure()

for variant, values in data.items():
    Ns = sorted(values.keys())
    avg_gflops = [
        sum(values[N]) / len(values[N])
        for N in Ns
    ]
    plt.plot(Ns, avg_gflops, marker="o", label=variant)

plt.xlabel("Matrix size N")
plt.ylabel("Performance (GFLOP/s)")
plt.title("DGETRF – vectorisation impact: seq (vect) vs nbloc (good loop order)")
plt.grid(True)
plt.legend()

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
plt.close()