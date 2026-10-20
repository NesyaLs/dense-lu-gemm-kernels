#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

CSV_PATH = "perfs/logs/dgetrf_seq_nbloc_bloc_vendor.csv"
OUT_DIR = Path("perfs/plots")
OUT_FILE = OUT_DIR / "dgetrf_seq_nbloc_bloc_vendorWithSize.png"

data = defaultdict(lambda: defaultdict(list))

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        variant = row["variant"]
        N = int(row["N"])

        g = row["gflops"].strip()
        if g == "NA" or g == "":
            continue

        gflops = float(g)
        data[variant][N].append(gflops)

plt.figure()

for variant in ["seq", "nbloc", "bloc", "vendor"]:
    if variant not in data:
        continue

    values = data[variant]
    Ns = sorted(values.keys())
    avg = [sum(values[N]) / len(values[N]) for N in Ns]
    plt.plot(Ns, avg, marker="o", label=variant)

plt.xlabel("Matrix size N")
plt.ylabel("Performance (GFLOP/s)")
plt.title("DGETRF: seq (vectorized) vs nbloc (loop order) vs bloc (dgemm+seq, b=128) vs vendor (LAPACK)")
plt.grid(True)
plt.legend()

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
plt.close()