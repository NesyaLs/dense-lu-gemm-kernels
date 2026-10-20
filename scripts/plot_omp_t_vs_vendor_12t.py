#!/usr/bin/env python3
# Plot comparison: omp-t vs vendor (12 threads) from a CSV (no pandas)

import csv
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="perfs/logs/dgetrf_omp_t_vs_vendor_12t.csv",
                    help="Input CSV path")
    ap.add_argument("--out", default="perfs/plots/dgetrf_omp_t_vs_vendor_12t.png",
                    help="Output image path")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # data[variant][N] = list of gflops
    data = defaultdict(lambda: defaultdict(list))

    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]     # "omp-t" or "vendor"
            N = int(row["N"])
            g = row["gflops"].strip()
            if not g or g == "NA":
                continue
            data[variant][N].append(float(g))

    plt.figure()

    for variant in ["omp-t", "vendor"]:
        if variant not in data:
            continue
        Ns = sorted(data[variant].keys())
        avg = [sum(data[variant][N]) / len(data[variant][N]) for N in Ns]
        plt.plot(Ns, avg, marker="o", label=variant)

    plt.xlabel("Matrix size N")
    plt.ylabel("Performance (GFLOP/s)")
    plt.title("DGETRF (12 threads): omp-t vs vendor")
    plt.grid(True)
    plt.legend()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()