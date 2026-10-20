#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

def main(csv_path):
    # data[(variant, threads)][N] = [gflops1, gflops2, ...]
    data = defaultdict(lambda: defaultdict(list))

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                variant = row.get("variant", "").strip()
                if not variant:
                    continue

                # N doit exister
                N = int(row["N"])

                # gflops peut être "NA"
                gflops_str = row.get("gflops", "").strip()
                gflops = float(gflops_str)

                # threads peut ne pas exister (ancien CSV)
                threads = row.get("threads", "").strip()
                threads_key = threads if threads else "-"  # "-" = non défini

            except (KeyError, ValueError):
                continue

            data[(variant, threads_key)][N].append(gflops)

    plt.figure()

    # Tri: variant puis threads
    for (variant, threads_key) in sorted(data.keys(), key=lambda x: (x[0], x[1])):
        Ns = sorted(data[(variant, threads_key)].keys())
        means = []
        for N in Ns:
            vals = data[(variant, threads_key)][N]
            means.append(sum(vals) / len(vals))

        label = variant if threads_key == "-" else f"{variant} (threads={threads_key})"
        plt.plot(Ns, means, marker="o", label=label)

    plt.xlabel("Taille N (matrice NxN)")
    plt.ylabel("Performance (GFlop/s)")
    plt.title("Performances dgetrf en fonction de la taille")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} benchmark.csv")
        sys.exit(1)
    main(sys.argv[1])