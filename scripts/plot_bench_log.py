import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import re
import math

def parse_data(filename):
    """
    Lit le fichier et extrait les données sous forme de dictionnaire:
    { taille: [perf1, perf2, ...] }
    """
    data = {}
    current_size = None
    
    # Regex pour capturer la taille et les GFlop/s
    re_size = re.compile(r"Taille:\s+(\d+)")
    re_gflops = re.compile(r"([\d\.e\+\-]+)\s+GFlop/s")

    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Détection de la taille
                match_size = re_size.search(line)
                if match_size:
                    current_size = int(match_size.group(1))
                    if current_size not in data:
                        data[current_size] = []
                    continue

                # Détection de la performance
                match_gflops = re_gflops.search(line)
                if match_gflops and current_size is not None:
                    val = float(match_gflops.group(1))
                    data[current_size].append(val)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' est introuvable.")
        sys.exit(1)
                
    return data

def calculate_cache_limits():
    """
    Calcule la taille N de matrice (NxN) qui remplit les caches.
    Hypothèse: 3 matrices (A, B, C) de doubles (8 octets).
    """
    caches = {
        "L1": 64 * 1024,       # 64 kB
        "L2": 256 * 1024,      # 256 kB
        "L3": 15360 * 1024     # 15360 kB
    }
    
    limits = {}
    for name, size_bytes in caches.items():
        # N = sqrt(Size / 24) car 3 matrices * 8 octets = 24 octets par élément de matrice
        n = math.sqrt(size_bytes / 24.0)
        limits[name] = n
    return limits

def plot_performance(data, output_path):
    # Préparation des données
    if not data:
        print("Aucune donnée valide trouvée dans le fichier.")
        sys.exit(1)

    sizes = sorted(data.keys())
    means = []
    mins = []
    maxs = []

    for s in sizes:
        perfs = data[s]
        means.append(np.mean(perfs))
        mins.append(np.min(perfs))
        maxs.append(np.max(perfs))

    sizes = np.array(sizes)
    means = np.array(means)
    mins = np.array(mins)
    maxs = np.array(maxs)
    
    # Bornes des données pour filtrer les caches
    min_data_size = np.min(sizes)
    max_data_size = np.max(sizes)

    # Configuration du style du plot
    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.grid(True, linestyle='--', alpha=0.5)

    # 1. Tracer la moyenne
    plt.plot(sizes, means, label='Performance Moyenne', color='#0072B2', linewidth=2)
    
    # 2. Zone d'écart (Min - Max)
    if np.any(maxs > mins):
        plt.fill_between(sizes, mins, maxs, color='#0072B2', alpha=0.2, label='Écart Min-Max')

    # 3. Lignes verticales pour les caches (Conditionnelles)
    cache_limits = calculate_cache_limits()
    colors = {'L1': '#D55E00', 'L2': '#E69F00', 'L3': '#009E73'}
    
    y_max_graph = np.max(maxs)

    for name, n_limit in cache_limits.items():
        # MODIFICATION ICI : On vérifie si la limite est dans la plage des données
        if min_data_size <= n_limit <= max_data_size:
            plt.axvline(x=n_limit, color=colors[name], linestyle='--', alpha=0.8)
            plt.text(n_limit * 1.05, y_max_graph * 0.95, f"Limite {name}\n(N≈{int(n_limit)})", 
                     color=colors[name], fontsize=9, rotation=90, verticalalignment='top')

    # Configuration des axes
    plt.xscale('log')
    
    plt.xlabel('Taille des matrices (NxN)', fontsize=12)
    plt.ylabel('Performance (GFlop/s)', fontsize=12)
    plt.title(f'Performance DGEMM, Version séquentielle tuilée et vectorisée (warm cache)', fontsize=14, pad=15)
    
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig(output_path, dpi=300)
    print(f"Graphique généré avec succès : {output_path}")

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère une courbe de performance DGEMM.")
    parser.add_argument("input_file", help="Chemin vers le fichier .txt contenant les mesures")
    
    args = parser.parse_args()

    # Définition du dossier et fichier de sortie
    output_dir = os.path.join("perfs", "graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(args.input_file)
    file_root, _ = os.path.splitext(base_name)
    output_filename = f"{file_root}_plot.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Lecture de {args.input_file}...")
    data = parse_data(args.input_file)
    plot_performance(data, output_path)