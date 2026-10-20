#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de benchmark pour projet HPC avec échantillonnage logarithmique (compatible Python 2.7 et 3)
"""

import subprocess
import math
import sys
import os
import time

try:
    raw_input
except NameError:
    raw_input = input

def get_user_input():
    """Récupère les paramètres utilisateur"""
    print("=== Configuration du benchmark ===\n")
    valid_versions = ["seq", "til", "mpi", "omp-t", "omp", "vendor", "starpu"]
    
    try:
        compute = str(raw_input("dgemm ou dgetrf : "))
        if compute != "dgemm" and compute != "dgetrf":
            print("Erreur : il faut choisir entre dgemm ou dgetrf")
            sys.exit(1)

        version = str(raw_input("Version à tester (seq, omp, mpi) : "))
        if version not in valid_versions:
            print("Erreur : la version sélectionnée n'est pas valide")
            sys.exit(1)

        cache = str(raw_input("Cold ou warm cache : "))
        if cache not in ["warm", "cold"]:
            print("Erreur : il faut choisir entre warm et cold")
            sys.exit(1)

        min_size = int(raw_input("Taille minimale des matrices : "))
        if min_size <= 0:
            print("Erreur : les tailles doivent être positives")
            sys.exit(1)

        max_size = int(raw_input("Taille maximale des matrices : "))
        if min_size > max_size or max_size <= 0:
            print("Erreur : la taille minimale doit être inférieure à la taille maximale")
            sys.exit(1)

        block_size = int(raw_input("Taille des blocs (pour les versions bloquées) : "))
        if block_size <= 0:
            print("Erreur : la taille des blocs doit être plus grande que 0")
            sys.exit(1)
        
        num_samples = int(raw_input("Nombre de tailles différentes à tester : "))
        if num_samples <= 0:
            print("Erreur : le nombre d'échantillons doit être plus grand que 0")
            sys.exit(1)

        num_iter = int(raw_input("Nombre de mesures pour chaque taille de matrice : "))
        if num_iter <= 0 or num_iter > 5:
            print("Erreur : le nombre de mesures pour chaque taille doit être compris entre 1 et 5")
            sys.exit(1)
        
        return compute, version, cache, min_size, max_size, block_size, num_samples, num_iter
    
    except ValueError:
        print("Erreur : veuillez entrer des nombres entiers valides")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
        sys.exit(0)

def generate_sizes(min_size, max_size, num_samples):
    """Génère une série de tailles avec progression logarithmique"""
    if num_samples == 1:
        return [min_size]
    
    log_min = math.log10(min_size)
    log_max = math.log10(max_size)
    step = (log_max - log_min) / float(num_samples - 1)
    
    sizes = []
    for i in range(num_samples):
        log_value = log_min + i * step
        size = int(round(10 ** log_value))
        sizes.append(size)
    
    return sorted(set(sizes))

# def generate_sizes(min_size, max_size, num_samples):
#     """Génère une série de tailles avec progression linéaire"""
#     if num_samples == 1:
#         return [min_size]
    
#     step = (max_size - min_size) / float(num_samples - 1)
#     sizes = []
    
#     for i in range(num_samples):
#         size = int(round(min_size + i * step))
#         sizes.append(size)
    
#     return sorted(set(sizes))

def run_benchmark(size, compute, version, num_iter, cache, block_size):
    """Exécute le benchmark pour une taille donnée"""
    if cache == "cold":
        c_cache = "perf"
    else:
        c_cache = cache
    cmd = [
        "./build/default/testings/{}_{}".format(c_cache, compute),
        "-N", str(size),
        "-K", str(size),
        "-M", str(size),
        "-b", str(block_size),
        "-i", str(num_iter),
        "-v", version
    ]
    
    try:
        print("  Exécution : {}".format(' '.join(cmd)))
        result = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = result.communicate()
        
        lines = stdout.strip().split('\n')
        del lines[0]
        
        print("  Résultat: {}".format(lines))
        if stderr:
            print("  Erreurs: {}".format(stderr.strip()))
        
        return {
            'size': size,
            'returncode': result.returncode,
            'stdout': lines,
            'stderr': stderr
        }
    
    except OSError:
        print("  ❌ Erreur: ./build/default/testings/perf_dgemm non trouvé")
        sys.exit(1)
    
    except Exception as e:
        print("  ❌ Erreur inattendue: {}".format(e))
        return {
            'size': size,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def save_results(results, filename, compute):
    """Sauvegarde les résultats dans un fichier (crée le dossier si besoin)"""
    # Créer le dossier parent s’il n’existe pas
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print("📁 Dossier créé : {}".format(directory))
        except OSError as e:
            print("❌ Erreur lors de la création du dossier : {}".format(e))
            sys.exit(1)

    # Écriture du fichier
    with open(filename, 'w') as f:
        f.write("Version: {}\n".format(compute))
        f.write("\n{}\n\n".format('-'*70))
        for result in results:
            f.write("Taille: {0}\n".format(result['size']))
            #f.write("Code retour: {}\n".format(result['returncode']))
            f.write("Sortie:")
            for line in result['stdout']:
                f.write("\n{}".format(line))
            if result['stderr']:
                f.write("Erreurs:\n{}\n".format(result['stderr']))
            f.write("\n{}\n\n".format('-'*70))

def main():
    print("\n" + "="*70)
    print("  SCRIPT DE BENCHMARK")
    print("="*70 + "\n")
    
    compute, version, cache, min_size, max_size, block_size, num_samples, num_iter = get_user_input()
    sizes = generate_sizes(min_size, max_size, num_samples)
    
    print("\n{}\nTailles à tester ({} valeurs) :\n  {}\n{}\n".format(
        '='*70, len(sizes), sizes, '='*70))
    
    try:
        response = raw_input("Lancer les benchmarks ? (o/N) : ").strip().lower()
        if response not in ['o', 'oui', 'y', 'yes']:
            print("Annulé.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nAnnulé.")
        sys.exit(0)
    
    print("\n{}\nDÉBUT DES BENCHMARKS\n{}\n".format('='*70, '='*70))
    
    results = []
    for i, size in enumerate(sizes, 1):
        print("[{}/{}] Test avec matrices {}x{}".format(i, len(sizes), size, size))
        result = run_benchmark(size, compute, version, num_iter, cache, block_size)
        results.append(result)
        print()
    
    output_file = "perfs/logs/benchmark_{}_{}_{}_{}-{}_{}.txt".format(
    cache, compute, version, min_size, max_size, block_size)

    save_results(results, output_file, compute)
    
    print("="*70)
    print("✅ Benchmarks terminés !")
    print("📄 Résultats sauvegardés dans : {}".format(output_file))
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
