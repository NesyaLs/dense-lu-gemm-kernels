#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
BUILD_DIR="build/default"
PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

# --- Output ---
OUT_DIR="perfs/logs"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/omp_strong_scaling.csv"

# --- Variants OpenMP ---
# (dgetrf_omp.c, dgetrf_omp2.c, dgetrf_tiled_omp.c)
VARIANTS=(omp omp2 omp-t)

# --- Strong scaling: fixed problem size ---
# Choisis une taille "grosse" mais qui passe sur la RAM du noeud.
# Tu peux mettre 10000 si tu es sûr (10000^2 doubles ~ 800 MB pour A seul).
N_FIXED=3072
M_FIXED="${N_FIXED}"

# --- Threads to test (up to 12) ---
THREADS_LIST=(1 2 3 4 6 8 10 12)

# --- Repeats and averaging ---
REPEATS=3

# Runner (local/slurm)
RUNNER=()
# RUNNER=(srun -C miriel -n 1 -c 16)

extract_gflops() {
  awk '{
    for (i=1;i<=NF;i++)
      if ($i ~ /GFlop\/s/) { print $(i-1); exit }
  }'
}

mean_stdin() {
  awk '{s+=$1; n++} END{ if(n>0) printf "%.15g", s/n; else printf "NA"}'
}

if [[ ! -x "${PERF_BIN}" ]]; then
  echo "no binary file ${PERF_BIN}"
  exit 1
fi

# --- Env: avoid nested parallelism (BLAS threads) ---
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- OpenMP binding hints ---
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "variant,M,N,threads,repeat,gflops" > "${OUT_CSV}"

for v in "${VARIANTS[@]}"; do
  for t in "${THREADS_LIST[@]}"; do
    export OMP_NUM_THREADS="${t}"

    echo "STRONG OMP | v=${v} | N=${N_FIXED} | thr=${t} | repeats=${REPEATS}"

    vals=""
    for r in $(seq 1 "${REPEATS}"); do
      cmd=("${RUNNER[@]}" "${PERF_BIN}" -v "${v}" -M "${M_FIXED}" -N "${N_FIXED}")
      out="$("${cmd[@]}")"

      gflops="$(printf "%s\n" "${out}" | extract_gflops)"
      if [[ -n "${gflops}" ]]; then
        vals+="${gflops}"$'\n'
      else
        echo "WARN: no GFlop/s parsed (v=${v}, thr=${t}, run=${r})" >&2
      fi
    done

    avg="$(printf "%s" "${vals}" | mean_stdin)"
    echo "${v},${M_FIXED},${N_FIXED},${t},mean${REPEATS},${avg}" >> "${OUT_CSV}"
  done
done

echo "Result in ${OUT_CSV}"