#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
BUILD_DIR="build/default"
PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

# --- Output ---
OUT_DIR="perfs/logs"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/quick_benchmark_dgetrf.csv"

# --- What to test ---
VARIANTS=(seq nbloc bloc omp omp2 omp-t)

# Tailles "quick" (toutes variantes)
SIZES=(256 512 1024 1536)

# Tailles "big" (seulement variantes parallèles)
SIZES_BIG=(2048 3072 4096 7168 9216)

REPEATS=1
REPEATS_BIG=$((REPEATS + 2))   # "en rajoutant deux itérations"

# Threads OpenMP à tester (seulement pour omp/omp2)
OMP_THREADS=(4 12)

# Runner (local/slurm)
RUNNER=()
# RUNNER=(srun -C miriel -n 1 -c 16)

extract_gflops() {
  awk '{
    for (i=1;i<=NF;i++)
      if ($i ~ /GFlop\/s/) print $(i-1)
  }'
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

  # Choix des tailles selon la variante
  if [[ "${v}" == "omp" || "${v}" == "omp2" || "${v}" == "omp-t" ]]; then
    SIZE_LIST=("${SIZES[@]}" "${SIZES_BIG[@]}")
  else
    SIZE_LIST=("${SIZES[@]}")
  fi

  # Choix des threads selon la variante
  if [[ "${v}" == "omp" || "${v}" == "omp2" || "${v}" == "omp-t" ]]; then
    THREAD_LIST=("${OMP_THREADS[@]}")
  else
    THREAD_LIST=(1)
  fi

  for n in "${SIZE_LIST[@]}"; do

    # Repeats: +2 itérations uniquement sur les grandes tailles parallèles
    if [[ "${v}" == "omp" || "${v}" == "omp2" || "${v}" == "omp-t" ]]; then
      is_big=0
      for bn in "${SIZES_BIG[@]}"; do
        if [[ "${n}" -eq "${bn}" ]]; then is_big=1; break; fi
      done
      if [[ "${is_big}" -eq 1 ]]; then
        RPT="${REPEATS_BIG}"
      else
        RPT="${REPEATS}"
      fi
    else
      RPT="${REPEATS}"
    fi

    for t in "${THREAD_LIST[@]}"; do
      export OMP_NUM_THREADS="${t}"

      for r in $(seq 1 "${RPT}"); do
        echo "QUICK GETRF | v=${v} | N=${n} | thr=${t} | run=${r}"

        cmd=("${RUNNER[@]}" "${PERF_BIN}" -v "${v}" -M "${n}" -N "${n}")
        out="$("${cmd[@]}")"

        gflops="$(printf "%s\n" "${out}" | extract_gflops | head -n 1)"
        [[ -z "${gflops}" ]] && gflops="NA"

        echo "${v},${n},${n},${t},${r},${gflops}" >> "${OUT_CSV}"
      done
    done
  done
done

echo "Result in ${OUT_CSV}"