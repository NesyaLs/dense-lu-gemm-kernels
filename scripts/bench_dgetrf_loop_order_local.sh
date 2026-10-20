#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Paths (relatifs au repo, comme quand tu bosses en local)
# ------------------------------------------------------------------
ROOT="$(pwd)"
BUILD_DIR="${ROOT}/build/default"
PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

OUT_DIR="${ROOT}/perfs/logs"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/dgetrf_loop_order_seq_vs_nbloc_local.csv"

# ------------------------------------------------------------------
# Vérification
# ------------------------------------------------------------------
if [[ ! -x "${PERF_BIN}" ]]; then
  echo "ERROR: ${PERF_BIN} not found or not executable"
  exit 1
fi

# ------------------------------------------------------------------
# Forcer le mono-thread (TRÈS IMPORTANT pour comprendre les perfs)
# ------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ------------------------------------------------------------------
# Expérience
# ------------------------------------------------------------------
VARIANTS=(seq nbloc)
SIZES=(128 256 512 768 1024)
REPEATS=3

extract_gflops() {
  awk '{
    for (i=1;i<=NF;i++) {
      if ($i ~ /GFlop\/s/) {
        print $(i-1)
        exit
      }
    }
  }'
}

echo "test,variant,N,repeat,gflops" > "${OUT_CSV}"

for v in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    for r in $(seq 1 "${REPEATS}"); do
      echo "DGETRF LOOP-ORDER | v=${v} | N=${n} | run=${r}"

      out="$("${PERF_BIN}" -v "${v}" -M "${n}" -N "${n}")"
      gflops="$(printf "%s\n" "${out}" | extract_gflops)"

      [[ -z "${gflops}" ]] && gflops="NA"
      echo "loop_order,${v},${n},run${r},${gflops}" >> "${OUT_CSV}"
    done
  done
done

echo "Results written to: ${OUT_CSV}"