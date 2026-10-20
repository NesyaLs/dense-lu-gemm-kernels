#!/usr/bin/env bash
#SBATCH -J dgetrf_loop_order
#SBATCH -p routage
#SBATCH -C miriel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -o perfs/logs/%x_%j.out
#SBATCH -e perfs/logs/%x_%j.err

set -euo pipefail

module purge
module load build/cmake/3.21.3
module load compiler/gcc/11.2.0
module load mpi/openmpi/4.1.5
module load linalg/mkl/2020_update4

ROOT="${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
BUILD_DIR="${ROOT}/build/default"
PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

OUT_DIR="${ROOT}/perfs/logs"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/dgetrf_loop_order_seq_vs_nbloc_${SLURM_JOB_ID}.csv"

if [[ ! -x "${PERF_BIN}" ]]; then
  echo "ERROR: ${PERF_BIN} not found/executable"
  exit 1
fi

# (Même si on ne fait pas de MPI ici, on garde ton réglage cluster)
export OMPI_MCA_btl="^openib"

# Force mono-thread partout
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OMP_PROC_BIND=close
export OMP_PLACES=cores

VARIANTS=(seq nbloc)
SIZES=(128 256 512 768 1024)
REPEATS=3

extract_gflops() {
  awk '{
    for (i=1;i<=NF;i++)
      if ($i ~ /GFlop\/s/) { print $(i-1); exit }
  }'
}

echo "test,variant,N,repeat,gflops" > "${OUT_CSV}"

for v in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    for r in $(seq 1 "${REPEATS}"); do
      echo "DGETRF LOOP-ORDER | v=${v} | N=${n} | run=${r}"

      out="$(srun --exclusive -N 1 -n 1 "${PERF_BIN}" -v "${v}" -M "${n}" -N "${n}")"
      gflops="$(printf "%s\n" "${out}" | extract_gflops)"

      [[ -z "${gflops}" ]] && gflops="NA"
      echo "loop_order,${v},${n},run${r},${gflops}" >> "${OUT_CSV}"
    done
  done
done

echo "CSV written to: ${OUT_CSV}"