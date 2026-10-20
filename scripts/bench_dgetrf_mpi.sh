#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
BUILD_DIR="build/default"
PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

# --- Output ---
OUT_DIR="perfs/logs"
mkdir -p "${OUT_DIR}"
OUT_CSV="${OUT_DIR}/benchmark_dgetrf_mpi.csv"

# --- MPI variants to test ---
MPI_VARIANTS=(mpi mpi2)

# --- Tests requested ---
# 1) Strong scaling: fixed size, varying #processes
STRONG_N=4096
PROCS_LIST=(1 2 4 8 16)

# 2) Size scaling: fixed #processes, varying size up to 5000
# "on en test deux version pour chaque" -> 2 process counts
FIXED_PROCS_LIST=(4 16)
SIZES_GROW=(1024 2048 3072 4096 5000)

REPEATS=1

# Runner:
# - local: uses mpirun
# - slurm: comment mpirun and use srun below
MPI_LAUNCHER="mpirun"
# MPI_LAUNCHER="srun"

# Extra args (optional)
MPI_EXTRA_ARGS=()
# For OpenMPI you might want binding like:
# MPI_EXTRA_ARGS=(--bind-to core --map-by socket)

extract_gflops() {
  awk '{
    for (i=1;i<=NF;i++)
      if ($i ~ /GFlop\/s/) { print $(i-1); exit }
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
# if your code uses OpenMP inside MPI (hybrid), keep at 1 for pure MPI tests:
export OMP_NUM_THREADS=1

echo "test,variant,M,N,procs,repeat,gflops" > "${OUT_CSV}"

run_case() {
  local test_name="$1"
  local variant="$2"
  local n="$3"
  local procs="$4"
  local repeat="$5"

  echo "${test_name} | v=${variant} | N=${n} | np=${procs} | run=${repeat}"

  # Launch command
  # perf_dgetrf must accept -v mpi/mpi2 and -M/-N
  local -a cmd
  if [[ "${MPI_LAUNCHER}" == "srun" ]]; then
    # Slurm: one task per MPI rank
    cmd=(srun -n "${procs}" "${MPI_EXTRA_ARGS[@]}" "${PERF_BIN}" -v "${variant}" -M "${n}" -N "${n}")
  else
    # mpirun/mpiexec
    cmd=(mpirun -np "${procs}" "${MPI_EXTRA_ARGS[@]}" "${PERF_BIN}" -v "${variant}" -M "${n}" -N "${n}")
  fi

  local out gflops
  out="$("${cmd[@]}")"
  gflops="$(printf "%s\n" "${out}" | extract_gflops)"
  [[ -z "${gflops}" ]] && gflops="NA"

  echo "${test_name},${variant},${n},${n},${procs},${repeat},${gflops}" >> "${OUT_CSV}"
}

# -------------------------
# 1) Strong scaling (N fixed, procs varies)
# -------------------------
for v in "${MPI_VARIANTS[@]}"; do
  for p in "${PROCS_LIST[@]}"; do
    for r in $(seq 1 "${REPEATS}"); do
      run_case "strong_scaling" "${v}" "${STRONG_N}" "${p}" "${r}"
    done
  done
done

# -------------------------
# 2) Size scaling (procs fixed, N grows)
# -------------------------
for v in "${MPI_VARIANTS[@]}"; do
  for p in "${FIXED_PROCS_LIST[@]}"; do
    for n in "${SIZES_GROW[@]}"; do
      for r in $(seq 1 "${REPEATS}"); do
        run_case "size_scaling" "${v}" "${n}" "${p}" "${r}"
      done
    done
  done
done

echo "Result in ${OUT_CSV}"