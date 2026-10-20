#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build/default"

PERF_BIN="${BUILD_DIR}/testings/perf_dgetrf"

OUT_DIR="perfs/logs"
mkdir -p "${OUT_DIR}"

OUT_CSV="${OUT_DIR}/benchmark_dgetrf.csv"

VARIANTS=(seq nbloc bloc)

SIZES=(100 200 400 800 1200 1600)

REPEATS=2

# Runner (MPI / Slurm / local)
# RUNNER=(mpirun -np 4)
RUNNER=()


extract_gflops() {
  awk '{ for (i=1;i<=NF;i++) if ($i ~ /GFlop\/s/) print $(i-1) }'
}


if [[ ! -x "${PERF_BIN}" ]]; then
  echo "no binary file ${PERF_BIN}"
  exit 1
fi


echo "variant,M,N,repeat,gflops" > "${OUT_CSV}"

for v in "${VARIANTS[@]}"; do
  for n in "${SIZES[@]}"; do
    for r in $(seq 1 "${REPEATS}"); do

      echo "GETRF | v=${v} | N=${n} | run=${r}"

      cmd=("${RUNNER[@]}" "${PERF_BIN}" -v "${v}" -M "${n}" -N "${n}")
      out="$("${cmd[@]}")"

      gflops="$(printf "%s\n" "${out}" | extract_gflops)"
      [[ -z "${gflops}" ]] && gflops="NA"

      echo "${v},${n},${n},${r},${gflops}" >> "${OUT_CSV}"
    done
  done
done

echo "Result in ${OUT_CSV}"