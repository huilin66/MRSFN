#!/usr/bin/env bash
set -euo pipefail

# Run the smoke variant of every currently implemented experiment.  Training
# scripts change only the total iteration override to 100; all model/data/
# optimizer/loss/batch settings remain inherited from their normal configs.
# EXP-04 is inference-only, so its --smoke flag uses a separate artifact
# directory but keeps all six evaluation conditions.
# EXP-05 is intentionally absent until its CMX model/configuration exists.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

smoke_dir="smoke_test"
memory_log="${smoke_dir}/gpu_memory.csv"
memory_summary="${smoke_dir}/gpu_memory_summary.json"
mkdir -p "${smoke_dir}"

monitor_pid=""
finish_smoke() {
  set +e
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1
    wait "${monitor_pid}" >/dev/null 2>&1
  fi
  if [[ -s "${memory_log}" ]]; then
    python tools/summarize_smoke_memory.py \
      --input "${memory_log}" \
      --output "${memory_summary}"
  else
    echo "[SMOKE] nvidia-smi did not produce a memory log: ${memory_log}"
  fi
}
trap finish_smoke EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    -l 1 > "${memory_log}" &
  monitor_pid=$!
else
  echo "[SMOKE] nvidia-smi is unavailable; training will still run without GPU memory logging."
fi

echo "[SMOKE] EXP-01"
bash scripts/exp01_city_all_models.sh --smoke

echo "[SMOKE] EXP-02"
bash scripts/exp02_capacity_control.sh --smoke

echo "[SMOKE] EXP-03"
bash scripts/exp03_repeatability.sh --smoke

echo "[SMOKE] EXP-04"
bash scripts/exp04_pmrg_evidence.sh --smoke

echo "[SMOKE] all currently implemented experiments completed"
