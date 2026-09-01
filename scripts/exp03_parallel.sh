#!/usr/bin/env bash
set -euo pipefail

# EXP-03 one-click launcher: starts the 3 seed-parallel groups in the
# background (via nohup) and returns immediately, so a single terminal is
# enough. Monitor progress from any terminal with exp03_status.sh.
#
# Each group is ONE exp03_repeatability.sh process running its seed's 7-model
# block (cxup_1b..4b and the 4b variants). The 3 groups train concurrently on
# the same GPU (~9GB/model, 80GB total). Group stdout is captured under
# log/exp03_parallel/group{0,1,2}.log; the group PIDs are written to
# .exp03_pids so exp03_status.sh can show which are still alive.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

smoke_mode=false
resume_mode=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) smoke_mode=true; shift ;;
    --resume) resume_mode=true; shift ;;
    -h|--help) echo "Usage: $0 [--smoke] [--resume]" >&2; exit 0 ;;
    *) echo "Usage: $0 [--smoke] [--resume]" >&2; exit 2 ;;
  esac
done

mkdir -p log/exp03_parallel
: > .exp03_pids

declare -a pids=()
for g in 0 1 2; do
  args=(--group "$g")
  $smoke_mode && args+=(--smoke)
  $resume_mode && args+=(--resume)
  nohup bash "${script_dir}/exp03_repeatability.sh" "${args[@]}" \
      > "log/exp03_parallel/group${g}.log" 2>&1 &
  pids[$g]=$!
  echo "group${g} ${pids[$g]}" >> .exp03_pids
done

echo "[EXP-03] launched 3 parallel groups:"
echo "  group0 (seed 1919810): pid ${pids[0]}  -> log/exp03_parallel/group0.log"
echo "  group1 (seed 1919811): pid ${pids[1]}  -> log/exp03_parallel/group1.log"
echo "  group2 (seed 1919812): pid ${pids[2]}  -> log/exp03_parallel/group2.log"
echo "pids written to .exp03_pids"
echo "monitor with:  bash scripts/exp03_status.sh --watch 30"
