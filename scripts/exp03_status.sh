#!/usr/bin/env bash
set -euo pipefail

# EXP-03 progress monitor. Read-only: never touches checkpoints or training.
#
#   bash scripts/exp03_status.sh            # print once, exit
#   bash scripts/exp03_status.sh --watch 30 # redraw every 30s (like watch -n 30)
#   bash scripts/exp03_status.sh --watch 1  # every 1s (heavier IO), Ctrl-C to stop
#
# For each of the 3 seed-groups it figures out which of the 7 models is
# currently training (the model whose log was most recently written), shows
# that model's iter/total, loss, per-iter time and ETA, then lists the models
# already finished with their best validation mIoU. A GPU line is included so
# you can see memory usage at a glance.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

SEEDS=(1919810 1919811 1919812)
MODELS=(cxup_1b_BW cxup_2b_BW cxup_3b_BW cxup_4b_BW \
        cxup_4b_BW_PMRG cxup_4b_BW_loss cxup_4b_BW_PMRG_v2_lossV2)
LOG_PREFIX="log/exp03"
CMD_LINE="PaddleCD/train.py"

watch_secs=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) watch_secs="$2"; shift 2 ;;
    --watch=*) watch_secs="${1#*=}"; shift ;;
    -h|--help) echo "Usage: $0 [--watch SECS]" >&2; exit 0 ;;
    *) echo "Usage: $0 [--watch SECS]" >&2; exit 2 ;;
  esac
done

# last matching number on the newest [TRAIN]/[EVAL] line
scrape() { grep -aoE "$1" "$2" 2>/dev/null | tail -n1 || true; }
# best validation mIoU for a finished/training model (just the number)
best_miou() { grep -aoE 'best validation mIoU \(([0-9.]+)\)' "$1" 2>/dev/null | tail -n1 | sed -E 's/.*\(([0-9.]+)\).*/\1/' || true; }

# row_seed <seed_index> :: print one row for the group's current model,
#                          then one line per already-finished model.
row_seed() {
  local si="$1"
  local seed="${SEEDS[$si]}"
  local cur="" idx="" best_done=""
  for m in "${MODELS[@]}"; do
    local dir="$LOG_PREFIX/${m}_seed${seed}"
    if compgen -G "$dir/*.log" >/dev/null 2>&1; then
      local newlog
      newlog="$(ls -t "${dir}"/*.log | head -n1)"
      if [[ -z "$cur" || "$newlog" -nt "$cur" ]]; then
        cur="$newlog"; idx="$m"
        best_done="$(best_miou "$cur")"
      fi
    fi
  done

  if [[ -z "$idx" ]]; then
    printf '  %-4s  %-8s  %-28s %s\n' "$si" "$seed" "(no training yet)" "waiting for launch"
    return
  fi

  local line lit loss bc eta full iter total pct cur_best
  line="$(scrape '\[TRAIN\] .*iter: [0-9]+/[0-9]+.*' "$cur")"
  lit="$(scrape 'iter: [0-9]+/[0-9]+' "$cur")"
  loss="$(scrape 'loss: [0-9.]+' "$cur" | cut -c7-)"
  bc="$(scrape 'batch_cost: [0-9.]+' "$cur" | cut -c13-)"
  eta="$(scrape 'ETA [0-9:]+' "$cur" | sed 's/.*ETA //')"
  full="${lit#iter: }"
  iter="${full%/*}"; total="${full#*/}"
  pct="?%"
  if (( total > 0 )); then pct=$(awk -v i="$iter" -v t="$total" 'BEGIN{if(t<=0)t=1; x=i*100/t; printf (x<10?"%.1f%%":"%d%%"), x}'); fi
  cur_best=""; [[ -n "$best_done" ]] && cur_best=" (best mIoU ${best_done})"
  printf '  %-4s  %-8s  %-28s %-12s %-6s %-11s %s\n' \
      "$si" "$seed" "$idx" "${full:-?/?}" "$pct" "${loss:---} ${bc:--}s" "${eta}${cur_best}"

  for m in "${MODELS[@]}"; do
    [[ "$m" == "$idx" ]] && break
    local d="$LOG_PREFIX/${m}_seed${seed}"
    if compgen -G "$d/*.log" >/dev/null 2>&1; then
      local dl b
      dl="$(ls -t "${d}"/*.log | head -n1)"
      b="$(best_miou "$dl")"
      printf '        finished %-24s best mIoU %s\n' "$m" "${b:-n/a}"
    fi
  done
}

print_all() {
  echo "EXP-03 status  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1) MiB | running train jobs: $(pgrep -fc "$CMD_LINE" 2>/dev/null || true)"
  echo
  printf ' %-4s  %-8s  %-28s %-12s %-6s %-11s %s\n' "GRP" "SEED" "MODEL" "ITER" "PCT" "LOSS/IT" "ETA/note"
  printf ' %s\n' "--------------------------------------------------------------------------------"
  row_seed 0
  row_seed 1
  row_seed 2
}

if [[ -n "$watch_secs" ]]; then
  interval="${watch_secs//[^0-9]/}"
  [[ -z "$interval" ]] && interval=30
  if command -v tput >/dev/null 2>&1 && [[ -n "${TERM:-}" ]]; then
    while true; do tput clear 2>/dev/null || true; print_all; sleep "$interval"; done
  else
    while true; do print_all; sleep "$interval"; done
  fi
else
  print_all
fi
