#!/usr/bin/env bash
set -euo pipefail

# One command per prepared experiment. Each experiment can also be run
# independently from scripts/.
#
# Usage:
#   bash exp_add.sh                run every experiment from scratch
#   bash exp_add.sh resume         resume every training experiment from its
#   bash exp_add.sh --resume       latest saved iter checkpoint

resume_mode=false
for arg in "$@"; do
  case "$arg" in
    resume|--resume) resume_mode=true ;;
    *) echo "Usage: $0 [resume]" >&2; exit 2 ;;
  esac
done

resume_flag=()
if $resume_mode; then
  echo "[exp_add] running experiments in resume mode"
  resume_flag=(--resume)
fi

bash scripts/exp01_city_all_models.sh "${resume_flag[@]}"
bash scripts/exp02_capacity_control.sh "${resume_flag[@]}"
bash scripts/exp03_repeatability.sh "${resume_flag[@]}"
bash scripts/exp04_pmrg_evidence.sh "${resume_flag[@]}"
