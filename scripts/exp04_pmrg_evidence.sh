#!/usr/bin/env bash
set -euo pipefail

# EXP-04 is an independent inference-only experiment.  It reads the existing
# BW checkpoints directly; it does not invoke EXP-03 and does not depend on
# the order or contents of exp_add.sh.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

python tools/eval_exp04_pmrg_evidence.py \
  --baseline-config "PaddleCD/c2seg_config/cxup_4b_BW.yml" \
  --pmrg-config "PaddleCD/c2seg_config/cxup_4b_BW_PMRG.yml" \
  --baseline-checkpoint "output/cxup_4b_BW/best_model/model.pdparams" \
  --pmrg-checkpoint "output/cxup_4b_BW_PMRG/best_model/model.pdparams" \
  --seed 1919810 \
  --conditions clean missing_rgb missing_nirgb missing_sar missing_hsi noisy_hsi \
  --output-dir "ana/exp04" \
  --sample-indices 0 1 2
