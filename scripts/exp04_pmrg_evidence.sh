#!/usr/bin/env bash
set -euo pipefail

# EXP-04 is an independent inference-only experiment.  It reads the existing
# BW checkpoints directly; it does not invoke EXP-03 and does not depend on
# the order or contents of exp_add.sh.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

smoke_mode=false
for arg in "$@"; do
  case "$arg" in
    --smoke) smoke_mode=true ;;
    # EXP-04 is inference-only, so resume is a no-op; accepted so that
    # exp_add.sh can forward --resume uniformly to every experiment.
    --resume) echo "[EXP-04] inference-only; --resume ignored" >&2 ;;
    *) echo "Usage: $0 [--smoke]" >&2; exit 2 ;;
  esac
done

if $smoke_mode; then
  exp04_output_dir="smoke_test/exp04"
else
  exp04_output_dir="ana/exp04"
fi

python tools/eval_exp04_pmrg_evidence.py \
  --baseline-config "PaddleCD/c2seg_config/cxup_4b_BW.yml" \
  --pmrg-config "PaddleCD/c2seg_config/cxup_4b_BW_PMRG.yml" \
  --baseline-checkpoint "output/cxup_4b_BW/best_model/model.pdparams" \
  --pmrg-checkpoint "output/cxup_4b_BW_PMRG_v2/best_model/model.pdparams" \
  --seed 1919810 \
  --conditions clean missing_rgb missing_nirgb missing_sar missing_hsi noisy_hsi \
  --output-dir "${exp04_output_dir}" \
  --sample-indices 0 1 2
