#!/usr/bin/env bash

# Shared helpers for resumable experiment scripts. Source this file from an
# experiment script that has already resolved and switched to the repository
# root (the scripts set `script_dir`/`repo_root` before sourcing).

# latest_iter_ckpt <save_dir> ::
#   Print the newest <save_dir>/iter_* directory that contains both
#   model.pdparams and model.pdopt, so that train.py --resume_model can
#   continue from it. Prints an empty string when no such checkpoint exists.
#
#   Resume must target an iter_NNNN directory (not best_model): PaddleSeg's
#   resume() parses the starting iteration from the trailing number of the
#   directory name, so best_model (no trailing number) is not resumable.
#
#   This function always returns 0 so it is safe under `set -e` when used in
#   a command substitution (an empty result is a normal "nothing to resume").
latest_iter_ckpt() {
    local save_dir="$1"
    local ckpt=""
    if [[ -d "${save_dir}" ]]; then
        ckpt="$(ls -1d "${save_dir}"/iter_* 2>/dev/null \
            | grep -E '/iter_[0-9]+$' \
            | sort -V | tail -n1 || true)"
        if [[ ! -f "${ckpt}/model.pdparams" || ! -f "${ckpt}/model.pdopt" ]]; then
            ckpt=""
        fi
    fi
    echo "${ckpt}"
    return 0
}
