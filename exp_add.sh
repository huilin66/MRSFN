#!/usr/bin/env bash
set -euo pipefail

# One command per prepared experiment. Each experiment can also be run
# independently from scripts/.
bash scripts/exp01_city_all_models.sh
bash scripts/exp02_capacity_control.sh
bash scripts/exp03_repeatability.sh
bash scripts/exp04_pmrg_evidence.sh
