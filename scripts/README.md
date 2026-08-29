# Experiment scripts

Each experiment has an independent Bash entry point. Run them from the
repository root or from another directory; each script resolves and switches
to the repository root before invoking training.

- `exp01_city_all_models.sh`: EXP-01, all 15 city-split models plus Wuhan full-scene inference.
- `exp02_capacity_control.sh`: EXP-02, the two larger-backbone 1B supplemental BW runs.

The root-level `exp_add.sh` is only the sequential dispatcher. It contains one
command per prepared experiment.
