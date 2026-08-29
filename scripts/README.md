# Experiment scripts

Each experiment has an independent Bash entry point. Run them from the
repository root or from another directory; each script resolves and switches
to the repository root before invoking training.

- `exp01_city_all_models.sh`: EXP-01, all 15 city-split models plus Wuhan full-scene inference.
- `exp02_capacity_control.sh`: EXP-02, the two larger-backbone 1B supplemental BW runs.
- `exp03_repeatability.sh`: EXP-03, the 1B--4B branch-count chain and the 4B PMRG/loss ablations over three fixed seeds (21 runs).

After EXP-03 finishes, generate the paired statistics with:

```bash
python tools/summarize_exp03.py
```

The root-level `exp_add.sh` is only the sequential dispatcher. It contains one
command per prepared experiment.
