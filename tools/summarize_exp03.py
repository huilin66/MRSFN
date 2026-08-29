#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize the EXP-03 repeated ordinary-BW runs.

The runner stores one log directory per condition/seed under ``log/exp03``.
This tool selects the validation block with the highest mIoU in each log,
then writes per-run metrics, condition mean/std values, matched-seed deltas,
and a sign-consistency report.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

CONDITIONS = (
    "cxup_1b_BW",
    "cxup_2b_BW",
    "cxup_3b_BW",
    "cxup_4b_BW",
    "cxup_4b_BW_PMRG",
    "cxup_4b_BW_loss",
    "cxup_4b_BW_PMRG_v2_lossV2",
)
SEEDS = (1919810, 1919811, 1919812)
METRICS = ("miou", "f1", "acc", "kappa")

EFFECTS = (
    ("2B-1B", "cxup_1b_BW", "cxup_2b_BW"),
    ("3B-2B", "cxup_2b_BW", "cxup_3b_BW"),
    ("4B-3B", "cxup_3b_BW", "cxup_4b_BW"),
    ("4B-1B", "cxup_1b_BW", "cxup_4b_BW"),
    ("4B+PMRG-4B", "cxup_4b_BW", "cxup_4b_BW_PMRG"),
    ("4B+Loss-4B", "cxup_4b_BW", "cxup_4b_BW_loss"),
    (
        "4B+PMRG+Loss-4B",
        "cxup_4b_BW",
        "cxup_4b_BW_PMRG_v2_lossV2",
    ),
    (
        "4B+PMRG+Loss-4B+PMRG",
        "cxup_4b_BW_PMRG",
        "cxup_4b_BW_PMRG_v2_lossV2",
    ),
    (
        "4B+PMRG+Loss-4B+Loss",
        "cxup_4b_BW_loss",
        "cxup_4b_BW_PMRG_v2_lossV2",
    ),
)

OVERALL_RE = re.compile(
    rf"^(?P<timestamp>\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}}).*?"
    rf"\[EVAL\]\s*#Images:\s*(?P<images>\d+)\s+"
    rf"F1:\s*(?P<f1>{FLOAT})\s*,?\s*"
    rf"mIoU:\s*(?P<miou>{FLOAT})\s+"
    rf"Acc:\s*(?P<acc>{FLOAT})\s+"
    rf"Kappa:\s*(?P<kappa>{FLOAT})",
    re.MULTILINE,
)

COMPLEXITY_RE = re.compile(
    rf"\[EVAL\]\s*Params:\s*(?P<params>{FLOAT})M\s+"
    rf"Trainable:\s*(?P<trainable>{FLOAT})M\s+"
    rf"FLOPs:\s*(?P<flops>{FLOAT})G\s+"
    rf"FPS:\s*(?P<fps>{FLOAT})"
)


@dataclass
class RunResult:
    condition: str
    seed: int
    log_path: str
    eval_count: int
    best_eval_index: int
    timestamp: str
    miou: float
    f1: float
    acc: float
    kappa: float
    params_m: Optional[float]
    trainable_m: Optional[float]
    flops_g: Optional[float]
    fps: Optional[float]


def read_text_auto(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_log(path: Path, condition: str, seed: int) -> RunResult:
    text = read_text_auto(path)
    matches = list(OVERALL_RE.finditer(text))
    if not matches:
        raise ValueError("no [EVAL] overall metric line found")

    candidates = []
    for source_index, match in enumerate(matches):
        next_start = (
            matches[source_index + 1].start()
            if source_index + 1 < len(matches)
            else len(text)
        )
        block = text[match.start():next_start]
        complexity = COMPLEXITY_RE.search(block)
        candidates.append(
            {
                "match": match,
                "complexity": complexity,
            }
        )

    best = max(candidates, key=lambda item: float(item["match"].group("miou")))
    match = best["match"]
    complexity = best["complexity"]

    def complexity_value(name: str) -> Optional[float]:
        return float(complexity.group(name)) if complexity else None

    return RunResult(
        condition=condition,
        seed=seed,
        log_path=str(path.resolve()),
        eval_count=len(matches),
        best_eval_index=matches.index(match) + 1,
        timestamp=match.group("timestamp"),
        miou=float(match.group("miou")),
        f1=float(match.group("f1")),
        acc=float(match.group("acc")),
        kappa=float(match.group("kappa")),
        params_m=complexity_value("params"),
        trainable_m=complexity_value("trainable"),
        flops_g=complexity_value("flops"),
        fps=complexity_value("fps"),
    )


def discover_logs(log_root: Path) -> tuple[dict[tuple[str, int], RunResult], list[str]]:
    results: dict[tuple[str, int], RunResult] = {}
    issues: list[str] = []
    if not log_root.is_dir():
        raise FileNotFoundError(f"log root does not exist: {log_root}")

    for path in sorted(log_root.rglob("*.log")):
        match = re.fullmatch(r"(?P<condition>.+)_seed(?P<seed>\d+)", path.parent.name)
        if not match:
            issues.append(f"ignored log with unexpected parent directory: {path}")
            continue

        condition = match.group("condition")
        seed = int(match.group("seed"))
        key = (condition, seed)
        if condition not in CONDITIONS or seed not in SEEDS:
            issues.append(f"ignored log outside EXP-03 design: {path}")
            continue
        if key in results:
            raise ValueError(f"duplicate log for {condition}, seed {seed}: {path}")

        try:
            results[key] = parse_log(path, condition, seed)
        except ValueError as exc:
            raise ValueError(f"cannot parse {path}: {exc}") from exc

    return results, issues


def format_number(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.8f}"


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def direction(values: list[float]) -> str:
    if all(value > 0 for value in values):
        return "positive"
    if all(value < 0 for value in values):
        return "negative"
    if all(value == 0 for value in values):
        return "zero"
    return "mixed"


def export_results(results: dict[tuple[str, int], RunResult], output_dir: Path) -> None:
    ordered = [results[(condition, seed)] for condition in CONDITIONS for seed in SEEDS]

    write_csv(
        output_dir / "per_run.csv",
        [
            "Condition", "Seed", "Config Path", "Checkpoint Path", "Log Path",
            "Eval Count", "Best Eval Index", "Best Eval Timestamp", "mIoU", "F1",
            "Acc", "Kappa", "Params (M)", "Trainable (M)", "FLOPs (G)", "FPS",
        ],
        [
            [
                item.condition, item.seed,
                f"PaddleCD/c2seg_config/{item.condition}.yml",
                f"output/exp03_{item.condition}_seed{item.seed}/best_model/model.pdparams",
                item.log_path, item.eval_count, item.best_eval_index, item.timestamp,
                format_number(item.miou),
                format_number(item.f1), format_number(item.acc),
                format_number(item.kappa), format_number(item.params_m),
                format_number(item.trainable_m), format_number(item.flops_g),
                format_number(item.fps),
            ]
            for item in ordered
        ],
    )

    summary_rows = []
    for condition in CONDITIONS:
        group = [results[(condition, seed)] for seed in SEEDS]
        row: list[object] = [condition, len(group)]
        for metric in METRICS:
            average, std = mean_std([getattr(item, metric) for item in group])
            row.extend((format_number(average), format_number(std)))
        for metric in ("params_m", "trainable_m", "flops_g", "fps"):
            values = [getattr(item, metric) for item in group]
            present = [value for value in values if value is not None]
            row.append(format_number(statistics.mean(present)) if present else "")
        summary_rows.append(row)

    write_csv(
        output_dir / "condition_summary.csv",
        [
            "Condition", "Runs",
            "mIoU Mean", "mIoU Std", "F1 Mean", "F1 Std",
            "Acc Mean", "Acc Std", "Kappa Mean", "Kappa Std",
            "Params Mean (M)", "Trainable Mean (M)", "FLOPs Mean (G)", "FPS Mean",
        ],
        summary_rows,
    )

    lookup = {(item.condition, item.seed): item for item in ordered}
    delta_rows = []
    for effect, baseline, candidate in EFFECTS:
        for seed in SEEDS:
            base = lookup[(baseline, seed)]
            target = lookup[(candidate, seed)]
            delta_rows.append(
                [
                    effect, baseline, candidate, seed,
                    *[
                        format_number(getattr(target, metric) - getattr(base, metric))
                        for metric in METRICS
                    ],
                ]
            )

    write_csv(
        output_dir / "matched_deltas.csv",
        ["Effect", "Baseline", "Candidate", "Seed", "mIoU Delta", "F1 Delta", "Acc Delta", "Kappa Delta"],
        delta_rows,
    )

    stability_rows = []
    for effect, baseline, candidate in EFFECTS:
        for metric in METRICS:
            deltas = [
                getattr(lookup[(candidate, seed)], metric)
                - getattr(lookup[(baseline, seed)], metric)
                for seed in SEEDS
            ]
            average, std = mean_std(deltas)
            positive = sum(value > 0 for value in deltas)
            negative = sum(value < 0 for value in deltas)
            zero = sum(value == 0 for value in deltas)
            delta_direction = direction(deltas)
            stability_rows.append(
                [
                    effect, metric, len(deltas), positive, negative, zero,
                    "yes" if delta_direction != "mixed" else "no",
                    delta_direction, format_number(average), format_number(std),
                    format_number(min(deltas)), format_number(max(deltas)),
                ]
            )

    write_csv(
        output_dir / "stability_summary.csv",
        [
            "Effect", "Metric", "Runs", "Positive", "Negative", "Zero",
            "Sign Consistent", "Direction", "Mean Delta", "Delta Std",
            "Min Delta", "Max Delta",
        ],
        stability_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the 21 EXP-03 repeated ordinary-BW runs."
    )
    parser.add_argument(
        "--log-root", type=Path, default=Path("log/exp03"),
        help="Root containing <condition>_seed<seed> log folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ana/exp03"),
        help="Directory for CSV summaries; default: ana/exp03.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        results, issues = discover_logs(args.log_root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    missing = sorted(expected - set(results))
    if missing:
        print("ERROR: EXP-03 is incomplete; missing:", file=sys.stderr)
        for condition, seed in missing:
            print(f"  {condition}, seed {seed}", file=sys.stderr)
        return 3

    export_results(results, args.output_dir)
    print(f"Parsed runs : {len(results)}/{len(expected)}")
    print(f"Output dir  : {args.output_dir.resolve()}")
    if issues:
        print(f"Ignored logs: {len(issues)}")
    print("Selection   : highest validation mIoU block in each log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
