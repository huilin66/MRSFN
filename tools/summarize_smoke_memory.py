#!/usr/bin/env python3
"""Summarize the nvidia-smi CSV captured by the smoke-test harness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize smoke-test GPU memory")
    parser.add_argument("--input", required=True, help="nvidia-smi CSV log")
    parser.add_argument("--output", required=True, help="JSON summary path")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    per_gpu: Dict[str, Dict[str, float]] = {}
    samples = 0

    if input_path.exists():
        with input_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if len(row) < 6:
                    continue
                try:
                    gpu_id = row[1].strip()
                    used_mib = float(row[-3].strip())
                    total_mib = float(row[-2].strip())
                    utilization = float(row[-1].strip().replace("%", ""))
                except (TypeError, ValueError):
                    continue
                samples += 1
                record = per_gpu.setdefault(
                    gpu_id,
                    {
                        "memory_total_mib": total_mib,
                        "peak_memory_used_mib": 0.0,
                        "peak_utilization_percent": 0.0,
                    },
                )
                record["memory_total_mib"] = total_mib
                record["peak_memory_used_mib"] = max(
                    record["peak_memory_used_mib"], used_mib
                )
                record["peak_utilization_percent"] = max(
                    record["peak_utilization_percent"], utilization
                )
                record["peak_memory_percent"] = (
                    100.0 * record["peak_memory_used_mib"] / total_mib
                    if total_mib > 0
                    else None
                )

    summary = {
        "input": str(input_path),
        "samples": samples,
        "gpu": per_gpu,
        "status": "recorded" if samples else "unavailable_or_empty",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
