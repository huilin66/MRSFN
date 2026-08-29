#!/usr/bin/env python3
"""Run the independent EXP-04 PMRG evidence evaluation.

The tool evaluates the two already-existing BW checkpoints directly.  It does
not start EXP-03, read EXP-03 output directories, or depend on ``exp_add.sh``.
The default pair is the repository's existing one-seed pair::

    output/cxup_4b_BW/best_model/model.pdparams
    output/cxup_4b_BW_PMRG/best_model/model.pdparams

For each model it evaluates clean input, four branch-level missing conditions,
and fixed normalized-space HSI noise.  The PMRG model additionally exports
three-scale gate maps, global/class gate statistics, and selected sample
figures.  The evaluator intentionally treats gate values as feature
modulation weights, not calibrated reliability probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLECD_ROOT = REPO_ROOT / "PaddleCD"
if str(PADDLECD_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLECD_ROOT))

import paddle
from paddleseg.cvlibs import Config
from paddleseg.models import cx_uper  # noqa: F401: register custom models
from paddleseg.utils import config_check, metrics, utils


CONDITIONS = (
    "clean",
    "missing_rgb",
    "missing_nirgb",
    "missing_sar",
    "missing_hsi",
    "noisy_hsi",
)
CONDITION_LABELS = {
    "clean": "Clean",
    "missing_rgb": "Missing-RGB (branch-level view missing)",
    "missing_nirgb": "Missing-NIRGB (branch-level view missing)",
    "missing_sar": "Missing-SAR",
    "missing_hsi": "Missing-HSI",
    "noisy_hsi": "Noisy-HSI",
}
STAGE_LABELS = ("1/4", "1/8", "1/16")
STREAM_NAMES = ("NIRGB", "RGB", "SAR", "HSI")


@dataclass
class MetricResult:
    condition: str
    num_samples: int
    miou: float
    f1: float
    acc: float
    kappa: float
    class_iou: np.ndarray
    class_f1: np.ndarray
    class_acc: np.ndarray


class GateAccumulator:
    """Accumulate global and class-conditioned gate statistics."""

    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.sums: Optional[List[np.ndarray]] = None
        self.squares: Optional[List[np.ndarray]] = None
        self.entropy_sums: Optional[List[float]] = None
        self.counts: Optional[List[int]] = None
        self.class_sums: Dict[int, List[np.ndarray]] = {}
        self.class_squares: Dict[int, List[np.ndarray]] = {}
        self.class_entropy_sums: Dict[int, List[float]] = {}
        self.class_counts: Dict[int, List[int]] = {}

    def _initialize(self, num_stages: int) -> None:
        self.sums = [np.zeros(4, dtype=np.float64) for _ in range(num_stages)]
        self.squares = [np.zeros(4, dtype=np.float64) for _ in range(num_stages)]
        self.entropy_sums = [0.0 for _ in range(num_stages)]
        self.counts = [0 for _ in range(num_stages)]

    def _initialize_class(self, class_id: int, num_stages: int) -> None:
        self.class_sums[class_id] = [
            np.zeros(4, dtype=np.float64) for _ in range(num_stages)
        ]
        self.class_squares[class_id] = [
            np.zeros(4, dtype=np.float64) for _ in range(num_stages)
        ]
        self.class_entropy_sums[class_id] = [0.0 for _ in range(num_stages)]
        self.class_counts[class_id] = [0 for _ in range(num_stages)]

    def update(self, gates: Sequence[np.ndarray], labels: np.ndarray) -> None:
        if self.sums is None:
            self._initialize(len(gates))
        assert self.sums is not None
        assert self.squares is not None
        assert self.entropy_sums is not None
        assert self.counts is not None

        labels = np.asarray(labels)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]

        for stage_index, gate in enumerate(gates):
            gate = np.asarray(gate, dtype=np.float64)
            if gate.ndim != 4 or gate.shape[1] != 4:
                raise ValueError(
                    "Expected gate shape [B, 4, H, W], got {}".format(gate.shape)
                )
            batch_size, _, gate_height, gate_width = gate.shape
            flat_gate = gate.transpose(0, 2, 3, 1).reshape(-1, 4)
            entropy = -np.sum(
                flat_gate * np.log(np.clip(flat_gate, 1e-8, 1.0)), axis=1
            )
            self.sums[stage_index] += np.sum(flat_gate, axis=0)
            self.squares[stage_index] += np.sum(flat_gate * flat_gate, axis=0)
            self.entropy_sums[stage_index] += float(np.sum(entropy))
            self.counts[stage_index] += int(flat_gate.shape[0])

            # Nearest-neighbour label resizing keeps class masks aligned with
            # each gate scale without introducing fractional class IDs.
            import cv2

            for sample_index in range(batch_size):
                label_small = cv2.resize(
                    labels[sample_index].astype(np.int32),
                    (gate_width, gate_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                sample_gate = gate[sample_index].transpose(1, 2, 0).reshape(-1, 4)
                sample_entropy = -np.sum(
                    sample_gate * np.log(np.clip(sample_gate, 1e-8, 1.0)),
                    axis=1,
                )
                flat_label = label_small.reshape(-1)
                for class_id in range(self.num_classes):
                    mask = flat_label == class_id
                    if not np.any(mask):
                        continue
                    if class_id not in self.class_sums:
                        self._initialize_class(class_id, len(gates))
                    self.class_sums[class_id][stage_index] += np.sum(
                        sample_gate[mask], axis=0
                    )
                    self.class_squares[class_id][stage_index] += np.sum(
                        sample_gate[mask] * sample_gate[mask], axis=0
                    )
                    self.class_entropy_sums[class_id][stage_index] += float(
                        np.sum(sample_entropy[mask])
                    )
                    self.class_counts[class_id][stage_index] += int(np.sum(mask))

    @staticmethod
    def _summary_row(
        sums: np.ndarray,
        squares: np.ndarray,
        entropy_sum: float,
        count: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        if count <= 0:
            return (
                np.full(4, np.nan, dtype=np.float64),
                np.full(4, np.nan, dtype=np.float64),
                float("nan"),
                0,
            )
        mean = sums / count
        variance = np.maximum(squares / count - mean * mean, 0.0)
        return mean, np.sqrt(variance), entropy_sum / count, count

    def rows(self, condition: str, model: str, seed: int) -> List[dict]:
        if self.sums is None:
            return []
        assert self.squares is not None
        assert self.entropy_sums is not None
        assert self.counts is not None
        rows: List[dict] = []
        for scope, class_ids in (("global", [None]), ("class", sorted(self.class_sums))):
            for class_id in class_ids:
                if class_id is None:
                    sums = self.sums
                    squares = self.squares
                    entropy_sums = self.entropy_sums
                    counts = self.counts
                else:
                    sums = self.class_sums[class_id]
                    squares = self.class_squares[class_id]
                    entropy_sums = self.class_entropy_sums[class_id]
                    counts = self.class_counts[class_id]
                for stage_index, stage_name in enumerate(STAGE_LABELS):
                    mean, std, entropy, count = self._summary_row(
                        sums[stage_index],
                        squares[stage_index],
                        entropy_sums[stage_index],
                        counts[stage_index],
                    )
                    for stream_index, stream_name in enumerate(STREAM_NAMES):
                        rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "condition": condition,
                                "condition_label": CONDITION_LABELS[condition],
                                "scope": scope,
                                "class_id": "" if class_id is None else class_id,
                                "stage": stage_name,
                                "stream": stream_name,
                                "mean": float(mean[stream_index]),
                                "std": float(std[stream_index]),
                                "uniform_offset": float(mean[stream_index] - 0.25),
                                "entropy_mean": float(entropy),
                                "pixel_count": count,
                            }
                        )
        return rows

    def global_means(self) -> Dict[Tuple[str, str], float]:
        if self.sums is None:
            return {}
        assert self.counts is not None
        result: Dict[Tuple[str, str], float] = {}
        for stage_index, stage_name in enumerate(STAGE_LABELS):
            if self.counts[stage_index] <= 0:
                continue
            mean = self.sums[stage_index] / self.counts[stage_index]
            for stream_index, stream_name in enumerate(STREAM_NAMES):
                result[(stage_name, stream_name)] = float(mean[stream_index])
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent EXP-04 PMRG gate and missing-stream evaluation"
    )
    parser.add_argument(
        "--baseline-config",
        default="PaddleCD/c2seg_config/cxup_4b_BW.yml",
        help="Existing baseline configuration",
    )
    parser.add_argument(
        "--pmrg-config",
        default="PaddleCD/c2seg_config/cxup_4b_BW_PMRG.yml",
        help="Existing PMRG configuration",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="output/cxup_4b_BW/best_model/model.pdparams",
        help="Existing baseline checkpoint; not generated by this evaluator",
    )
    parser.add_argument(
        "--pmrg-checkpoint",
        default="output/cxup_4b_BW_PMRG/best_model/model.pdparams",
        help="Existing PMRG checkpoint; not generated by this evaluator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1919810,
        help="Seed recorded as checkpoint metadata (the evaluator does not retrain)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Validation batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Validation data-loader workers"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Paddle device; auto selects GPU when available",
    )
    parser.add_argument(
        "--output-dir",
        default="ana/exp04",
        help="Directory for EXP-04 tables, figures, and manifest",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
        help="Conditions to evaluate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Normalized-space standard deviation for noisy_hsi",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=20260829,
        help="Fixed seed for per-sample normalized-space noise",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="*",
        type=int,
        default=[0, 1, 2],
        help="Fixed validation sample indices to visualize",
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help="Collect gate statistics but skip PNG/NPZ visual artifacts",
    )
    return parser.parse_args()


def absolute_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        compiled = bool(paddle.is_compiled_with_cuda())
        gpu_count = int(paddle.device.cuda.device_count()) if compiled else 0
    except Exception:
        compiled, gpu_count = False, 0
    return "gpu" if compiled and gpu_count > 0 else "cpu"


def build_model(config_path: Path, checkpoint_path: Path, capture_gates: bool):
    if not config_path.exists():
        raise FileNotFoundError("Configuration not found: {}".format(config_path))
    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    cfg = Config(str(config_path))
    val_dataset = cfg.val_dataset
    if val_dataset is None:
        raise RuntimeError("No validation dataset in {}".format(config_path))
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty: {}".format(config_path))
    config_check(cfg, val_dataset=val_dataset)
    model = cfg.model
    utils.load_entire_model(model, str(checkpoint_path))
    model.eval()
    if capture_gates:
        if not hasattr(model, "set_gate_capture"):
            raise TypeError(
                "PMRG model {} does not expose set_gate_capture()".format(config_path)
            )
        model.set_gate_capture(True)
    return cfg, val_dataset, model


def make_loader(dataset, batch_size: int, num_workers: int):
    batch_sampler = paddle.io.BatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )


def make_noise(
    shape: Sequence[int], noise_seed: int, sample_offset: int, sigma: float
) -> paddle.Tensor:
    # The seed depends only on the validation sample offset.  Consequently the
    # same per-sample tensor is used for the baseline/PMRG pair and every run
    # that points to an equivalent checkpoint pair.
    rng = np.random.default_rng(int(noise_seed) + int(sample_offset))
    noise = rng.normal(0.0, sigma, size=tuple(shape)).astype("float32")
    return paddle.to_tensor(noise)


def as_label_array(label: paddle.Tensor) -> np.ndarray:
    result = label.numpy()
    if result.ndim == 4 and result.shape[1] == 1:
        result = result[:, 0]
    return result.astype(np.int32, copy=False)


def load_rgb_for_visual(path_value: str, fallback: np.ndarray) -> np.ndarray:
    """Load pre-normalization MSI RGB channels, with a tensor fallback."""
    try:
        from skimage import io

        image = np.asarray(io.imread(path_value))
        if image.ndim == 3 and image.shape[-1] >= 3:
            image = image[..., :3].astype(np.float32)
        else:
            raise ValueError("expected HWC image with at least three channels")
    except Exception as exc:
        print("[EXP-04] warning: RGB read failed for {}: {}".format(path_value, exc))
        image = np.asarray(fallback[:3]).transpose(1, 2, 0).astype(np.float32)

    rgb = np.zeros_like(image, dtype=np.uint8)
    for channel in range(3):
        band = image[..., channel]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            rgb[..., channel] = 0
        else:
            rgb[..., channel] = np.clip(
                (band - low) / (high - low) * 255.0, 0, 255
            ).astype(np.uint8)
    return rgb


def save_gate_visual(
    output_dir: Path,
    model_name: str,
    seed: int,
    condition: str,
    sample_index: int,
    rgb: np.ndarray,
    label: np.ndarray,
    prediction: np.ndarray,
    gates: Sequence[np.ndarray],
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate_dir = output_dir / "gate_maps" / model_name / str(seed) / condition
    gate_dir.mkdir(parents=True, exist_ok=True)
    stem = "sample_{:05d}".format(sample_index)
    np.savez_compressed(
        gate_dir / (stem + ".npz"),
        **{
            "gate_1_4": np.asarray(gates[0][0], dtype=np.float32),
            "gate_1_8": np.asarray(gates[1][0], dtype=np.float32),
            "gate_1_16": np.asarray(gates[2][0], dtype=np.float32),
        },
    )

    height, width = rgb.shape[:2]
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)
    valid_error = (label != 255) & (prediction != label)
    error = rgb.copy().astype(np.float32)
    error[valid_error] = 0.5 * error[valid_error] + np.array(
        [127.0, 0.0, 0.0], dtype=np.float32
    )
    error = np.clip(error, 0, 255).astype(np.uint8)

    figure, axes = plt.subplots(4, 4, figsize=(16, 16))
    base_images = (
        (rgb, "Original RGB", None),
        (label, "GT", "tab20"),
        (prediction, "Clean/condition prediction", "tab20"),
        (error, "Error overlay", None),
    )
    for axis, (image, title, cmap) in zip(axes[0], base_images):
        axis.imshow(image, cmap=cmap, vmin=0 if cmap else None)
        axis.set_title(title)
        axis.axis("off")

    for row, (stage_name, stage_gate) in enumerate(zip(STAGE_LABELS, gates), 1):
        stage_gate = np.asarray(stage_gate[0])
        for column, stream_name in enumerate(STREAM_NAMES):
            gate_map = cv2.resize(
                stage_gate[column], (width, height), interpolation=cv2.INTER_LINEAR
            )
            axis = axes[row, column]
            image = axis.imshow(gate_map, cmap="viridis", vmin=0.0, vmax=1.0)
            axis.set_title("{} — {}".format(stage_name, stream_name))
            axis.axis("off")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        "EXP-04 PMRG gate evidence | model={} seed={} condition={} sample={}".format(
            model_name, seed, CONDITION_LABELS[condition], sample_index
        ),
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(gate_dir / (stem + ".png"), dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    model_name: str,
    seed: int,
    model,
    val_dataset,
    conditions: Sequence[str],
    batch_size: int,
    num_workers: int,
    sigma: float,
    noise_seed: int,
    sample_indices: Sequence[int],
    output_dir: Path,
    capture_gates: bool,
    skip_visuals: bool,
) -> Tuple[Dict[str, MetricResult], Dict[str, GateAccumulator]]:
    num_classes = int(getattr(val_dataset, "num_classes", 14))
    metric_results: Dict[str, MetricResult] = {}
    gate_results: Dict[str, GateAccumulator] = {}
    requested_samples = set(int(index) for index in sample_indices)

    for condition in conditions:
        print(
            "[EXP-04] model={} seed={} condition={} ({})".format(
                model_name, seed, condition, CONDITION_LABELS[condition]
            )
        )
        loader = make_loader(val_dataset, batch_size, num_workers)
        intersect_total = pred_total = label_total = None
        num_samples = 0
        gate_accumulator = GateAccumulator(num_classes) if capture_gates else None
        sample_offset = 0

        model.eval()
        with paddle.no_grad():
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError(
                        "EXP-04 expects validation batches (im1, im2, label), got {} items".format(
                            len(batch)
                        )
                    )
                im1, im2, label = batch
                current_batch = int(im1.shape[0])
                noise = None
                if condition == "noisy_hsi":
                    noise = make_noise(
                        im2.shape, noise_seed, sample_offset, sigma
                    )
                if condition == "clean":
                    output = model(im1, im2)
                else:
                    output = model(
                        im1,
                        im2,
                        perturbation=condition,
                        noise=noise,
                    )
                logits = output[0] if isinstance(output, (list, tuple)) else output
                prediction = paddle.argmax(logits, axis=1)
                label = label.astype("int64")
                intersect, pred_area, label_area = metrics.calculate_area(
                    prediction, label, num_classes
                )
                if intersect_total is None:
                    intersect_total = intersect
                    pred_total = pred_area
                    label_total = label_area
                else:
                    intersect_total += intersect
                    pred_total += pred_area
                    label_total += label_area

                label_np = as_label_array(label)
                prediction_np = prediction.numpy().astype(np.int32, copy=False)
                if capture_gates:
                    gates = getattr(model, "last_gates", None)
                    if gates is None or len(gates) != 3:
                        raise RuntimeError(
                            "PMRG model did not cache three gate tensors in condition {}".format(
                                condition
                            )
                        )
                    gates_np = [gate.numpy() for gate in gates]
                    assert gate_accumulator is not None
                    gate_accumulator.update(gates_np, label_np)

                    if not skip_visuals:
                        for local_index in range(current_batch):
                            global_index = sample_offset + local_index
                            if global_index not in requested_samples:
                                continue
                            try:
                                image_path = val_dataset.file_list[global_index][0]
                                rgb = load_rgb_for_visual(
                                    image_path, im1[local_index].numpy()
                                )
                                save_gate_visual(
                                    output_dir=output_dir,
                                    model_name=model_name,
                                    seed=seed,
                                    condition=condition,
                                    sample_index=global_index,
                                    rgb=rgb,
                                    label=label_np[local_index],
                                    prediction=prediction_np[local_index],
                                    gates=[
                                        gate[local_index : local_index + 1]
                                        for gate in gates_np
                                    ],
                                )
                            except Exception as exc:
                                print(
                                    "[EXP-04] warning: visual sample {} failed: {}".format(
                                        global_index, exc
                                    )
                                )
                num_samples += current_batch
                sample_offset += current_batch

        if intersect_total is None:
            raise RuntimeError("Validation loader produced no batches")
        class_iou, miou = metrics.mean_iou(
            intersect_total, pred_total, label_total
        )
        class_f1, f1 = metrics.get_f1(intersect_total, pred_total, label_total)
        class_acc, acc = metrics.accuracy(intersect_total, pred_total)
        kappa = float(metrics.kappa(intersect_total, pred_total, label_total))
        metric_results[condition] = MetricResult(
            condition=condition,
            num_samples=num_samples,
            miou=float(miou),
            f1=float(f1),
            acc=float(acc),
            kappa=kappa,
            class_iou=np.asarray(class_iou),
            class_f1=np.asarray(class_f1),
            class_acc=np.asarray(class_acc),
        )
        if gate_accumulator is not None:
            gate_results[condition] = gate_accumulator

    return metric_results, gate_results


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(
    model_name: str,
    seed: int,
    config_path: Path,
    checkpoint_path: Path,
    results: Mapping[str, MetricResult],
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    class_rows: List[dict] = []
    clean = results.get("clean")
    for condition, result in results.items():
        row = {
            "model": model_name,
            "seed": seed,
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            "num_samples": result.num_samples,
            "miou": result.miou,
            "f1": result.f1,
            "acc": result.acc,
            "kappa": result.kappa,
            "delta_miou_from_clean": "" if clean is None else clean.miou - result.miou,
            "delta_f1_from_clean": "" if clean is None else clean.f1 - result.f1,
            "delta_acc_from_clean": "" if clean is None else clean.acc - result.acc,
            "delta_kappa_from_clean": "" if clean is None else clean.kappa - result.kappa,
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        }
        rows.append(row)
        for class_id, (iou, f1, acc) in enumerate(
            zip(result.class_iou, result.class_f1, result.class_acc)
        ):
            class_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "class_id": class_id,
                    "iou": float(iou),
                    "f1": float(f1),
                    "acc": float(acc),
                    "config": str(config_path),
                    "checkpoint": str(checkpoint_path),
                }
            )
    return rows, class_rows


def main(args: argparse.Namespace) -> None:
    output_dir = absolute_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {
        "baseline": absolute_path(args.baseline_config),
        "pmrg": absolute_path(args.pmrg_config),
    }
    checkpoint_paths = {
        "baseline": absolute_path(args.baseline_checkpoint),
        "pmrg": absolute_path(args.pmrg_checkpoint),
    }
    conditions = list(args.conditions)
    if "clean" not in conditions:
        raise ValueError("EXP-04 requires clean as the delta reference condition")
    if "noisy_hsi" in conditions and args.sigma < 0:
        raise ValueError("sigma must be non-negative")

    device = select_device(args.device)
    paddle.set_device(device)
    print("[EXP-04] device={}".format(device))
    print("[EXP-04] this run reads existing checkpoints; it does not retrain")

    manifest = {
        "experiment": "EXP-04",
        "purpose": "PMRG gate evidence and branch-level missing/noisy-stream evaluation",
        "independent_from": ["EXP-01", "EXP-02", "EXP-03", "exp_add.sh"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_metadata": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "conditions": conditions,
        "condition_labels": CONDITION_LABELS,
        "noise": {
            "condition": "noisy_hsi",
            "space": "after Normalize2",
            "stream": "HSI",
            "sigma": args.sigma,
            "seed": args.noise_seed,
            "same_per_sample_realization_across_models": True,
        },
        "masking": {
            "space": "after Normalize2",
            "location": "after four-stream split",
            "zero_semantics": "replace by each branch's normalized training mean",
            "rgb_and_nirgb_label": "branch-level view missing",
            "sar_and_hsi_label": "independent-modality-like missing",
        },
        "gate": {
            "stages": list(STAGE_LABELS),
            "streams": list(STREAM_NAMES),
            "interpretation": "feature modulation weight, not calibrated reliability probability",
        },
        "visual_sample_indices": [int(index) for index in args.sample_indices],
        "models": {
            name: {
                "config": str(config_paths[name]),
                "checkpoint": str(checkpoint_paths[name]),
            }
            for name in ("baseline", "pmrg")
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    all_metric_rows: List[dict] = []
    all_class_rows: List[dict] = []
    all_gate_rows: List[dict] = []
    gate_delta_rows: List[dict] = []

    for model_name in ("baseline", "pmrg"):
        capture_gates = model_name == "pmrg"
        cfg, val_dataset, model = build_model(
            config_paths[model_name], checkpoint_paths[model_name], capture_gates
        )
        results, gate_results = evaluate_model(
            model_name=model_name,
            seed=args.seed,
            model=model,
            val_dataset=val_dataset,
            conditions=conditions,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sigma=args.sigma,
            noise_seed=args.noise_seed,
            sample_indices=args.sample_indices,
            output_dir=output_dir,
            capture_gates=capture_gates,
            skip_visuals=args.skip_visuals,
        )
        metric_rows_now, class_rows_now = metric_rows(
            model_name,
            args.seed,
            config_paths[model_name],
            checkpoint_paths[model_name],
            results,
        )
        all_metric_rows.extend(metric_rows_now)
        all_class_rows.extend(class_rows_now)

        if model_name == "pmrg":
            for condition, accumulator in gate_results.items():
                all_gate_rows.extend(accumulator.rows(condition, model_name, args.seed))
            clean_means = gate_results.get("clean", GateAccumulator(14)).global_means()
            for condition, accumulator in gate_results.items():
                if condition == "clean":
                    continue
                perturbed_means = accumulator.global_means()
                for key, clean_mean in clean_means.items():
                    stage_name, stream_name = key
                    perturbed_mean = perturbed_means.get(key, float("nan"))
                    gate_delta_rows.append(
                        {
                            "model": model_name,
                            "seed": args.seed,
                            "condition": condition,
                            "condition_label": CONDITION_LABELS[condition],
                            "stage": stage_name,
                            "stream": stream_name,
                            "clean_mean": clean_mean,
                            "perturbed_mean": perturbed_mean,
                            "delta_perturbed_minus_clean": perturbed_mean - clean_mean,
                        }
                    )

    write_csv(
        output_dir / "metrics.csv",
        all_metric_rows,
        (
            "model",
            "seed",
            "condition",
            "condition_label",
            "num_samples",
            "miou",
            "f1",
            "acc",
            "kappa",
            "delta_miou_from_clean",
            "delta_f1_from_clean",
            "delta_acc_from_clean",
            "delta_kappa_from_clean",
            "config",
            "checkpoint",
        ),
    )
    write_csv(
        output_dir / "class_metrics.csv",
        all_class_rows,
        (
            "model",
            "seed",
            "condition",
            "condition_label",
            "class_id",
            "iou",
            "f1",
            "acc",
            "config",
            "checkpoint",
        ),
    )
    write_csv(
        output_dir / "gate_stats.csv",
        all_gate_rows,
        (
            "model",
            "seed",
            "condition",
            "condition_label",
            "scope",
            "class_id",
            "stage",
            "stream",
            "mean",
            "std",
            "uniform_offset",
            "entropy_mean",
            "pixel_count",
        ),
    )
    write_csv(
        output_dir / "gate_deltas_from_clean.csv",
        gate_delta_rows,
        (
            "model",
            "seed",
            "condition",
            "condition_label",
            "stage",
            "stream",
            "clean_mean",
            "perturbed_mean",
            "delta_perturbed_minus_clean",
        ),
    )
    print("[EXP-04] wrote results to {}".format(output_dir))


if __name__ == "__main__":
    main(parse_args())
