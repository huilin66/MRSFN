"""Prepare the WHU-H2SR-MT data set as 128 x 128 TIFF patches.

Raw release layout (default root ``Z:/huilin/backup/data_backup/WHU_H2SR_MT``)::

    <root>/<stem>.tif                   HSI, 300 x 300 x 249, float32
    <root>/H2SR_image_11/<stem>.tif     HSI for block 11 (only block packaged
                                        inside a folder; other blocks are at root)
    <root>/H2SR_label_<b>/<stem>.tif    HSI label, 300 x 300, uint8
    <root>/MT_image_<b>/<stem>.npy      Sentinel-2, (31, 10, 30, 30), float32
    <root>/MT_label_<b>/<stem>.tif      Sentinel-2 label, 30 x 30, uint8

The release has eight classes (0-7); value 15 (and 255, if present) is treated
as ignore and written as 255.  The raw patches are already 300 x 300, so the
default operation is a whole-patch resize to 128 x 128 (bilinear for images,
nearest for labels), not a 3 x 3 tiling.  Sentinel-2 is upsampled from 30 x 30
to 128 x 128 and stored channel-last as ``[128, 128, 310]`` with channel order
``c = date * 10 + band``.  Use ``--mt-layout dates`` to store
``[31, 10, 128, 128]`` instead.

Output layout::

    <out>/hsi/<id>.tif        [128, 128, 249] float32
    <out>/mt/<id>.tif         [128, 128, 310] float32 (default)
    <out>/lbl/<id>.tif        [128, 128]       uint8, 255 = ignore
    <out>/mt_lbl/<id>.tif     [128, 128]       uint8, 255 = ignore
    <out>/patches.txt         hsi/<id>.tif mt/<id>.tif lbl/<id>.tif mt_lbl/<id>.tif
    <out>/metadata.json

Subcommands
-----------
``info``
    Print sample counts, block distribution, raw shapes/dtypes and label values.

``build``
    Resize every sample to 128 x 128 and write the TIFF data set.  ``--dry-run``
    prints the sample count and estimated output size.  ``--limit`` and
    ``--skip-existing`` are useful for smoke tests and resuming.

``verify``
    Check the output list, file existence, shapes and dtypes.

``split``
    Write a patch-level random ``train.txt`` / ``val.txt`` split, or a
    block-wise split with ``--group-by block`` (safer against spatial leakage).

``stats``
    Compute per-channel mean/std in the C2Seg ``data_sta.py`` style: per-patch
    mean/std first, then average across patches.  For Sentinel-2 the flat
    310-channel statistics are also written as 31 x 10 arrays.

Examples
--------
    python tools/h2sr_tools.py info

    python tools/h2sr_tools.py build --dry-run

    python tools/h2sr_tools.py build --limit 3 --out Z:/.../WHU_H2SR_MT_128_test

    python tools/h2sr_tools.py build --skip-existing

    python tools/h2sr_tools.py verify

    python tools/h2sr_tools.py split --val-ratio 0.1 --seed 1919810

    python tools/h2sr_tools.py stats --lists <out>/train.txt <out>/val.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

try:  # OpenCV is by far the fastest option for 249/310-channel arrays.
    import cv2

    cv2.setNumThreads(1)  # keep the tool from saturating every CPU core
except ImportError:  # pragma: no cover - PIL fallback
    cv2 = None
    from PIL import Image

DEFAULT_ROOT = Path(r"Z:/huilin/backup/data_backup/WHU_H2SR_MT")
DEFAULT_OUT = Path(r"Z:/huilin/backup/data_backup/WHU_H2SR_MT_128")

SRC_SIZE = 300
OUT_SIZE = 128
HSI_BANDS = 249
MT_DATES = 31
MT_BANDS = 10
MT_CHANNELS = MT_DATES * MT_BANDS
NUM_CLASSES = 8

CLASS_NAMES: tuple[str, ...] = (
    "paddy field",
    "dry farmland",
    "forest land",
    "grassland",
    "building",
    "highway",
    "greenhouse",
    "water body",
)

# The release uses 15 for unlabeled/background; some labels may already use 255.
SRC_IGNORE_VALUES: tuple[int, ...] = (15, 255)
IGNORE_INDEX = 255


@dataclass(frozen=True)
class Sample:
    block: int
    stem: str
    hsi: Path
    label: Path
    mt: Path
    mt_label: Path

    @property
    def sample_id(self) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.stem)
        return f"b{self.block:02d}_{stem}"


def _parse_block(name: str, prefix: str) -> int | None:
    match = re.fullmatch(rf"{prefix}_(\d+)", name)
    return int(match.group(1)) if match else None


def discover_samples(
    root: Path,
    allow_missing: bool = False,
) -> tuple[list[Sample], dict]:
    """Match HSI, HSI labels, MT images and MT labels by file stem.

    Block membership is inferred from ``H2SR_label_<b>``: the release's HSI
    files for blocks 1-10 are at the root, while block 11 is in
    ``H2SR_image_11``.  A stem must occur in exactly one label folder.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    label_sets: dict[int, set[str]] = {}
    for directory in sorted(root.glob("H2SR_label_*")):
        block = _parse_block(directory.name, "H2SR_label")
        if block is None:
            continue
        stems = {p.stem for p in directory.glob("*.tif")}
        if stems:
            label_sets[block] = stems
    if not label_sets:
        raise FileNotFoundError(f"no H2SR_label_* folders under {root}")

    stem_to_blocks: dict[str, list[int]] = defaultdict(list)
    for block, stems in label_sets.items():
        for stem in stems:
            stem_to_blocks[stem].append(block)

    hsi_paths: list[Path] = [p for p in root.glob("*.tif") if p.is_file()]
    for directory in sorted(root.glob("H2SR_image_*")):
        if directory.is_dir():
            hsi_paths.extend(p for p in directory.glob("*.tif") if p.is_file())

    diagnostics: dict = {
        "label_blocks": {b: len(stems) for b, stems in sorted(label_sets.items())},
        "hsi_candidates": len(hsi_paths),
        "orphan_hsi": [],
        "ambiguous_hsi": [],
        "missing_files": [],
        "duplicate_hsi": [],
    }

    seen_hsi: set[tuple[int, str]] = set()
    samples: list[Sample] = []
    for hsi in sorted(hsi_paths):
        stem = hsi.stem
        blocks = stem_to_blocks.get(stem, [])
        if not blocks:
            diagnostics["orphan_hsi"].append(str(hsi))
            continue
        if len(blocks) > 1:
            diagnostics["ambiguous_hsi"].append((str(hsi), blocks))
            continue
        block = blocks[0]
        key = (block, stem)
        if key in seen_hsi:
            diagnostics["duplicate_hsi"].append(str(hsi))
            continue
        seen_hsi.add(key)

        label = root / f"H2SR_label_{block}" / f"{stem}.tif"
        mt = root / f"MT_image_{block}" / f"{stem}.npy"
        mt_label = root / f"MT_label_{block}" / f"{stem}.tif"
        missing = [str(p) for p in (label, mt, mt_label) if not p.exists()]
        if missing:
            diagnostics["missing_files"].append({"hsi": str(hsi), "missing": missing})
            continue
        samples.append(
            Sample(block=block, stem=stem, hsi=hsi, label=label, mt=mt, mt_label=mt_label)
        )

    samples.sort(key=lambda s: (s.block, s.stem))
    diagnostics["samples"] = len(samples)
    diagnostics["blocks"] = dict(sorted(Counter(s.block for s in samples).items()))
    return samples, diagnostics


def load_hsi(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.shape != (SRC_SIZE, SRC_SIZE, HSI_BANDS):
        raise ValueError(f"{path}: expected {(SRC_SIZE, SRC_SIZE, HSI_BANDS)}, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def load_mt(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    if array.shape == (MT_DATES, MT_BANDS, 30, 30):
        # date-major, band-minor -> channel-last (H, W, date * band)
        array = np.transpose(array, (2, 3, 0, 1)).reshape(30, 30, MT_CHANNELS)
    elif array.shape == (30, 30, MT_DATES, MT_BANDS):
        array = array.reshape(30, 30, MT_CHANNELS)
    else:
        raise ValueError(
            f"{path}: expected {(MT_DATES, MT_BANDS, 30, 30)} or "
            f"{(30, 30, MT_DATES, MT_BANDS)}, got {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=np.float32)


def load_label(path: Path, shape: tuple[int, int]) -> np.ndarray:
    array = tifffile.imread(path)
    if array.shape != shape:
        raise ValueError(f"{path}: expected {shape}, got {array.shape}")
    label = array.astype(np.uint8, copy=True)
    label[np.isin(label, SRC_IGNORE_VALUES)] = IGNORE_INDEX
    valid = label[label != IGNORE_INDEX]
    if valid.size:
        bad = sorted({int(v) for v in np.unique(valid) if int(v) >= NUM_CLASSES})
        if bad:
            raise ValueError(
                f"{path}: class ids {bad} outside 0..{NUM_CLASSES - 1}; "
                "update NUM_CLASSES/class mapping if the release changed"
            )
    return label


def resize_array(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    """Resize an (H, W) or (H, W, C) array to ``size x size``."""
    if array.shape[0] == size and array.shape[1] == size:
        return array
    if cv2 is not None:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(array, (size, size), interpolation=interpolation)

    # Fallback: PIL, one channel at a time.  Slower but dependency-light.
    resample = Image.NEAREST if nearest else Image.BILINEAR
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array).resize((size, size), resample=resample))
    out = np.empty((size, size, array.shape[2]), dtype=array.dtype)
    for channel in range(array.shape[2]):
        band = array[..., channel]
        mode = "F" if band.dtype == np.float32 else "L"
        out[..., channel] = np.asarray(
            Image.fromarray(band, mode=mode).resize((size, size), resample=resample)
        )
    return out


def _write_tiff(path: Path, array: np.ndarray, compression: str, level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"photometric": "minisblack", "metadata": None}
    if array.ndim in (2, 3):
        # Keep [H, W, C] as one multi-sample page (same style as the raw HSI),
        # rather than one page per band.
        kwargs["planarconfig"] = "contig"
    elif array.ndim == 4:
        kwargs["metadata"] = {"axes": "TBYX"}
    if compression != "none":
        kwargs["compression"] = compression
        if compression == "zlib":
            kwargs["compressionargs"] = {"level": level}
    tifffile.imwrite(path, array, **kwargs)


def _estimate_bytes(size: int, samples: int) -> dict:
    hsi = size * size * HSI_BANDS * 4
    mt = size * size * MT_CHANNELS * 4
    label = size * size
    total = (hsi + mt + 2 * label) * samples
    return {"per_sample": hsi + mt + 2 * label, "hsi": hsi, "mt": mt, "total": total}


def _class_stats(hist: np.ndarray) -> dict:
    total = int(hist.sum())
    classes = {
        name: {
            "class_id": i,
            "pixels": int(hist[i]),
            "percent": (100.0 * float(hist[i]) / total) if total else 0.0,
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    return {
        "total_pixels": total,
        "classes": classes,
        "ignore": {
            "class_id": IGNORE_INDEX,
            "pixels": int(hist[IGNORE_INDEX]),
            "percent": (100.0 * float(hist[IGNORE_INDEX]) / total) if total else 0.0,
        },
    }


def cmd_info(args: argparse.Namespace) -> int:
    samples, diagnostics = discover_samples(args.root, allow_missing=True)
    print(f"root: {args.root}")
    print(f"samples: {len(samples)}")
    print(f"blocks: {diagnostics['blocks']}")
    for key in ("orphan_hsi", "ambiguous_hsi", "missing_files", "duplicate_hsi"):
        value = diagnostics[key]
        if value:
            print(f"warning: {key} = {len(value)}; first: {value[0]}")

    if samples:
        sample = samples[0]
        print(f"\nfirst sample: {sample.sample_id}")
        print(f"  HSI   {sample.hsi}")
        print(f"  label {sample.label}")
        print(f"  MT    {sample.mt}")
        print(f"  MT lbl{sample.mt_label}")
        with tifffile.TiffFile(sample.hsi) as tif:
            series = tif.series[0]
            print(f"  HSI shape={series.shape} dtype={series.dtype}")
        with tifffile.TiffFile(sample.label) as tif:
            series = tif.series[0]
            print(f"  HSI label shape={series.shape} dtype={series.dtype}")
        mt = np.load(sample.mt, mmap_mode="r")
        print(f"  MT shape={mt.shape} dtype={mt.dtype}")
        with tifffile.TiffFile(sample.mt_label) as tif:
            series = tif.series[0]
            print(f"  MT label shape={series.shape} dtype={series.dtype}")
        label = tifffile.imread(sample.label)
        values, counts = np.unique(label, return_counts=True)
        print(f"  HSI label values: {dict(zip(values.tolist(), counts.tolist()))}")
        mt_label = tifffile.imread(sample.mt_label)
        values, counts = np.unique(mt_label, return_counts=True)
        print(f"  MT label values : {dict(zip(values.tolist(), counts.tolist()))}")

    if args.full_stats:
        hist_hsi = np.zeros(256, dtype=np.int64)
        hist_mt = np.zeros(256, dtype=np.int64)
        print("\nreading all labels ...")
        for index, sample in enumerate(samples, 1):
            hist_hsi += np.bincount(tifffile.imread(sample.label).ravel(), minlength=256)
            hist_mt += np.bincount(tifffile.imread(sample.mt_label).ravel(), minlength=256)
            if index % 500 == 0:
                print(f"  {index}/{len(samples)}")
        print("HSI labels:", json.dumps(_class_stats(hist_hsi), indent=2, ensure_ascii=False))
        print("MT labels :", json.dumps(_class_stats(hist_mt), indent=2, ensure_ascii=False))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    samples, diagnostics = discover_samples(args.root, allow_missing=args.allow_missing)
    if not samples:
        raise RuntimeError("no complete samples found")
    if diagnostics["missing_files"] and not args.allow_missing:
        raise RuntimeError(
            f"{len(diagnostics['missing_files'])} samples are missing files; "
            "pass --allow-missing to skip them"
        )

    estimate = _estimate_bytes(args.size, len(samples))
    print(f"root    : {args.root}")
    print(f"out     : {args.out}")
    print(f"samples : {len(samples)}  blocks={diagnostics['blocks']}")
    print(
        f"output  : {args.size}x{args.size}  "
        f"HSI={args.size}x{args.size}x{HSI_BANDS} float32  "
        f"MT={args.size}x{args.size}x{MT_CHANNELS} float32"
    )
    print(
        f"estimate: {estimate['total']/1e9:.2f} GB total, "
        f"{estimate['per_sample']/1e6:.1f} MB/sample"
    )
    if args.dry_run:
        print("[dry-run] no files written")
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "patches.txt").exists() and not (args.overwrite or args.skip_existing):
        raise FileExistsError(
            f"{out / 'patches.txt'} already exists; use --overwrite or --skip-existing"
        )
    for sub in ("hsi", "mt", "lbl", "mt_lbl"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    hist_hsi = np.zeros(256, dtype=np.int64)
    hist_mt = np.zeros(256, dtype=np.int64)
    written = 0
    skipped = 0
    hsi_min = hsi_max = None
    mt_min = mt_max = None
    t0 = time.time()
    total = min(len(samples), args.limit) if args.limit else len(samples)

    for index, sample in enumerate(samples[:total], 1):
        sample_id = sample.sample_id
        paths = {
            "hsi": out / "hsi" / f"{sample_id}.tif",
            "mt": out / "mt" / f"{sample_id}.tif",
            "lbl": out / "lbl" / f"{sample_id}.tif",
            "mt_lbl": out / "mt_lbl" / f"{sample_id}.tif",
        }
        line = " ".join(
            paths[key].relative_to(out).as_posix()
            for key in ("hsi", "mt", "lbl", "mt_lbl")
        )
        if args.skip_existing and all(p.exists() and p.stat().st_size > 0 for p in paths.values()):
            lines.append(line)
            skipped += 1
            # Keep the metadata histograms complete on a resumed run.
            hist_hsi += np.bincount(tifffile.imread(paths["lbl"]).ravel(), minlength=256)
            hist_mt += np.bincount(tifffile.imread(paths["mt_lbl"]).ravel(), minlength=256)
            if index % 100 == 0:
                print(f"  {index}/{total} (skipped {skipped})")
            continue

        try:
            hsi = load_hsi(sample.hsi)
            label = load_label(sample.label, (SRC_SIZE, SRC_SIZE))
            mt = load_mt(sample.mt)
            mt_label = load_label(sample.mt_label, (30, 30))

            hsi_out = resize_array(hsi, args.size, nearest=False)
            mt_out = resize_array(mt, args.size, nearest=False)
            lbl_out = resize_array(label, args.size, nearest=True)
            mt_lbl_out = resize_array(mt_label, args.size, nearest=True)

            if args.mt_layout == "channels":
                mt_array = np.ascontiguousarray(mt_out, dtype=np.float32)
            else:  # dates: dates as the first axes, useful for temporal models
                mt_array = (
                    mt_out.reshape(args.size, args.size, MT_DATES, MT_BANDS)
                    .transpose(2, 3, 0, 1)
                )

            _write_tiff(paths["hsi"], hsi_out, args.compression, args.compression_level)
            _write_tiff(paths["mt"], mt_array, args.compression, args.compression_level)
            _write_tiff(paths["lbl"], lbl_out.astype(np.uint8), args.compression, args.compression_level)
            _write_tiff(
                paths["mt_lbl"],
                mt_lbl_out.astype(np.uint8),
                args.compression,
                args.compression_level,
            )
        except Exception as exc:
            raise RuntimeError(f"failed on {sample_id}: {exc}") from exc

        lines.append(line)
        hist_hsi += np.bincount(lbl_out.ravel(), minlength=256)
        hist_mt += np.bincount(mt_lbl_out.ravel(), minlength=256)
        h_min, h_max = float(hsi_out.min()), float(hsi_out.max())
        m_min, m_max = float(mt_out.min()), float(mt_out.max())
        hsi_min = h_min if hsi_min is None else min(hsi_min, h_min)
        hsi_max = h_max if hsi_max is None else max(hsi_max, h_max)
        mt_min = m_min if mt_min is None else min(mt_min, m_min)
        mt_max = m_max if mt_max is None else max(mt_max, m_max)
        written += 1
        if index % 25 == 0 or index == total:
            elapsed = time.time() - t0
            rate = index / elapsed if elapsed else 0.0
            print(
                f"  {index}/{total} written={written} skipped={skipped} "
                f"({rate:.2f} samples/s, {elapsed/60:.1f} min)"
            )

    (out / "patches.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "tool": "tools/h2sr_tools.py build",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "out": str(out),
        "source": {
            "patch_size": SRC_SIZE,
            "hsi_bands": HSI_BANDS,
            "mt_dates": MT_DATES,
            "mt_bands": MT_BANDS,
            "mt_source_shape": [MT_DATES, MT_BANDS, 30, 30],
        },
        "output": {
            "size": args.size,
            "hsi_shape": [args.size, args.size, HSI_BANDS],
            "mt_shape": (
                [args.size, args.size, MT_CHANNELS]
                if args.mt_layout == "channels"
                else [MT_DATES, MT_BANDS, args.size, args.size]
            ),
            "lbl_shape": [args.size, args.size],
            "mt_lbl_shape": [args.size, args.size],
            "mt_layout": args.mt_layout,
            "mt_channel_order": (
                "date-major: c = date * 10 + band"
                if args.mt_layout == "channels"
                else "dates-first: [date, band, y, x]"
            ),
        },
        "resize": {
            "images": "bilinear",
            "labels": "nearest",
            "hsi": f"{SRC_SIZE}x{SRC_SIZE} -> {args.size}x{args.size}",
            "mt": f"30x30 -> {args.size}x{args.size}",
        },
        "labels": {
            "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES),
            "source_ignore_values": list(SRC_IGNORE_VALUES),
            "ignore_index": IGNORE_INDEX,
        },
        "compression": args.compression,
        "counts": {
            "samples": len(samples),
            "written": written,
            "skipped": skipped,
            "limited_to": args.limit,
            "by_block": diagnostics["blocks"],
        },
        "class_pixels_hsi": _class_stats(hist_hsi),
        "class_pixels_mt": _class_stats(hist_mt),
        "value_ranges": {"hsi": [hsi_min, hsi_max], "mt": [mt_min, mt_max]},
        "list_file": "patches.txt",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nwritten {written} samples, skipped {skipped}, list={out / 'patches.txt'}")
    print(f"metadata -> {out / 'metadata.json'}")
    return 0


def _read_patch_list(path: Path) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    for line in lines:
        if len(line.split()) != 4:
            raise ValueError(f"expected four columns in {path}: {line!r}")
    return lines


def _label_stats(lines: list[str], root: Path) -> dict:
    hist = np.zeros(256, dtype=np.int64)
    ratios: list[float] = []
    for line in lines:
        label = tifffile.imread(root / line.split()[2])
        hist += np.bincount(label.ravel(), minlength=256)
        ratios.append(float((label != IGNORE_INDEX).mean()))
    total = int(hist.sum())
    classes = {
        name: {
            "class_id": i,
            "pixels": int(hist[i]),
            "percent": (100.0 * float(hist[i]) / total) if total else 0.0,
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    return {
        "patches": len(lines),
        "pixels": total,
        "valid_pixels": int(total - hist[IGNORE_INDEX]),
        "valid_percent": (100.0 * float(total - hist[IGNORE_INDEX]) / total) if total else 0.0,
        "ignore_pixels": int(hist[IGNORE_INDEX]),
        "classes": classes,
        "valid_ratio_patch": {
            "min": min(ratios) if ratios else None,
            "mean": float(np.mean(ratios)) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
    }


def cmd_verify(args: argparse.Namespace) -> int:
    patches = Path(args.patches)
    lines = _read_patch_list(patches)
    root = patches.parent
    errors: list[str] = []
    bad_values: Counter = Counter()
    for index, line in enumerate(lines, 1):
        parts = line.split()
        for rel, expected in zip(
            parts,
            (
                (OUT_SIZE, OUT_SIZE, HSI_BANDS),
                (OUT_SIZE, OUT_SIZE, MT_CHANNELS),
                (OUT_SIZE, OUT_SIZE),
                (OUT_SIZE, OUT_SIZE),
            ),
        ):
            path = root / rel
            if not path.exists():
                errors.append(f"missing: {path}")
                continue
            try:
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    shape = series.shape
                    dtype = series.dtype
                if "mt/" in rel:
                    allowed = {
                        (OUT_SIZE, OUT_SIZE, MT_CHANNELS),
                        (MT_DATES, MT_BANDS, OUT_SIZE, OUT_SIZE),
                    }
                    if shape not in allowed:
                        errors.append(f"shape {shape} not in {sorted(allowed)}: {path}")
                elif shape != expected:
                    errors.append(f"shape {shape} != {expected}: {path}")
                if "lbl" in rel and dtype != np.uint8:
                    errors.append(f"label dtype {dtype} != uint8: {path}")
            except Exception as exc:
                errors.append(f"unreadable {path}: {exc}")
        if args.full and (root / parts[2]).exists():
            label = tifffile.imread(root / parts[2])
            bad_values.update(np.unique(label).tolist())
        if index % 500 == 0:
            print(f"  checked {index}/{len(lines)}")
    if args.full:
        invalid = {v: n for v, n in bad_values.items() if v not in range(NUM_CLASSES) and v != IGNORE_INDEX}
        if invalid:
            errors.append(f"unexpected label values: {invalid}")
    print(f"checked {len(lines)} samples, errors={len(errors)}")
    for error in errors[:20]:
        print("  ", error)
    if len(errors) > 20:
        print(f"  ... {len(errors) - 20} more")
    return 1 if errors else 0


def _block_of(line: str) -> int:
    match = re.match(r"hsi/b(\d+)_", line)
    if not match:
        raise ValueError(f"cannot parse block from {line!r}")
    return int(match.group(1))


def cmd_split(args: argparse.Namespace) -> int:
    patches = Path(args.patches)
    lines = _read_patch_list(patches)
    root = patches.parent

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.val_blocks:
        args.group_by = "block"
    if args.min_valid_ratio > 0:
        kept = []
        for line in lines:
            label = tifffile.imread(root / line.split()[2])
            if float((label != IGNORE_INDEX).mean()) >= args.min_valid_ratio:
                kept.append(line)
        print(f"filtered {len(lines) - len(kept)} patches by valid ratio")
        lines = kept
    if len(lines) < 2:
        raise ValueError("need at least two patches")

    if args.group_by == "block":
        groups: dict[int, list[str]] = defaultdict(list)
        for line in lines:
            groups[_block_of(line)].append(line)
        blocks = sorted(groups)
        if args.val_blocks:
            val_blocks = sorted({int(x) for x in args.val_blocks.split(",") if x.strip()})
            unknown = [b for b in val_blocks if b not in groups]
            if unknown:
                raise ValueError(f"--val-blocks contains unknown blocks: {unknown}")
            if not val_blocks or len(val_blocks) == len(blocks):
                raise ValueError("--val-blocks must leave at least one training block")
            style = f"block-wise fixed (val blocks {val_blocks})"
        else:
            rng = np.random.default_rng(args.seed)
            rng.shuffle(blocks)
            target = int(round(len(lines) * args.val_ratio))
            val_blocks = []
            count = 0
            for block in blocks[:-1]:
                if count >= target:
                    break
                val_blocks.append(block)
                count += len(groups[block])
            if not val_blocks:
                val_blocks = [blocks[0]]
            style = f"block-wise random (val blocks {sorted(val_blocks)})"
        train_lines = [line for b in blocks if b not in val_blocks for line in groups[b]]
        val_lines = [line for b in blocks if b in val_blocks for line in groups[b]]
    else:
        n_val = max(1, int(round(len(lines) * args.val_ratio)))
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(lines))
        val_idx = set(int(i) for i in perm[:n_val])
        train_lines = [line for i, line in enumerate(lines) if i not in val_idx]
        val_lines = [line for i, line in enumerate(lines) if i in val_idx]
        style = "patch-level random"

    train_stats = _label_stats(train_lines, root)
    val_stats = _label_stats(val_lines, root)
    print(
        f"{style}: {len(lines)} -> train {len(train_lines)} / val {len(val_lines)} "
        f"(seed={args.seed}, val_ratio={args.val_ratio:g})"
    )
    missing = [
        name
        for name in CLASS_NAMES
        if train_stats["classes"][name]["pixels"] == 0 or val_stats["classes"][name]["pixels"] == 0
    ]
    if missing:
        print(f"warning: classes missing from one split: {', '.join(missing)}")

    meta = {
        "tool": "tools/h2sr_tools.py split",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patches": str(patches),
        "split_style": style,
        "group_by": args.group_by,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "min_valid_ratio": args.min_valid_ratio,
        "total_patches": len(lines),
        "train_patches": len(train_lines),
        "val_patches": len(val_lines),
        "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "train": train_stats,
        "val": val_stats,
    }

    if args.dry_run:
        print("[dry-run] no split files written")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    meta_path = out_dir / "split_metadata.json"
    if not args.overwrite:
        existing = [p for p in (train_path, val_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "split files exist (use --overwrite): " + ", ".join(str(p) for p in existing)
            )
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"written: {train_path} ({len(train_lines)})")
    print(f"written: {val_path} ({len(val_lines)})")
    print(f"written: {meta_path}")
    return 0


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def cmd_stats(args: argparse.Namespace) -> int:
    list_paths = [Path(p) for p in args.lists]
    if not list_paths:
        default_out = Path(args.out)
        candidates = [default_out / "train.txt", default_out / "val.txt"]
        list_paths = [p for p in candidates if p.exists()]
        if not list_paths:
            candidate = default_out / "patches.txt"
            if not candidate.exists():
                raise FileNotFoundError("no --lists and no default train/val/patches.txt")
            list_paths = [candidate]
    for path in list_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    roots = {p.parent for p in list_paths}
    if len(roots) != 1:
        raise ValueError("all --lists files must live in the same directory")
    root = next(iter(roots))

    lines: list[str] = []
    seen: set[str] = set()
    for path in list_paths:
        for line in _read_patch_list(path):
            if line not in seen:
                seen.add(line)
                lines.append(line)
    if args.max_samples:
        lines = lines[: args.max_samples]
    if not lines:
        raise ValueError("no samples to process")

    hsi_means: list[np.ndarray] = []
    hsi_stds: list[np.ndarray] = []
    mt_means: list[np.ndarray] = []
    mt_stds: list[np.ndarray] = []
    t0 = time.time()
    for index, line in enumerate(lines, 1):
        parts = line.split()
        hsi = tifffile.imread(root / parts[0]).astype(np.float64)
        mt = tifffile.imread(root / parts[1]).astype(np.float64)
        if mt.ndim == 4:  # optional dates-first layout: [31, 10, H, W]
            mt = mt.transpose(2, 3, 0, 1).reshape(mt.shape[2], mt.shape[3], MT_CHANNELS)
        if args.mask == "ignore":
            label = tifffile.imread(root / parts[2])
            mt_label = tifffile.imread(root / parts[3])
            valid = label != IGNORE_INDEX
            valid_mt = mt_label != IGNORE_INDEX
            if not valid.any() or not valid_mt.any():
                continue
            hsi = hsi[valid]
            mt = mt[valid_mt]
        else:
            hsi = hsi.reshape(-1, hsi.shape[-1])
            mt = mt.reshape(-1, mt.shape[-1])
        hsi_means.append(hsi.mean(axis=0))
        hsi_stds.append(hsi.std(axis=0))
        mt_means.append(mt.mean(axis=0))
        mt_stds.append(mt.std(axis=0))
        if index % 50 == 0:
            print(f"  stats {index}/{len(lines)} ({time.time()-t0:.0f}s)")

    if not hsi_means:
        raise ValueError("no usable pixels after masking")
    mean_hsi = np.mean(np.stack(hsi_means), axis=0)
    std_hsi = np.mean(np.stack(hsi_stds), axis=0)
    mean_mt = np.mean(np.stack(mt_means), axis=0)
    std_mt = np.mean(np.stack(mt_stds), axis=0)

    meta = {
        "tool": "tools/h2sr_tools.py stats",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "C2Seg data_sta.py Col_avg: per-patch mean/std, then average across patches",
        "mask": args.mask,
        "list_files": [str(p) for p in list_paths],
        "num_patches": len(hsi_means),
        "hsi_bands": HSI_BANDS,
        "mt_dates": MT_DATES,
        "mt_bands": MT_BANDS,
        "mean_hsi": mean_hsi.tolist(),
        "std_hsi": std_hsi.tolist(),
        "mean_mt": mean_mt.tolist(),
        "std_mt": std_mt.tolist(),
        "mean_mt_by_date_band": mean_mt.reshape(MT_DATES, MT_BANDS).tolist(),
        "std_mt_by_date_band": std_mt.reshape(MT_DATES, MT_BANDS).tolist(),
    }
    stats_path = Path(args.stats_json) if args.stats_json else root / "normalize_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"C2Seg-style Col_avg over {len(hsi_means)} patches (mask={args.mask})")
    print(f"mean_hsi[:5]={_fmt(mean_hsi[:5])} ... [-5:]={_fmt(mean_hsi[-5:])}")
    print(f"mean_mt [:5]={_fmt(mean_mt[:5])} ... [-5:]={_fmt(mean_mt[-5:])}")
    print(f"written: {stats_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="inspect the raw data set")
    info.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    info.add_argument("--full-stats", action="store_true", help="read every label and print histograms")
    info.set_defaults(func=cmd_info)

    build = sub.add_parser("build", help="resize every sample to 128x128 TIFFs")
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build.add_argument("--size", type=int, default=OUT_SIZE)
    build.add_argument(
        "--mt-layout",
        choices=("channels", "dates"),
        default="channels",
        help="channels = [H,W,310] (default); dates = [31,10,H,W]",
    )
    build.add_argument("--compression", choices=("none", "zlib", "lzw"), default="none")
    build.add_argument("--compression-level", type=int, default=1)
    build.add_argument("--limit", type=int, default=None, help="write only the first N samples")
    build.add_argument("--skip-existing", action="store_true", help="resume: skip complete sample files")
    build.add_argument("--overwrite", action="store_true", help="allow an existing output list")
    build.add_argument("--allow-missing", action="store_true", help="skip samples with missing files")
    build.add_argument("--dry-run", action="store_true", help="print counts/estimate only")
    build.set_defaults(func=cmd_build)

    verify = sub.add_parser("verify", help="check output shapes/dtypes")
    verify.add_argument("--patches", type=Path, default=DEFAULT_OUT / "patches.txt")
    verify.add_argument("--full", action="store_true", help="also read labels and check values")
    verify.set_defaults(func=cmd_verify)

    split = sub.add_parser("split", help="write train/val split lists")
    split.add_argument("--patches", type=Path, default=DEFAULT_OUT / "patches.txt")
    split.add_argument("--out-dir", type=Path, default=None)
    split.add_argument("--val-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=1919810)
    split.add_argument("--group-by", choices=("patch", "block"), default="patch")
    split.add_argument(
        "--val-blocks",
        default=None,
        help="comma-separated block ids for validation (implies --group-by block)",
    )
    split.add_argument("--min-valid-ratio", type=float, default=0.0)
    split.add_argument("--overwrite", action="store_true")
    split.add_argument("--dry-run", action="store_true")
    split.set_defaults(func=cmd_split)

    stats = sub.add_parser("stats", help="compute per-channel mean/std")
    stats.add_argument("--lists", nargs="+", type=Path, default=None)
    stats.add_argument("--out", type=Path, default=DEFAULT_OUT, help="default output root")
    stats.add_argument("--mask", choices=("none", "ignore"), default="none")
    stats.add_argument("--max-samples", type=int, default=None)
    stats.add_argument("--stats-json", type=Path, default=None)
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
