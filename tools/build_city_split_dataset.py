"""Build a city-level train/val split dataset from the two C2Seg-BW full scenes.

Motivation
----------
The default random patch split can leak geography between train/val: patches
cropped from the same full scene (Beijing or Wuhan) land in both sets, so nearby
spatial locations are shared. To address the reviewer concern, this script
builds a **geographically disjoint** split:

    train  <- patches cropped from one city's full scene
    val    <- patches cropped from the other city's full scene

The city pairing is chosen with a single ``--split`` direction (default
``train_B_val_W`` = train on Beijing, validate on Wuhan):

    python tools/build_city_split_dataset.py --split train_B_val_W
    python tools/build_city_split_dataset.py --split train_W_val_B

The full-scene TIFFs are located automatically from ``C2SEG_BW_ROOT`` in
``.env`` (they live in ``.../C2Seg/src/tif_BW``), so no scene root needs to be
passed. When ``C2SEG_CITY_ROOT`` is set in ``.env``, output defaults to
``C2SEG_CITY_ROOT/C2SEG_<SPLIT>`` (for example,
``C2SEG_CITY_ROOT/C2SEG_TRAIN_B_VAL_W``); otherwise it falls back to
``data/C2Seg_BW_city_<split>``.

Output layout (matches ``RS_MD3B`` + ``Normalize2`` used by the PaddleCD configs):

    <output>/
      train/msisar/<id>.tiff    6 ch  (MSI 4 + SAR 2)             uint16
      train/hsi/<id>.tiff     116 ch                              uint16
      train/lbl/<id>.tiff       1 ch   semantic label              uint8
      train.txt                 "msi/<id>.tiff sar/<id>.tiff lbl/<id>.tiff"
      val/                      identical layout from the other city
      val.txt
      metadata.json             split settings + per-set statistics

``RS_MD3B`` expands ``items[0].replace('msi', 'msisar')`` and
``items[1].replace('sar', 'hsi')`` before joining to ``dataset_root``, so the
txt columns keep the ``msi/`` ``sar/`` ``lbl/`` folder names while the files on
disk live under ``msisar/`` ``hsi/`` ``lbl/``.

The full-scene TIFFs are channel-first ``[C, H, W]`` and live next to the
official MAT sources (see ``convert_c2seg_full_mat_to_tif.py``). Patch TIFFs are
written channel-last ``[H, W, C]`` to match how ``skimage.io.imread`` returns the
official C2Seg-BW patches. Full-scene HSI is 0-1 reflectance; it is rescaled by
10000 to the DN range (mean ~ 900-1000) the Normalize2 stats were computed on.

Example
-------
    # default: train Beijing, validate Wuhan (scene root from .env)
    python tools/build_city_split_dataset.py

    # swap the direction (train Wuhan, validate Beijing)
    python tools/build_city_split_dataset.py --split train_W_val_B

    # keep all patches (no nodata filter), 512 stride
    python tools/build_city_split_dataset.py --valid-ratio 0 --stride 512 512
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import cv2

try:
    import tifffile
except ImportError:  # pragma: no cover - experiment box only
    tifffile = None

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

NUM_CLASSES = 14  # C2Seg-BW classes are 0..13 (Background is a real class)

LABEL_NAMES = [
    "Background", "Surface water", "Street", "Urban Fabric",
    "Industrial, commercial and transport", "Mine, dump, and construction sites",
    "Artificial, vegetated areas", "Arable Land", "Permanent Crops", "Pastures",
    "Forests", "Shrub", "Open spaces with no vegetation", "Inland wetlands",
]

# Geographic direction -> (train_scene, val_scene).  B = Beijing, W = Wuhan.
SPLITS = {
    "train_B_val_W": ("beijing", "wuhan"),
    "train_W_val_B": ("wuhan", "beijing"),
}


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


# --------------------------------------------------------------------------- #
# Path discovery
# --------------------------------------------------------------------------- #

def _candidates(*parts: str) -> list[str]:
    return ["_".join(parts), "".join(parts)]


def find_scene_file(scene_root: Path, scene: str, kind: str,
                    suffixes=(".tif", ".tiff", ".TIF", ".TIFF")) -> Path | None:
    """Find ``Beijing_MSI.tif``-style files with loose case/order matching."""
    scene_variants = {scene, scene.lower(), scene.capitalize(), scene.upper()}
    kind_variants = []
    if kind == "msi":
        kind_variants = ["msi", "MSI"]
    elif kind == "sar":
        kind_variants = ["sar", "SAR"]
    elif kind == "hsi":
        kind_variants = ["hsi", "HSI"]
    elif kind == "label":
        kind_variants = ["label_mod5",  # official UCMerced-style key; unlikely, harmless
                         "label", "label_rawcode", "lbl", "labels", "gt"]
    for s in scene_variants:
        for k in kind_variants:  # noqa: SIM110
            for name in _candidates(s, k):
                for suffix in suffixes:
                    cand = scene_root / f"{name}{suffix}"
                    if cand.is_file():
                        return cand
    return None


def resolve_scene_root(cli_root: str, env: dict[str, str]) -> Path:
    if cli_root:
        return Path(cli_root)
    bw_root = env.get("C2SEG_BW_ROOT", "")
    if bw_root:
        # C2SEG_BW_ROOT = .../C2Seg/src/C2Seg_BW ; sources live in .../C2Seg/src/tif_BW
        return Path(bw_root).parent / "tif_BW"
    return Path("")


def resolve_output_root(cli_output: str, env: dict[str, str], split: str) -> Path:
    """Resolve the output directory for a city split.

    ``C2SEG_CITY_ROOT`` is a parent directory. The split name is converted to
    the canonical dataset directory name automatically, so the two supported
    splits become ``C2SEG_TRAIN_B_VAL_W`` and ``C2SEG_TRAIN_W_VAL_B``.
    An explicit ``--output`` always takes precedence.
    """
    if cli_output:
        return Path(cli_output)

    city_root = env.get("C2SEG_CITY_ROOT", "") or os.environ.get("C2SEG_CITY_ROOT", "")
    city_root = city_root.strip().strip('"').strip("'")
    if city_root:
        return Path(city_root) / f"C2SEG_{split.upper()}"

    return REPO_ROOT / "data" / f"C2Seg_BW_city_{split}"


# --------------------------------------------------------------------------- #
# Block reader
# --------------------------------------------------------------------------- #

def channel_axis(shape: tuple[int, ...]) -> int:
    """Return the channel axis of a 3D stack.

    The converted full-scene TIFFs are channel-first ``[C, H, W]``. Layout is
    detected by looking for the axis that is small and smaller than the other
    two; square fallback images (H == W) default to CHW, the documented layout.
    """
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D image stack, got shape={shape}")
    h, w, c = shape
    if h <= 512 and h < w and h < c:   # [C, H, W]
        return 0
    if c <= 512 and c < h:             # [H, W, C]
        return 2
    return 0


class TiffReader:
    """Memmap-backed reader for one full-scene TIFF stack (read-only)."""

    def __init__(self, path: Path, name: str):
        if tifffile is None:
            raise RuntimeError("tifffile is required. pip install tifffile")
        self.path = path
        self.name = name
        try:
            self.array = tifffile.memmap(path)
        except ValueError:
            self.array = tifffile.imread(path)
        self.shape = tuple(int(v) for v in self.array.shape)
        if len(self.shape) == 2:
            self.axis = None
            self.count, self.height, self.width = 1, self.shape[0], self.shape[1]
        else:
            self.axis = channel_axis(self.shape)
            if self.axis == 0:
                self.count, self.height, self.width = self.shape
            else:
                self.height, self.width, self.count = self.shape

    def read_patch(self, x: int, y: int, w: int, h: int,
                   bands: list[int] | None = None) -> np.ndarray:
        """Return patch as float32 [C, H, W] (2D input -> [1, H, W])."""
        idx = [b - 1 for b in bands] if bands else None
        if self.axis is None:  # 2D label
            return np.asarray(self.array[y:y + h, x:x + w])[None, ...].astype("float32")
        if self.axis == 0:
            block = np.asarray(self.array[:, y:y + h, x:x + w])
            out = block[idx, :, :] if idx else block
        else:  # axis == 2
            block = np.asarray(self.array[y:y + h, x:x + w, :])
            if idx:
                out = np.transpose(block[:, :, idx], (2, 0, 1))
            else:
                out = np.transpose(block, (2, 0, 1))
        return out.astype("float32", copy=False)

    def close(self) -> None:
        pass


def resize_chw(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a channel-first array to ``(height, width)``."""
    out_h, out_w = out_hw
    if arr.ndim != 3 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError(f"Cannot resize an empty/non-CHW array: shape={arr.shape}")
    hwc = np.transpose(arr, (1, 2, 0))
    resized = cv2.resize(hwc, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.transpose(resized, (2, 0, 1)).astype("float32", copy=False)


def read_hsi_patch(reader: TiffReader, x: int, y: int, w: int, h: int,
                   target_width: int, target_height: int,
                   bands: list[int] | None = None) -> np.ndarray:
    """Read an HSI patch using MSI/SAR coordinates and resize it to target size.

    Full-scene HSI can have a lower spatial resolution than MSI/SAR. Directly
    slicing it with MSI coordinates produces empty strips once the MSI window
    moves beyond the HSI extent. Map the window into HSI coordinates first,
    then resize the sampled HSI block back to the model patch size.
    """
    if reader.width <= 0 or reader.height <= 0:
        raise ValueError(f"HSI source has invalid spatial size: {reader.shape}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"Target scene has invalid spatial size: {target_width}x{target_height}")

    hx0 = int(np.floor(x * reader.width / target_width))
    hy0 = int(np.floor(y * reader.height / target_height))
    hx1 = int(np.ceil((x + w) * reader.width / target_width))
    hy1 = int(np.ceil((y + h) * reader.height / target_height))

    hx0 = min(max(hx0, 0), reader.width - 1)
    hy0 = min(max(hy0, 0), reader.height - 1)
    hx1 = min(max(hx1, hx0 + 1), reader.width)
    hy1 = min(max(hy1, hy0 + 1), reader.height)

    hsi = reader.read_patch(hx0, hy0, hx1 - hx0, hy1 - hy0, bands)
    if hsi.shape[1:] != (h, w):
        hsi = resize_chw(hsi, (h, w))
    return hsi


# --------------------------------------------------------------------------- #
# Window grid
# --------------------------------------------------------------------------- #

def axis_starts(length: int, crop: int, stride: int) -> list[int]:
    if crop <= 0 or stride <= 0:
        raise ValueError("crop_size and stride must be positive.")
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return starts


def iter_windows(width: int, height: int, crop_size: list[int],
                 stride: list[int]) -> Iterator[tuple[int, int, int, int]]:
    crop_w, crop_h = crop_size
    stride_w, stride_h = stride
    for y in axis_starts(height, crop_h, stride_h):
        for x in axis_starts(width, crop_w, stride_w):
            yield x, y, min(crop_w, width - x), min(crop_h, height - y)


# --------------------------------------------------------------------------- #
# HSI scaling
# --------------------------------------------------------------------------- #

def sample_hsi_median(reader: TiffReader, step: int = 6) -> float:
    """Median |HSI| over a coarse strided sample of the full scene."""
    sl = (slice(None, None, step), slice(None, None, step)) if reader.axis == 0 else None
    if sl is not None:
        sample = np.asarray(reader.array[sl])
    else:
        sample = np.asarray(reader.array[::step, ::step, :])
    return float(np.nanmedian(np.abs(sample)))


def resolve_hsi_scale(cli_scale: float | None, hsi_median: float) -> float:
    """Full-scene HSI is 0-1 reflectance; patches are DN (~ x10000)."""
    if cli_scale is not None:
        return cli_scale
    return 10000.0 if hsi_median < 10.0 else 1.0


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #

def build_scene_split(reader_msi, reader_sar, reader_hsi, reader_lbl,
                      scene: str, out_split: Path, txt_path: Path,
                      crop_size: list[int], stride: list[int],
                      valid_ratio: float, hsi_scale: float,
                      max_patches: int | None,
                      msi_bands, sar_bands, hsi_bands,
                      verbose: bool = True) -> dict:
    """Crop patches from one city, filter, and write one split (train or val)."""
    for sub in ("msisar", "hsi", "lbl"):
        (out_split / sub).mkdir(parents=True, exist_ok=True)

    label_histo = np.zeros(NUM_CLASSES, dtype=np.int64)
    kept = dropped = 0
    lines: list[str] = []

    windows = list(iter_windows(reader_msi.width, reader_msi.height, crop_size, stride))
    total = len(windows)
    if verbose:
        print(f"[{scene}] full scene {reader_msi.width}x{reader_msi.height}, "
              f"{total} candidate windows, hsi_scale={hsi_scale:g}")

    for n, (x, y, w, h) in enumerate(windows):
        if max_patches is not None and kept >= max_patches:
            break

        msi = reader_msi.read_patch(x, y, w, h, msi_bands)     # [4, H, W]
        sar = reader_sar.read_patch(x, y, w, h, sar_bands)     # [2, H, W]
        hsi = read_hsi_patch(
            reader_hsi,
            x,
            y,
            w,
            h,
            target_width=reader_msi.width,
            target_height=reader_msi.height,
            bands=hsi_bands,
        )                                                        # [116, H, W]
        lbl = reader_lbl.read_patch(x, y, w, h, None)[0]       # [H, W]

        # valid pixels = classes 0..13 (nodata / boundary pixels are filtered)
        valid = (lbl >= 0) & (lbl <= NUM_CLASSES - 1)
        ratio = float(valid.mean())
        if ratio < valid_ratio:
            dropped += 1
            continue
        kept += 1
        class_ids = lbl[valid].astype("int64")
        label_histo += np.bincount(class_ids, minlength=NUM_CLASSES)

        patch_id = f"{scene}_{x:05d}_{y:05d}"
        tifffile.imwrite(
            out_split / "msisar" / f"{patch_id}.tiff",
            np.clip(np.concatenate([msi, sar], axis=0), 0, 65535)
            .transpose(1, 2, 0).astype("uint16"),
        )
        tifffile.imwrite(
            out_split / "hsi" / f"{patch_id}.tiff",
            np.clip(
                np.nan_to_num(
                    hsi * hsi_scale,
                    nan=0.0,
                    posinf=65535.0,
                    neginf=0.0,
                ),
                0,
                65535,
            ).transpose(1, 2, 0).astype("uint16"),
        )
        lbl_u8 = np.where(valid, lbl, 0).astype("uint8")
        tifffile.imwrite(out_split / "lbl" / f"{patch_id}.tiff", lbl_u8)

        lines.append(f"msi/{patch_id}.tiff sar/{patch_id}.tiff lbl/{patch_id}.tiff")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    stats = {
        "scene": scene,
        "candidate_windows": total,
        "kept_patches": kept,
        "dropped_patches": dropped,
        "valid_ratio_cutoff": valid_ratio,
        "label_pixels": int(label_histo.sum()),
        "label_histogram": {LABEL_NAMES[i]: int(label_histo[i]) for i in range(NUM_CLASSES)},
    }
    if verbose:
        print(f"[{scene}] kept {kept} / {total} ({dropped} dropped by valid_ratio {valid_ratio:g})")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene-root", default="",
                   help="Dir containing the full-scene TIFFs (Beijing_MSI.tif, ...). "
                        "Defaults to C2SEG_BW_ROOT/../tif_BW from .env.")
    p.add_argument("--split", choices=tuple(SPLITS), default="train_B_val_W",
                   help="Geographic direction: 'train_B_val_W' = train Beijing, "
                        "val Wuhan (default); 'train_W_val_B' = train Wuhan, "
                        "val Beijing.")
    p.add_argument("--output", default="",
                   help="Output root. Writes train/, val/, train.txt, val.txt, "
                        "metadata.json. Default: C2SEG_CITY_ROOT/C2SEG_<SPLIT> "
                        "when C2SEG_CITY_ROOT is set, otherwise "
                        "data/C2Seg_BW_city_<split>.")
    p.add_argument("--train-scene", default="", help=argparse.SUPPRESS)
    p.add_argument("--val-scene", default="", help=argparse.SUPPRESS)
    p.add_argument("--crop-size", nargs=2, type=int, default=[256, 256],
                   metavar=("W", "H"), help="Patch size (default: 256 256).")
    p.add_argument("--stride", nargs=2, type=int, default=[256, 256],
                   metavar=("W", "H"), help="Slide step; >crop-size leaves gaps "
                                            "(default: equal crop = non-overlap).")
    p.add_argument("--valid-ratio", type=float, default=0.5,
                   help="Keep a patch only if this fraction of its pixels carry a valid "
                        "class in 0..13. 0 = keep everything (default: 0.5).")
    p.add_argument("--hsi-scale", type=float, default=None,
                   help="Scale for full-scene HSI (0-1 reflectance -> DN). "
                        "Default auto-detects (10000 if median |HSI| < 10).")
    p.add_argument("--max-patches", type=int, default=None,
                   help="Cap the number of kept patches per split (for testing).")
    p.add_argument("--msi-bands", nargs="+", type=int, default=None)
    p.add_argument("--sar-bands", nargs="+", type=int, default=None)
    p.add_argument("--hsi-bands", nargs="+", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and report, but write no patches.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = load_dotenv(REPO_ROOT / ".env")
    scene_root = resolve_scene_root(args.scene_root, env)
    if not scene_root.is_dir():
        raise SystemExit(
            f"scene root not found: {scene_root}\n"
            "Pass --scene-root or point C2SEG_BW_ROOT at .../C2Seg/src/C2Seg_BW in .env."
        )

    out_root = resolve_output_root(args.output, env, args.split)
    meta: dict = {
        "scene_root": str(scene_root),
        "output_root": str(out_root),
        "argparse": vars(args),
    }

    train_scene, val_scene = SPLITS[args.split]
    if args.train_scene:  # hidden override, e.g. for non-B/W experiments
        train_scene = args.train_scene
    if args.val_scene:
        val_scene = args.val_scene

    for split, scene, sub in (("train", train_scene, "train"),
                              ("val", val_scene, "val")):
        msi_p = find_scene_file(scene_root, scene, "msi")
        sar_p = find_scene_file(scene_root, scene, "sar")
        hsi_p = find_scene_file(scene_root, scene, "hsi")
        lbl_p = find_scene_file(scene_root, scene, "label")
        missing = [str(p) for p, name in ((msi_p, "MSI"), (sar_p, "SAR"),
                                          (hsi_p, "HSI"), (lbl_p, "label"))
                   if p is None]
        if missing:
            raise SystemExit(f"[{scene}] missing from {scene_root}: {', '.join(missing)} "
                             "(run tools/convert_c2seg_full_mat_to_tif.py first if only "
                             "MAT files are available)")
        print(f"[{scene}] MSI={msi_p}\n         SAR={sar_p}\n         HSI={hsi_p}\n         lbl={lbl_p}")

        r_msi = TiffReader(msi_p, "MSI")
        r_sar = TiffReader(sar_p, "SAR")
        r_hsi = TiffReader(hsi_p, "HSI")
        r_lbl = TiffReader(lbl_p, "label")

        if (r_sar.width, r_sar.height) != (r_msi.width, r_msi.height):
            raise SystemExit(
                f"[{scene}] SAR spatial size {r_sar.width}x{r_sar.height} "
                f"does not match MSI {r_msi.width}x{r_msi.height}"
            )
        if (r_lbl.width, r_lbl.height) != (r_msi.width, r_msi.height):
            raise SystemExit(
                f"[{scene}] label spatial size {r_lbl.width}x{r_lbl.height} "
                f"does not match MSI {r_msi.width}x{r_msi.height}"
            )
        print(f"[{scene}] MSI/SAR/label={r_msi.width}x{r_msi.height}, "
              f"HSI={r_hsi.width}x{r_hsi.height} (mapped and resized per patch)")

        msi_bands = args.msi_bands or list(range(1, r_msi.count + 1))
        sar_bands = args.sar_bands or list(range(1, r_sar.count + 1))
        hsi_bands = args.hsi_bands or list(range(1, r_hsi.count + 1))

        hsi_scale = resolve_hsi_scale(args.hsi_scale, sample_hsi_median(r_hsi))
        if split == "train":
            print(f"[train] detected hsi_scale={hsi_scale:g} "
                  f"(if this looks wrong, pass --hsi-scale explicitly)")

        if args.dry_run:
            windows = list(iter_windows(r_msi.width, r_msi.height, args.crop_size, args.stride))
            print(f"[{scene}] DRY-RUN: {len(windows)} candidate windows, no output written.")
            r_msi.close(); r_sar.close(); r_hsi.close(); r_lbl.close()
            continue

        stats = build_scene_split(
            r_msi, r_sar, r_hsi, r_lbl, scene,
            out_split=out_root / sub, txt_path=out_root / f"{split}.txt",
            crop_size=args.crop_size, stride=args.stride,
            valid_ratio=args.valid_ratio, hsi_scale=hsi_scale,
            max_patches=args.max_patches,
            msi_bands=msi_bands, sar_bands=sar_bands, hsi_bands=hsi_bands,
        )
        meta[split] = stats
        r_msi.close(); r_sar.close(); r_hsi.close(); r_lbl.close()

    if args.dry_run:
        print("\nDry-run finished: no patches or metadata were written.")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    n_train = meta.get("train", {}).get("kept_patches", 0)
    n_val = meta.get("val", {}).get("kept_patches", 0)
    print(f"\nDone. split={args.split} -> train={n_train} patches, "
          f"val={n_val} patches -> {out_root}")
    print(f"Next: set C2SEG_BW_CITY_ROOT={out_root} in .env, then train with "
          "PaddleCD/c2seg_config/*_BW_city.yml")


if __name__ == "__main__":
    main()
