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
      train/msisar/<id>.tiff    6 ch  (MSI 4 + SAR 2)             float32
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
SAR is stored as signed float32 because the source values are backscatter in
dB and are negative. Keeping the combined MSI+SAR patch in float32 avoids
silently clipping SAR to zero when it is written to TIFF.

Example
-------
    # default: train Beijing, validate Wuhan (scene root from .env)
    python tools/build_city_split_dataset.py

    # swap the direction (train Wuhan, validate Beijing)
    python tools/build_city_split_dataset.py --split train_W_val_B

    # preview a random 10-cell validation split on both full scenes; this writes
    # only PNG/JSON preview files and does not build a patch dataset
    python tools/build_city_split_dataset.py --preview-grid

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

# The same palette is used by the full-scene conversion and thumbnail tools.
BRIGHT_COLORS = [
    (0, 0, 0),
    (180, 180, 180),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
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


def hsv_to_rgb_uint8(hue: int, saturation: float, value: float) -> tuple[int, int, int]:
    c = value * saturation
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = value - c
    if hue < 60:
        r, g, b = c, x, 0
    elif hue < 120:
        r, g, b = x, c, 0
    elif hue < 180:
        r, g, b = 0, c, x
    elif hue < 240:
        r, g, b = 0, x, c
    elif hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def build_label_palette(num_classes: int = 256) -> np.ndarray:
    """Return the project pseudo-color palette as an RGB lookup table."""
    colors = []
    for class_id in range(num_classes):
        if class_id < len(BRIGHT_COLORS):
            colors.append(BRIGHT_COLORS[class_id])
        else:
            colors.append(hsv_to_rgb_uint8((class_id * 47) % 360, 0.82, 1.0))
    return np.asarray(colors, dtype=np.uint8)


def scale_preview_band(arr: np.ndarray, pmin: float = 2.0,
                       pmax: float = 98.0) -> np.ndarray:
    """Percentile-stretch one preview band to uint8 without loading extra data."""
    arr = np.asarray(arr, dtype="float32")
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [pmin, pmax])
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def preview_step(height: int, width: int, max_dim: int) -> int:
    if max_dim <= 0:
        raise ValueError("preview_max_dim must be positive")
    return max(1, int(np.ceil(max(height, width) / max_dim)))


def read_preview(reader: TiffReader, step: int,
                 bands: list[int] | None = None) -> np.ndarray:
    """Read a downsampled preview, preserving channel-first layout when needed."""
    if reader.axis is None:
        return np.asarray(reader.array[::step, ::step])

    indices = [band - 1 for band in bands] if bands else None
    if reader.axis == 0:
        if indices is not None:
            block = reader.array[indices, ::step, ::step]
        else:
            block = reader.array[:, ::step, ::step]
        return np.asarray(block)

    if indices is not None:
        block = reader.array[::step, ::step, indices]
        return np.transpose(np.asarray(block), (2, 0, 1))
    block = reader.array[::step, ::step, :]
    return np.transpose(np.asarray(block), (2, 0, 1))


def make_rgb_preview(reader: TiffReader, rgb_bands: list[int],
                     max_dim: int) -> tuple[np.ndarray, int]:
    """Build an RGB preview from 1-based MSI band indices."""
    if len(rgb_bands) != 3:
        raise ValueError("preview RGB requires exactly three MSI bands")
    if any(band < 1 or band > reader.count for band in rgb_bands):
        raise ValueError(
            f"preview RGB bands {rgb_bands} exceed MSI band count {reader.count}")
    step = preview_step(reader.height, reader.width, max_dim)
    sampled = read_preview(reader, step, rgb_bands)
    rgb = np.stack([scale_preview_band(sampled[i]) for i in range(3)], axis=-1)
    return rgb, step


def make_label_preview(reader: TiffReader, step: int) -> np.ndarray:
    """Convert a downsampled raw label image to the project GT colors."""
    labels = read_preview(reader, step)
    labels = np.where(np.isfinite(labels), labels, 0).astype(np.int64)
    labels = np.where(labels < 0, 0, labels) % 256
    return build_label_palette(256)[labels]


def select_val_cells(grid_size: int, val_count: int, seed: int) -> list[int]:
    """Select row-major grid-cell IDs for validation using a fixed seed."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    total = grid_size * grid_size
    if val_count < 0 or val_count > total:
        raise ValueError(f"val_grid_count must be in [0, {total}]")
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(total, size=val_count, replace=False))


def draw_grid_overlay(image: np.ndarray, grid_size: int,
                      val_cells: set[int]) -> np.ndarray:
    """Draw grid boundaries and translucent validation cells on an RGB image."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    x_edges = np.rint(np.linspace(0, width, grid_size + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, height, grid_size + 1)).astype(int)
    val_color = np.asarray((225, 45, 45), dtype=np.float32)

    for row in range(grid_size):
        for col in range(grid_size):
            cell_id = row * grid_size + col
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            if cell_id in val_cells and x1 > x0 and y1 > y0:
                region = canvas[y0:y1, x0:x1].astype(np.float32)
                canvas[y0:y1, x0:x1] = (
                    region * 0.62 + val_color * 0.38
                ).astype(np.uint8)
                cv2.rectangle(canvas, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)),
                              (255, 35, 35), 3)
                cv2.putText(canvas, f"V{cell_id:02d}", (x0 + 8, y0 + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)),
                              (20, 20, 20), 1)
    return canvas


def add_preview_caption(image: np.ndarray, caption: str) -> np.ndarray:
    """Add a small readable caption above one preview panel."""
    height, width = image.shape[:2]
    header = np.full((38, width, 3), 245, dtype=np.uint8)
    cv2.putText(header, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (20, 20, 20), 2, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def save_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write preview image: {path}")


def visualize_scene_grid(reader_msi: TiffReader, reader_lbl: TiffReader,
                         scene: str, output_dir: Path, grid_size: int,
                         val_cells: set[int], rgb_bands: list[int],
                         max_dim: int) -> dict:
    """Write RGB, GT-color, and side-by-side grid previews for one scene."""
    if (reader_lbl.width, reader_lbl.height) != (reader_msi.width, reader_msi.height):
        raise ValueError(
            f"[{scene}] label size {reader_lbl.width}x{reader_lbl.height} does not match "
            f"MSI {reader_msi.width}x{reader_msi.height}"
        )
    rgb, step = make_rgb_preview(reader_msi, rgb_bands, max_dim)
    gt_color = make_label_preview(reader_lbl, step)
    rgb_grid = draw_grid_overlay(rgb, grid_size, val_cells)
    gt_grid = draw_grid_overlay(gt_color, grid_size, val_cells)
    pair = np.concatenate([
        add_preview_caption(rgb_grid, "Original MSI RGB; red cells = validation"),
        add_preview_caption(gt_grid, "GT color; red cells = validation"),
    ], axis=1)

    save_rgb_png(output_dir / f"{scene}_grid_rgb.png", rgb_grid)
    save_rgb_png(output_dir / f"{scene}_grid_gt_color.png", gt_grid)
    save_rgb_png(output_dir / f"{scene}_grid_pair.png", pair)
    return {
        "scene": scene,
        "source_size": [reader_msi.width, reader_msi.height],
        "preview_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "downsample_step": step,
        "rgb_bands_1_based": rgb_bands,
        "files": [
            f"{scene}_grid_rgb.png",
            f"{scene}_grid_gt_color.png",
            f"{scene}_grid_pair.png",
        ],
    }


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


def prepare_msisar_patch(msi: np.ndarray, sar: np.ndarray) -> np.ndarray:
    """Combine MSI and signed SAR without changing finite source values.

    The official Normalize2 statistics use SAR backscatter in dB (negative
    values, e.g. mean -15.968/-24.247). The previous uint16 conversion
    clipped those values to zero, making the city-split input inconsistent with
    the ordinary C2Seg-BW data. The dataset reader already converts TIFF
    arrays to float32, so a float32 output TIFF is the lossless representation
    for the mixed MSI+SAR stack.
    """
    msisar = np.concatenate([msi, sar], axis=0).astype("float32", copy=False)
    # Non-finite source values are not expected, but keep generated datasets
    # numerically safe without clipping valid negative SAR backscatter.
    return np.nan_to_num(
        msisar,
        nan=0.0,
        posinf=65535.0,
        neginf=-65535.0,
    )


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


def iter_grid_windows(width: int, height: int, crop_size: list[int],
                      stride: list[int], grid_size: int,
                      val_cells: set[int], target_split: str
                      ) -> Iterator[tuple[int, int, int, int]]:
    """Yield windows wholly contained in train or validation grid cells.

    Windows are regenerated independently inside each grid cell. This avoids
    assigning a patch by its center while allowing the patch to cross a
    train/validation boundary, which would reintroduce spatial leakage.
    """
    if target_split not in {"train", "val"}:
        raise ValueError(f"target_split must be train or val, got {target_split}")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    crop_w, crop_h = crop_size
    stride_w, stride_h = stride
    x_edges = np.rint(np.linspace(0, width, grid_size + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, height, grid_size + 1)).astype(int)

    for row in range(grid_size):
        for col in range(grid_size):
            cell_id = row * grid_size + col
            is_val = cell_id in val_cells
            if (target_split == "val") != is_val:
                continue
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            cell_width = x1 - x0
            cell_height = y1 - y0
            for local_y in axis_starts(cell_height, crop_h, stride_h):
                for local_x in axis_starts(cell_width, crop_w, stride_w):
                    yield (
                        x0 + local_x,
                        y0 + local_y,
                        min(crop_w, cell_width - local_x),
                        min(crop_h, cell_height - local_y),
                    )


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
                      verbose: bool = True,
                      windows: Iterator[tuple[int, int, int, int]] | None = None,
                      append_txt: bool = False) -> dict:
    """Crop patches from one city, filter, and write one split (train or val)."""
    for sub in ("msisar", "hsi", "lbl"):
        (out_split / sub).mkdir(parents=True, exist_ok=True)

    label_histo = np.zeros(NUM_CLASSES, dtype=np.int64)
    kept = dropped = 0
    lines: list[str] = []

    windows = list(windows) if windows is not None else list(
        iter_windows(reader_msi.width, reader_msi.height, crop_size, stride)
    )
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
            prepare_msisar_patch(msi, sar).transpose(1, 2, 0),
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
    existing = ""
    if append_txt and txt_path.is_file():
        existing = txt_path.read_text(encoding="utf-8").rstrip()
    chunks = [chunk for chunk in (existing, "\n".join(lines)) if chunk]
    txt_path.write_text("\n".join(chunks) + ("\n" if chunks else ""), encoding="utf-8")

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


def merge_split_stats(scene_stats: list[dict], valid_ratio: float) -> dict:
    """Aggregate per-scene patch statistics for one generated split."""
    label_histo = np.zeros(NUM_CLASSES, dtype=np.int64)
    for stats in scene_stats:
        for index, name in enumerate(LABEL_NAMES):
            label_histo[index] += int(stats["label_histogram"].get(name, 0))
    return {
        "scenes": scene_stats,
        "candidate_windows": sum(int(stats["candidate_windows"]) for stats in scene_stats),
        "kept_patches": sum(int(stats["kept_patches"]) for stats in scene_stats),
        "dropped_patches": sum(int(stats["dropped_patches"]) for stats in scene_stats),
        "valid_ratio_cutoff": valid_ratio,
        "label_pixels": int(label_histo.sum()),
        "label_histogram": {
            LABEL_NAMES[index]: int(label_histo[index])
            for index in range(NUM_CLASSES)
        },
    }


def build_grid_dataset(args: argparse.Namespace, scene_root: Path,
                       out_root: Path, val_cells: list[int]) -> None:
    """Build one BW dataset from both scenes using a spatial grid split."""
    existing_outputs = [out_root / name for name in ("train.txt", "val.txt", "metadata.json")]
    if out_root.exists() and any(path.exists() for path in existing_outputs):
        raise SystemExit(
            f"grid output already contains generated files: {out_root}; "
            "choose a new --output or remove the incomplete dataset explicitly"
        )

    val_cell_set = set(val_cells)
    per_split_stats: dict[str, list[dict]] = {"train": [], "val": []}
    scene_names = ("beijing", "wuhan")

    for scene_index, scene in enumerate(scene_names):
        msi_p = find_scene_file(scene_root, scene, "msi")
        sar_p = find_scene_file(scene_root, scene, "sar")
        hsi_p = find_scene_file(scene_root, scene, "hsi")
        lbl_p = find_scene_file(scene_root, scene, "label")
        missing = [name for path, name in (
            (msi_p, "MSI"), (sar_p, "SAR"), (hsi_p, "HSI"), (lbl_p, "label")
        ) if path is None]
        if missing:
            raise SystemExit(
                f"[{scene}] missing from {scene_root}: {', '.join(missing)}"
            )
        print(f"[{scene}] grid dataset MSI={msi_p}\n"
              f"         SAR={sar_p}\n"
              f"         HSI={hsi_p}\n"
              f"         label={lbl_p}")

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

        msi_bands = args.msi_bands or list(range(1, r_msi.count + 1))
        sar_bands = args.sar_bands or list(range(1, r_sar.count + 1))
        hsi_bands = args.hsi_bands or list(range(1, r_hsi.count + 1))
        hsi_scale = resolve_hsi_scale(args.hsi_scale, sample_hsi_median(r_hsi))
        print(f"[{scene}] MSI/SAR/label={r_msi.width}x{r_msi.height}, "
              f"HSI={r_hsi.width}x{r_hsi.height}, hsi_scale={hsi_scale:g}")

        for target_split in ("train", "val"):
            windows = iter_grid_windows(
                r_msi.width, r_msi.height, args.crop_size, args.stride,
                args.grid_size, val_cell_set, target_split,
            )
            stats = build_scene_split(
                r_msi, r_sar, r_hsi, r_lbl, scene,
                out_split=out_root / target_split,
                txt_path=out_root / f"{target_split}.txt",
                crop_size=args.crop_size, stride=args.stride,
                valid_ratio=args.valid_ratio, hsi_scale=hsi_scale,
                max_patches=args.max_patches,
                msi_bands=msi_bands, sar_bands=sar_bands,
                hsi_bands=hsi_bands,
                windows=windows,
                append_txt=scene_index > 0,
            )
            per_split_stats[target_split].append(stats)

        r_msi.close()
        r_sar.close()
        r_hsi.close()
        r_lbl.close()

    meta = {
        "scene_root": str(scene_root),
        "output_root": str(out_root),
        "argparse": vars(args),
        "grid": {
            "grid_size": args.grid_size,
            "val_grid_count": args.val_grid_count,
            "grid_seed": args.grid_seed,
            "val_cells": [
                {"id": cell, "row": cell // args.grid_size,
                 "col": cell % args.grid_size}
                for cell in val_cells
            ],
            "scenes": list(scene_names),
            "patches_are_contained_in_one_cell": True,
        },
        "train": merge_split_stats(per_split_stats["train"], args.valid_ratio),
        "val": merge_split_stats(per_split_stats["val"], args.valid_ratio),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nDone. grid seed={args.grid_seed} -> "
        f"train={meta['train']['kept_patches']} patches, "
        f"val={meta['val']['kept_patches']} patches -> {out_root}"
    )


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
    p.add_argument("--preview-grid", action="store_true",
                   help="Only visualize a random grid split on the original full scenes; "
                        "do not write train/val patches.")
    p.add_argument("--grid-split", action="store_true",
                   help="Build a BW dataset from both full scenes using the selected "
                        "spatial grid cells as validation.")
    p.add_argument("--grid-size", type=int, default=10,
                   help="Rows and columns in the preview grid (default: 10).")
    p.add_argument("--val-grid-count", type=int, default=10,
                   help="Number of randomly selected validation cells (default: 10).")
    p.add_argument("--grid-seed", type=int, default=1919810,
                   help="Random seed for selecting validation cells.")
    p.add_argument("--preview-scene", choices=("beijing", "wuhan", "both"),
                   default="both",
                   help="Scene(s) to preview (default: both).")
    p.add_argument("--preview-output", default="",
                   help="Preview output directory. Default: ana/grid_preview_<size>x<size>_seed<seed>.")
    p.add_argument("--preview-max-dim", type=int, default=1800,
                   help="Longest edge of each preview image (default: 1800).")
    p.add_argument("--preview-rgb-bands", nargs=3, type=int, default=[1, 2, 3],
                   metavar=("R", "G", "B"),
                   help="1-based MSI bands used for the original RGB preview (default: 1 2 3).")
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

    if args.preview_grid and args.grid_split:
        raise SystemExit("--preview-grid and --grid-split are mutually exclusive")

    if args.grid_split:
        val_cells = select_val_cells(args.grid_size, args.val_grid_count, args.grid_seed)
        if args.output:
            grid_output = Path(args.output)
        else:
            city_root = env.get("C2SEG_CITY_ROOT", "") or os.environ.get(
                "C2SEG_CITY_ROOT", ""
            )
            city_root = city_root.strip().strip('"').strip("'")
            grid_output = (
                Path(city_root) /
                f"C2SEG_BW_GRID_{args.grid_size}X{args.grid_size}_SEED{args.grid_seed}"
                if city_root else
                REPO_ROOT / "data" /
                f"C2Seg_BW_grid_{args.grid_size}x{args.grid_size}_seed{args.grid_seed}"
            )
        build_grid_dataset(args, scene_root, grid_output, val_cells)
        return

    if args.preview_grid:
        if args.dry_run:
            raise SystemExit("--preview-grid and --dry-run are mutually exclusive")
        val_cells = select_val_cells(args.grid_size, args.val_grid_count, args.grid_seed)
        preview_scenes = {
            "beijing": ["beijing"],
            "wuhan": ["wuhan"],
            "both": ["beijing", "wuhan"],
        }[args.preview_scene]
        preview_root = (
            Path(args.preview_output)
            if args.preview_output
            else REPO_ROOT / "ana" /
            f"grid_preview_{args.grid_size}x{args.grid_size}_seed{args.grid_seed}"
        )
        preview_meta: dict = {
            "scene_root": str(scene_root),
            "output_root": str(preview_root),
            "grid_size": args.grid_size,
            "val_grid_count": args.val_grid_count,
            "grid_seed": args.grid_seed,
            "val_cells": [
                {"id": cell, "row": cell // args.grid_size,
                 "col": cell % args.grid_size}
                for cell in val_cells
            ],
            "preview_scene": args.preview_scene,
            "scenes": [],
        }
        val_cell_set = set(val_cells)

        for scene in preview_scenes:
            msi_p = find_scene_file(scene_root, scene, "msi")
            lbl_p = find_scene_file(scene_root, scene, "label")
            if msi_p is None or lbl_p is None:
                missing = []
                if msi_p is None:
                    missing.append("MSI")
                if lbl_p is None:
                    missing.append("label")
                raise SystemExit(
                    f"[{scene}] missing from {scene_root}: {', '.join(missing)}"
                )
            print(f"[{scene}] preview MSI={msi_p}\n"
                  f"         label={lbl_p}")
            r_msi = TiffReader(msi_p, "MSI")
            r_lbl = TiffReader(lbl_p, "label")
            preview_meta["scenes"].append(
                visualize_scene_grid(
                    r_msi, r_lbl, scene, preview_root, args.grid_size,
                    val_cell_set, args.preview_rgb_bands, args.preview_max_dim,
                )
            )
            r_msi.close()
            r_lbl.close()

        preview_root.mkdir(parents=True, exist_ok=True)
        (preview_root / "grid_assignment.json").write_text(
            json.dumps(preview_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nGrid preview written to {preview_root}")
        print(f"Validation cells: {val_cells}")
        return

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
