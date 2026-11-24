"""Utilities for preparing the raw MDAS (Augsburg) data set.

The raw release stores the OSM reference rasters at 2.2 m while the EeteS
EnMAP / Sentinel-2 / Sentinel-1 rasters are on a 10 m grid (888 x 1371,
UTM 32N).  This tool aligns the 2.2 m land-use and water layers to the 10 m
reference grid using a geotransform-aware nearest-neighbour resample, so the
labels can be stacked with the rasters without any half-pixel drift.

The SAR layer is intentionally **not** resampled: its grid has the same size
and GSD as the reference, with a residual origin offset of only ~(0.28, 0.12)
pixels.  Keeping it index-aligned is acceptable for the supplementary
experiment; use ``info`` to print the exact offsets.

Subcommands
-----------
``info``
    Print the reference/label/SAR grids, their origins and the offsets in
    metres and reference pixels.

``align-labels``
    Resample ``osm_landuse.tif`` and ``osm_water.tif`` to the 10 m reference
    grid and write:

        <out>/landuse_10m.tif    14 (or 18) classes, 255 = ignore
        <out>/water_10m.tif      only with --with-water / --merge-water
        <out>/label_10m.tif      only with --merge-water
        <out>/alignment_meta.json

    Only the land-use layer is processed by default; pass ``--with-water`` (or
    ``--merge-water``) to also align the water mask.

    ``--class-scheme paper14`` keeps the 14 classes used in the MDAS paper
    (nature reserve, military, quarry and orchard are mapped to ignore).
    ``--class-scheme all18`` keeps all 18 OSM land-use classes in the release.
    With ``--merge-water`` water becomes class ``num_classes`` (14 or 18).

``build-patches``
    Tile the aligned land-use label and the three raw rasters into
    non-overlapping patches (default 64x64, right/bottom padded):

        <out>/hsi/<id>.tiff       [64, 64, 242] uint16
        <out>/msisar/<id>.tiff    [64, 64, 6]   float32 (MSI 4 + SAR 2 dB)
        <out>/lbl/<id>.tiff       [64, 64]      uint8, 255 = ignore
        <out>/patches.txt         "msi/<id>.tiff sar/<id>.tiff lbl/<id>.tiff"
        <out>/metadata.json

    MSI bands default to ``3 2 1 7`` (B4/B3/B2/B8); SAR is converted to dB and
    kept index-aligned (not resampled).  Padded inputs are 0, padded labels are
    255.  Use ``--no-pad`` to drop partial edge tiles instead.

``split``
    Shuffle the patch list and write a patch-level random ``train.txt`` /
    ``val.txt`` split (the same style as the default C2Seg split), plus
    ``split_metadata.json`` with per-split class statistics.  This intentionally
    ignores spatial adjacency; use the grid/strip scheme when a geographically
    disjoint split is required.

``stats``
    Compute ``Normalize2`` statistics the same way as the C2Seg
    ``tools/data_sta.py``: for every patch take the per-channel mean/std, then
    average those per-patch values across patches (``Col_avg``).  Pass both
    ``train.txt`` and ``val.txt`` to use the same patch scope as C2Seg's
    ``train/`` folder.  ``--update-config MDAS.yml`` replaces the four lists in
    both the train and val dataset blocks.

Examples
--------
    python tools/mdas_tool.py info \
        --root Z:/huilin/backup/data_backup/Augsburg_data_4_publication

    python tools/mdas_tool.py align-labels \
        --root Z:/huilin/backup/data_backup/Augsburg_data_4_publication \
        --out data/MDAS_10m --merge-water

    # compute the alignment and print the class histogram without writing
    python tools/mdas_tool.py align-labels --root ... --dry-run

    python tools/mdas_tool.py build-patches \
        --root Z:/huilin/backup/data_backup/Augsburg_data_4_publication \
        --labels data/MDAS_10m/landuse_10m.tif \
        --out data/MDAS_64

    python tools/mdas_tool.py split \
        --patches data/MDAS_64/patches.txt --val-ratio 0.1 --seed 0

    python tools/mdas_tool.py stats \
        --lists Z:/huilin/backup/data_backup/Augsburg_data_4_publication/mdas_clip/train.txt \
                Z:/huilin/backup/data_backup/Augsburg_data_4_publication/mdas_clip/val.txt \
        --mask none --update-config PaddleCD/c2seg_config/MDAS.yml
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import yaml

REF_REL = Path("entire_city") / "EeteS_EnMAP_10m.tif"
LANDUSE_REL = Path("entire_city") / "OSM_label" / "osm_landuse.tif"
WATER_REL = Path("entire_city") / "OSM_label" / "osm_water.tif"
S1_REL = Path("entire_city") / "Sentinel-1.tif"
S2_REL = Path("entire_city") / "Sentinel-2.tif"
HSI_REL = REF_REL
MSI_REL = S2_REL
SAR_REL = S1_REL

IGNORE_INDEX = 255

# OSM land-use codes present in osm_landuse.tif, in the order used by MDAS.
LANDUSE_LEGEND: tuple[tuple[int, str], ...] = (
    (7201, "forest"),
    (7202, "park"),
    (7203, "residential"),
    (7204, "industrial"),
    (7205, "farm"),
    (7206, "cemetery"),
    (7207, "allotments"),
    (7208, "meadow"),
    (7209, "commercial"),
    (7210, "nature reserve"),
    (7211, "recreation ground"),
    (7212, "retail"),
    (7213, "military"),
    (7214, "quarry"),
    (7215, "orchard"),
    (7217, "scrub"),
    (7218, "grass"),
    (7219, "heath"),
)

# The 14 classes used for the land-cover/land-use experiments in the paper.
PAPER14_CODES = frozenset(
    {7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7211, 7212, 7217, 7218, 7219}
)

WATER_CODES = frozenset({8200, 8201, 8202, 8221})  # water, reservoir, river, wetland

GEO_TAGS = (
    "ModelPixelScaleTag",
    "ModelTiepointTag",
    "GeoKeyDirectoryTag",
    "GeoAsciiParamsTag",
    "GeoDoubleParamsTag",
)


@dataclass(frozen=True)
class Grid:
    """Affine north-up grid read from a GeoTIFF."""

    path: Path
    x0: float  # easting of the upper-left corner (pixel (0, 0) corner)
    y0: float  # northing of the upper-left corner
    sx: float  # pixel width in metres
    sy: float  # pixel height in metres (positive)
    width: int
    height: int
    crs: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.x0 + self.width * self.sx, self.y0 - self.height * self.sy, self.y0

    def as_dict(self) -> dict:
        x_min, x_max, y_min, y_max = self.bounds()
        return {
            "path": str(self.path),
            "shape": [self.height, self.width],
            "gsd_m": [self.sx, self.sy],
            "origin": [self.x0, self.y0],
            "bounds_xyxy": [x_min, y_min, x_max, y_max],
            "crs": self.crs,
        }


def read_grid(path: Path) -> Grid:
    """Read the north-up grid definition from a GeoTIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tags = {tag.name: tag.value for tag in page.tags.values()}
        if "ModelPixelScaleTag" not in tags or "ModelTiepointTag" not in tags:
            raise ValueError(f"{path} has no ModelPixelScaleTag/ModelTiepointTag")
        sx, sy = (float(v) for v in tags["ModelPixelScaleTag"][:2])
        i, j, _k, x, y, _z = (float(v) for v in tags["ModelTiepointTag"][:6])
        # Tiepoint maps raster (i, j) to model (x, y); normalise to raster (0, 0).
        x0 = x - i * sx
        y0 = y + j * sy
        return Grid(
            path=path,
            x0=x0,
            y0=y0,
            sx=sx,
            sy=sy,
            width=int(page.imagewidth),
            height=int(page.imagelength),
            crs=tags.get("GeoAsciiParamsTag"),
        )


def geotiff_extratags(ref_path: Path) -> list[tuple]:
    """Copy the GeoTIFF georeferencing tags from ``ref_path``."""
    with tifffile.TiffFile(ref_path) as tif:
        page = tif.pages[0]
        out: list[tuple] = []
        for name in GEO_TAGS:
            if name in page.tags:
                tag = page.tags[name]
                out.append((tag.code, tag.dtype, tag.count, tag.value))
        return out


def write_geotiff(path: Path, array: np.ndarray, extratags: list[tuple]) -> None:
    """Write a single-band GeoTIFF using the reference georeferencing tags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        array,
        photometric="minisblack",
        extratags=extratags,
        metadata=None,
    )


def _nearest_source_indices(src: Grid, dst: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return source (row, col) indices for every destination pixel centre."""
    xs = dst.x0 + (np.arange(dst.width, dtype=np.float64) + 0.5) * dst.sx
    ys = dst.y0 - (np.arange(dst.height, dtype=np.float64) + 0.5) * dst.sy
    col = np.floor((xs - src.x0) / src.sx).astype(np.int64)
    row = np.floor((src.y0 - ys) / src.sy).astype(np.int64)
    return row, col


def align_nearest(
    src_array: np.ndarray,
    src: Grid,
    dst: Grid,
    fill: float = -1.0,
) -> np.ndarray:
    """Nearest-neighbour resample ``src_array`` onto ``dst``.

    The output keeps the source pixel whose cell contains each destination
    pixel centre.  Pixels outside the source extent are set to ``fill``.
    """
    if src_array.shape[:2] != (src.height, src.width):
        raise ValueError(
            f"source array shape {src_array.shape[:2]} does not match grid "
            f"{(src.height, src.width)}"
        )
    row, col = _nearest_source_indices(src, dst)
    row_ok = (row >= 0) & (row < src.height)
    col_ok = (col >= 0) & (col < src.width)
    inside = row_ok[:, None] & col_ok[None, :]

    row_c = np.clip(row, 0, src.height - 1)
    col_c = np.clip(col, 0, src.width - 1)
    sampled = src_array[np.ix_(row_c, col_c)]

    dtype = np.result_type(src_array.dtype, np.float64) if fill < 0 else src_array.dtype
    out = np.full((dst.height, dst.width), fill, dtype=dtype)
    out[inside] = sampled[inside]
    return out


def build_mapping(class_scheme: str) -> tuple[dict[int, int], list[str]]:
    """Return {osm_code: class_id} and the class names for a scheme."""
    if class_scheme == "paper14":
        selected = [(code, name) for code, name in LANDUSE_LEGEND if code in PAPER14_CODES]
    elif class_scheme == "all18":
        selected = list(LANDUSE_LEGEND)
    else:
        raise ValueError(f"unknown class scheme: {class_scheme}")
    mapping = {code: i for i, (code, _name) in enumerate(selected)}
    names = [name for _code, name in selected]
    return mapping, names


def map_codes(
    aligned_codes: np.ndarray,
    mapping: dict[int, int],
    ignore: int = IGNORE_INDEX,
) -> np.ndarray:
    """Map aligned OSM codes to contiguous class ids; anything else is ignore."""
    out = np.full(aligned_codes.shape, ignore, dtype=np.uint8)
    for code, class_id in mapping.items():
        out[aligned_codes == code] = class_id
    if mapping:
        assert len(mapping) <= 254, "class ids must fit in uint8 alongside ignore"
    return out


def water_mask(aligned_codes: np.ndarray, outside_fill: float = -1.0) -> np.ndarray:
    """Build a 0/1 water mask; values outside the source extent become 255."""
    out = np.zeros(aligned_codes.shape, dtype=np.uint8)
    for code in WATER_CODES:
        out[aligned_codes == code] = 1
    out[aligned_codes == outside_fill] = IGNORE_INDEX
    return out


def class_histogram(label: np.ndarray, names: list[str]) -> dict:
    counts = np.bincount(label.ravel(), minlength=256)
    total = int(label.size)
    per_class = {
        name: {
            "class_id": i,
            "pixels": int(counts[i]),
            "percent": 100.0 * float(counts[i]) / total,
        }
        for i, name in enumerate(names)
    }
    return {
        "total_pixels": total,
        "num_classes": len(names),
        "classes": per_class,
        "ignore": {
            "class_id": IGNORE_INDEX,
            "pixels": int(counts[IGNORE_INDEX]),
            "percent": 100.0 * float(counts[IGNORE_INDEX]) / total,
        },
    }


def _grid_offset(src: Grid, dst: Grid) -> dict:
    """Describe ``src`` relative to ``dst`` in metres and reference pixels."""
    dx_m = src.x0 - dst.x0
    dy_m = src.y0 - dst.y0
    return {
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dx_px": dx_m / dst.sx,
        "dy_px": dy_m / dst.sy,
        "size_delta_px": [src.width - dst.width, src.height - dst.height],
        "extent_delta_m": [
            src.width * src.sx - dst.width * dst.sx,
            src.height * src.sy - dst.height * dst.sy,
        ],
    }


def _print_grid(name: str, grid: Grid) -> None:
    x_min, x_max, y_min, y_max = grid.bounds()
    print(
        f"{name:10s} {grid.width:5d} x {grid.height:<5d}  "
        f"{grid.sx:g} m  origin=({grid.x0:.3f}, {grid.y0:.3f})  "
        f"extent=({x_min:.1f}, {y_min:.1f}) - ({x_max:.1f}, {y_max:.1f})"
    )


def cmd_info(args: argparse.Namespace) -> int:
    root = Path(args.root)
    ref = read_grid(root / REF_REL)
    landuse = read_grid(root / LANDUSE_REL)
    water = read_grid(root / WATER_REL)
    s1 = read_grid(root / S1_REL) if (root / S1_REL).exists() else None
    s2 = read_grid(root / S2_REL) if (root / S2_REL).exists() else None

    print("MDAS raw grids")
    print("-" * 100)
    _print_grid("reference", ref)
    _print_grid("landuse", landuse)
    _print_grid("water", water)
    if s2 is not None:
        _print_grid("s2", s2)
    if s1 is not None:
        _print_grid("s1", s1)
    print(f"\nreference CRS: {ref.crs!r}")

    print("\nOffsets relative to the reference grid")
    print("-" * 100)
    for name, grid in (("landuse", landuse), ("water", water), ("s2", s2), ("s1", s1)):
        if grid is None:
            continue
        off = _grid_offset(grid, ref)
        print(
            f"{name:8s} dx={off['dx_m']:+.3f} m ({off['dx_px']:+.4f} px)  "
            f"dy={off['dy_m']:+.3f} m ({off['dy_px']:+.4f} px)  "
            f"size_delta={off['size_delta_px']} px  "
            f"extent_delta={off['extent_delta_m'][0]:+.2f} x "
            f"{off['extent_delta_m'][1]:+.2f} m"
        )

    if s1 is not None:
        print(
            "\nNote: SAR is index-aligned to the reference (same size/GSD, "
            "sub-pixel origin shift). This tool does not resample it."
        )
    return 0


def cmd_align_labels(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out_dir = Path(args.out)
    ref_path = root / REF_REL
    landuse_path = root / LANDUSE_REL
    water_path = root / WATER_REL

    # Water is optional; the default only aligns the land-use layer.
    use_water = bool(args.with_water or args.merge_water)
    required = [ref_path, landuse_path]
    if use_water:
        required.append(water_path)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    ref = read_grid(ref_path)
    landuse_grid = read_grid(landuse_path)
    water_grid = read_grid(water_path) if use_water else None
    mapping, names = build_mapping(args.class_scheme)
    num_classes = len(names)

    print(f"reference: {ref.width}x{ref.height} @ {ref.sx:g} m")
    print(f"landuse  : {landuse_grid.width}x{landuse_grid.height} @ {landuse_grid.sx:g} m")
    if water_grid is not None:
        print(f"water    : {water_grid.width}x{water_grid.height} @ {water_grid.sx:g} m")
    off = _grid_offset(landuse_grid, ref)
    print(
        f"label offset: dx={off['dx_m']:+.3f} m ({off['dx_px']:+.4f} px), "
        f"dy={off['dy_m']:+.3f} m ({off['dy_px']:+.4f} px)"
    )

    print("resampling land-use to the reference grid ...")
    landuse_raw = tifffile.imread(landuse_path)
    landuse_aligned = align_nearest(landuse_raw, landuse_grid, ref, fill=-1.0)
    landuse = map_codes(landuse_aligned, mapping)

    water = None
    water_px = 0
    water_on_labeled = 0
    water_on_ignore = 0
    if use_water:
        print("resampling water to the reference grid ...")
        water_raw = tifffile.imread(water_path)
        water_aligned = align_nearest(water_raw, water_grid, ref, fill=-1.0)
        water = water_mask(water_aligned)
        water_px = int((water == 1).sum())
        water_on_labeled = int(((water == 1) & (landuse != IGNORE_INDEX)).sum())
        water_on_ignore = water_px - water_on_labeled

    label = None
    merged_px = 0
    if args.merge_water:
        assert water is not None
        label = landuse.copy()
        if args.water_policy == "override":
            merge_mask = water == 1
        else:  # unlabeled
            merge_mask = (water == 1) & (landuse == IGNORE_INDEX)
        label[merge_mask] = num_classes
        merged_px = int(merge_mask.sum())
        print(
            f"merged water as class {num_classes} "
            f"(policy={args.water_policy}, {merged_px} px)"
        )

    hist = class_histogram(landuse, names)
    output_hist = class_histogram(label, names + ["water"]) if label is not None else None

    print("\nlabel histogram (land-use classes):")
    for name, info in hist["classes"].items():
        print(f"  {info['class_id']:2d} {name:20s} {info['pixels']:>9d}  {info['percent']:6.2f}%")
    print(
        f"  {IGNORE_INDEX:2d} {'ignore':20s} {hist['ignore']['pixels']:>9d}  "
        f"{hist['ignore']['percent']:6.2f}%"
    )
    if water is not None:
        print(f"water mask: {water_px} px ({100.0 * water_px / water.size:.2f}%)")
        print(f"  on labeled land-use: {water_on_labeled} px")
        print(f"  on unlabeled areas : {water_on_ignore} px")

    meta = {
        "tool": "tools/mdas_tool.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "reference": ref.as_dict(),
        "landuse_grid": landuse_grid.as_dict(),
        "water_grid": water_grid.as_dict() if water_grid is not None else None,
        "offset_landuse_vs_ref": _grid_offset(landuse_grid, ref),
        "class_scheme": args.class_scheme,
        "class_names": names,
        "ignore_index": IGNORE_INDEX,
        "landuse_histogram": hist,
        "water": (
            {
                "codes": sorted(WATER_CODES),
                "pixels": water_px,
                "percent": 100.0 * water_px / water.size,
                "on_labeled": water_on_labeled,
                "on_unlabeled": water_on_ignore,
                "output_semantics": "0=not water, 1=water, 255=outside source extent",
            }
            if water is not None
            else None
        ),
        "merge_water": bool(args.merge_water),
        "water_policy": args.water_policy if args.merge_water else None,
        "merged_water_class": num_classes if args.merge_water else None,
        "merged_water_pixels": merged_px,
        "output_histogram": output_hist,
    }

    outputs = {"landuse": out_dir / "landuse_10m.tif"}
    if water is not None:
        outputs["water"] = out_dir / "water_10m.tif"
    if label is not None:
        outputs["label"] = out_dir / "label_10m.tif"
    meta_path = out_dir / "alignment_meta.json"

    if args.dry_run:
        print("\n[dry-run] no files written; planned outputs:")
        for name, path in outputs.items():
            print(f"  {name:8s} -> {path}")
        print(f"  metadata -> {meta_path}")
        return 0

    if not args.overwrite:
        existing = [p for p in (*outputs.values(), meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "output already exists (use --overwrite): "
                + ", ".join(str(p) for p in existing)
            )

    extratags = geotiff_extratags(ref_path)
    write_geotiff(outputs["landuse"], landuse, extratags)
    if water is not None:
        write_geotiff(outputs["water"], water, extratags)
    if label is not None:
        write_geotiff(outputs["label"], label, extratags)

    out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print("\nwritten:")
    for name, path in outputs.items():
        print(f"  {name:8s} -> {path}")
    print(f"  metadata -> {meta_path}")
    return 0


def _crop_pad(array: np.ndarray, x0: int, y0: int, size: int, fill) -> np.ndarray:
    """Crop a ``size x size`` window, padding right/bottom with ``fill``."""
    height, width = array.shape[:2]
    out = np.full((size, size, *array.shape[2:]), fill, dtype=array.dtype)
    x1 = min(x0 + size, width)
    y1 = min(y0 + size, height)
    out[: y1 - y0, : x1 - x0] = array[y0:y1, x0:x1]
    return out


def cmd_build_patches(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out_dir = Path(args.out)
    label_path = Path(args.labels)
    hsi_path = root / HSI_REL
    msi_path = root / MSI_REL
    sar_path = root / SAR_REL
    for path in (hsi_path, msi_path, sar_path, label_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found"
                + (" (run align-labels first)" if path == label_path else "")
            )

    ps = int(args.patch_size)
    if ps <= 0:
        raise ValueError("--patch-size must be positive")
    pad = not args.no_pad

    ref = read_grid(hsi_path)
    nx = (ref.width + ps - 1) // ps if pad else ref.width // ps
    ny = (ref.height + ps - 1) // ps if pad else ref.height // ps
    total = nx * ny
    planned = total if args.max_patches is None else min(total, args.max_patches)

    print(f"reference : {ref.width}x{ref.height} @ {ref.sx:g} m")
    print(
        f"tiling    : {ps}x{ps}, stride={ps}, pad={pad} -> "
        f"{nx} cols x {ny} rows = {total} patches"
    )
    if args.max_patches is not None and planned < total:
        print(f"limit     : writing only the first {planned} patches")

    if args.dry_run:
        print("[dry-run] no patches written")
        return 0

    if not args.overwrite and (out_dir / "patches.txt").exists():
        raise FileExistsError(
            f"{out_dir / 'patches.txt'} already exists (use --overwrite)"
        )

    mapping, names = build_mapping(args.class_scheme)
    label = tifffile.imread(label_path)
    if label.shape != (ref.height, ref.width):
        raise ValueError(
            f"label shape {label.shape} does not match reference "
            f"{(ref.height, ref.width)}"
        )
    valid_classes = label[label != IGNORE_INDEX]
    max_class = int(valid_classes.max()) if valid_classes.size else -1
    if max_class >= len(names):
        raise ValueError(
            f"label contains class id {max_class}, but --class-scheme "
            f"{args.class_scheme} defines only {len(names)} classes; "
            "use --class-scheme all18 or regenerate the label"
        )

    hsi = tifffile.imread(hsi_path)
    msi_full = tifffile.imread(msi_path)
    sar_full = tifffile.imread(sar_path)
    if hsi.shape[:2] != (ref.height, ref.width):
        raise ValueError(f"HSI shape {hsi.shape[:2]} != reference grid")
    if msi_full.shape[:2] != (ref.height, ref.width):
        raise ValueError(f"MSI shape {msi_full.shape[:2]} != reference grid")
    if sar_full.shape[:2] != (ref.height, ref.width):
        raise ValueError(f"SAR shape {sar_full.shape[:2]} != reference grid")

    bands = list(args.msi_bands)
    if len(bands) != 4 or any(b < 0 or b >= msi_full.shape[2] for b in bands):
        raise ValueError(
            f"--msi-bands must be four valid indices into {msi_full.shape[2]} bands"
        )
    msi = msi_full[..., bands]
    sar_db = (10.0 * np.log10(np.maximum(sar_full, float(args.sar_eps)))).astype(
        np.float32
    )

    for sub in ("hsi", "msisar", "lbl"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    hist = np.zeros(256, dtype=np.int64)
    valid_ratios: list[float] = []
    written = 0
    stop = False
    for row in range(ny):
        if stop:
            break
        for col in range(nx):
            if args.max_patches is not None and written >= args.max_patches:
                stop = True
                break
            x0, y0 = col * ps, row * ps
            hsi_patch = _crop_pad(hsi, x0, y0, ps, 0)
            msi_patch = _crop_pad(msi, x0, y0, ps, 0)
            sar_patch = _crop_pad(sar_db, x0, y0, ps, float(args.sar_pad_db))
            lbl_patch = _crop_pad(label, x0, y0, ps, IGNORE_INDEX)
            msisar_patch = np.concatenate(
                [msi_patch.astype(np.float32), sar_patch.astype(np.float32)], axis=2
            )

            pid = f"augsburg_{x0:05d}_{y0:05d}"
            tifffile.imwrite(out_dir / "hsi" / f"{pid}.tiff", hsi_patch)
            tifffile.imwrite(out_dir / "msisar" / f"{pid}.tiff", msisar_patch)
            tifffile.imwrite(out_dir / "lbl" / f"{pid}.tiff", lbl_patch)

            lines.append(f"msi/{pid}.tiff sar/{pid}.tiff lbl/{pid}.tiff")
            hist += np.bincount(lbl_patch.ravel(), minlength=256)
            valid_ratios.append(float((lbl_patch != IGNORE_INDEX).mean()))
            written += 1
            if written % 50 == 0:
                print(f"  wrote {written}/{planned}")

    with (out_dir / "patches.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    total_px = int(hist.sum())
    class_stats = {
        name: {
            "class_id": i,
            "pixels": int(hist[i]),
            "percent": 100.0 * float(hist[i]) / total_px,
        }
        for i, name in enumerate(names)
    }
    meta = {
        "tool": "tools/mdas_tool.py build-patches",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "label_path": str(label_path),
        "reference": ref.as_dict(),
        "patch_size": ps,
        "stride": ps,
        "overlap": 0,
        "pad": pad,
        "grid": {"cols": nx, "rows": ny},
        "total_patches": total,
        "written_patches": written,
        "limited_to": args.max_patches,
        "patch_layout": "channel-last [H, W, C]",
        "patch_files": "hsi/ (uint16), msisar/ (float32), lbl/ (uint8)",
        "msi_bands": bands,
        "msi_band_note": "0-based indices into Sentinel-2.tif; [3,2,1,7] = B4/B3/B2/B8",
        "sar": {
            "transform": f"10*log10(max(linear, {args.sar_eps:g}))",
            "pad_value_db": float(args.sar_pad_db),
            "alignment": "index-aligned to HSI grid; not resampled",
        },
        "padding": {
            "inputs": 0,
            "sar_db": float(args.sar_pad_db),
            "label": IGNORE_INDEX,
        },
        "class_scheme": args.class_scheme,
        "class_names": names,
        "ignore_index": IGNORE_INDEX,
        "class_pixels": class_stats,
        "ignore_pixels": int(hist[IGNORE_INDEX]),
        "valid_ratio": {
            "min": min(valid_ratios) if valid_ratios else None,
            "mean": float(np.mean(valid_ratios)) if valid_ratios else None,
            "max": max(valid_ratios) if valid_ratios else None,
        },
        "list_file": "patches.txt",
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\nwritten {written} patches to {out_dir}")
    print(f"  list     -> {out_dir / 'patches.txt'}")
    print(f"  metadata -> {meta_path}")
    print(
        f"  valid label ratio: min={meta['valid_ratio']['min']:.3f} "
        f"mean={meta['valid_ratio']['mean']:.3f} "
        f"max={meta['valid_ratio']['max']:.3f}"
    )
    return 0


def _read_patch_list(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    for line in lines:
        if len(line.split()) != 3:
            raise ValueError(
                f"patch list lines must have three columns, got: {line!r}"
            )
    return lines


def _label_stats(
    lines: list[str],
    patch_root: Path,
    names: list[str],
) -> dict:
    hist = np.zeros(256, dtype=np.int64)
    ratios: list[float] = []
    for line in lines:
        label = tifffile.imread(patch_root / line.split()[2])
        if label.ndim != 2:
            raise ValueError(f"expected a 2-D label patch, got {label.shape}")
        hist += np.bincount(label.ravel(), minlength=256)
        ratios.append(float((label != IGNORE_INDEX).mean()))
    total = int(hist.sum())
    classes = {
        name: {
            "class_id": i,
            "pixels": int(hist[i]),
            "percent": 100.0 * float(hist[i]) / total,
        }
        for i, name in enumerate(names)
    }
    return {
        "patches": len(lines),
        "pixels": total,
        "valid_pixels": int(total - hist[IGNORE_INDEX]),
        "valid_percent": 100.0 * float(total - hist[IGNORE_INDEX]) / total,
        "ignore_pixels": int(hist[IGNORE_INDEX]),
        "classes": classes,
        "valid_ratio_patch": {
            "min": min(ratios) if ratios else None,
            "mean": float(np.mean(ratios)) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
    }


def cmd_split(args: argparse.Namespace) -> int:
    patches_path = Path(args.patches)
    if not patches_path.exists():
        raise FileNotFoundError(patches_path)
    patch_root = patches_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else patch_root
    lines = _read_patch_list(patches_path)

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    if args.min_valid_ratio > 0:
        kept = []
        for line in lines:
            label = tifffile.imread(patch_root / line.split()[2])
            if float((label != IGNORE_INDEX).mean()) >= args.min_valid_ratio:
                kept.append(line)
        filtered = len(lines) - len(kept)
        lines = kept
    else:
        filtered = 0

    if not lines:
        raise ValueError("no patches left after filtering")

    n_total = len(lines)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    val_idx = set(int(i) for i in perm[:n_val])
    train_lines = [line for i, line in enumerate(lines) if i not in val_idx]
    val_lines = [line for i, line in enumerate(lines) if i in val_idx]

    mapping, names = build_mapping(args.class_scheme)
    # Validate that the labels fit the selected scheme.
    sample_label = tifffile.imread(patch_root / lines[0].split()[2])
    valid_classes = sample_label[sample_label != IGNORE_INDEX]
    if valid_classes.size and int(valid_classes.max()) >= len(names):
        raise ValueError(
            f"labels use class ids up to {int(valid_classes.max())}, but "
            f"--class-scheme {args.class_scheme} defines {len(names)} classes"
        )

    train_stats = _label_stats(train_lines, patch_root, names)
    val_stats = _label_stats(val_lines, patch_root, names)

    print(
        f"patch-level random split: {n_total} patches -> "
        f"train {n_train} / val {n_val} "
        f"(seed={args.seed}, val_ratio={args.val_ratio:g}"
        + (f", filtered {filtered} by valid ratio" if filtered else "")
        + ")"
    )
    print(
        f"  train valid {train_stats['valid_percent']:5.2f}%  "
        f"val valid {val_stats['valid_percent']:5.2f}%"
    )
    missing = [
        name for i, name in enumerate(names)
        if val_stats["classes"][name]["pixels"] == 0
    ]
    if missing:
        print(f"  warning: classes missing from val: {', '.join(missing)}")

    meta = {
        "tool": "tools/mdas_tool.py split",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "patches": str(patches_path),
        "out_dir": str(out_dir),
        "split_style": "patch-level random (same style as the default C2Seg split)",
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "min_valid_ratio": args.min_valid_ratio,
        "filtered_by_valid_ratio": filtered,
        "total_patches": n_total,
        "train_patches": n_train,
        "val_patches": n_val,
        "class_scheme": args.class_scheme,
        "class_names": names,
        "ignore_index": IGNORE_INDEX,
        "train": train_stats,
        "val": val_stats,
    }

    if args.dry_run:
        print("[dry-run] no split files written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    meta_path = out_dir / "split_metadata.json"
    if not args.overwrite:
        existing = [p for p in (train_path, val_path, meta_path) if p.exists()]
        if existing:
            raise FileExistsError(
                "split files already exist (use --overwrite): "
                + ", ".join(str(p) for p in existing)
            )
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"written: {train_path} ({n_train})")
    print(f"written: {val_path} ({n_val})")
    print(f"written: {meta_path}")
    return 0


def _fmt_stat_list(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


def _replace_yaml_list(text: str, key: str, values: np.ndarray) -> tuple[str, int]:
    pattern = rf"^(\s*{re.escape(key)}:\s*)\[.*\]\s*$"
    return re.subn(
        pattern,
        rf"\g<1>{_fmt_stat_list(values)}",
        text,
        flags=re.M,
    )


def cmd_stats(args: argparse.Namespace) -> int:
    list_paths = [Path(p) for p in args.lists]
    for path in list_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    roots = {path.parent for path in list_paths}
    if len(roots) != 1:
        raise ValueError("all --lists files must live in the same directory")
    patch_root = next(iter(roots))

    lines: list[str] = []
    seen: set[str] = set()
    for path in list_paths:
        for line in _read_patch_list(path):
            if line not in seen:
                seen.add(line)
                lines.append(line)
    if not lines:
        raise ValueError("no patches found in --lists")

    ms_means: list[np.ndarray] = []
    ms_stds: list[np.ndarray] = []
    hsi_means: list[np.ndarray] = []
    hsi_stds: list[np.ndarray] = []
    for index, line in enumerate(lines):
        parts = line.split()
        ms = tifffile.imread(
            patch_root / parts[0].replace("msi", "msisar")
        ).astype(np.float64)
        hsi = tifffile.imread(
            patch_root / parts[1].replace("sar", "hsi")
        ).astype(np.float64)

        if args.mask == "ignore":
            label = tifffile.imread(patch_root / parts[2])
            valid = label != IGNORE_INDEX
            if not valid.any():
                continue
            ms = ms[valid]
            hsi = hsi[valid]
        else:
            ms = ms.reshape(-1, ms.shape[-1])
            hsi = hsi.reshape(-1, hsi.shape[-1])

        ms_means.append(ms.mean(axis=0))
        ms_stds.append(ms.std(axis=0))
        hsi_means.append(hsi.mean(axis=0))
        hsi_stds.append(hsi.std(axis=0))
        if (index + 1) % 50 == 0:
            print(f"  stats {index + 1}/{len(lines)}")

    if not ms_means:
        raise ValueError("no usable pixels after masking")

    # C2Seg data_sta.py: per-patch mean/std first, then Col_avg across patches.
    mean1 = np.mean(np.stack(ms_means, axis=0), axis=0)
    std1 = np.mean(np.stack(ms_stds, axis=0), axis=0)
    mean2 = np.mean(np.stack(hsi_means, axis=0), axis=0)
    std2 = np.mean(np.stack(hsi_stds, axis=0), axis=0)

    print(
        f"C2Seg-style Col_avg over {len(ms_means)} patches "
        f"(mask={args.mask})"
    )
    print("mean1", _fmt_stat_list(mean1))
    print("std1 ", _fmt_stat_list(std1))
    print("mean2[:5] / [-5:]",
          _fmt_stat_list(mean2[:5]), _fmt_stat_list(mean2[-5:]))
    print("std2[:5]  / [-5:]",
          _fmt_stat_list(std2[:5]), _fmt_stat_list(std2[-5:]))
    if np.any(std1 == 0) or np.any(std2 == 0):
        raise ValueError("computed a zero standard deviation; check the data")
    if len(mean1) != 6 or len(mean2) not in (116, 242):
        print(
            f"warning: unusual channel counts mean1={len(mean1)} "
            f"mean2={len(mean2)}"
        )

    stats_json = (
        Path(args.stats_json)
        if args.stats_json
        else patch_root / "normalize_stats_c2seg.json"
    )
    meta = {
        "tool": "tools/mdas_tool.py stats",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "C2Seg data_sta.py Col_avg: per-patch per-channel mean/std, "
            "then averaged across patches"
        ),
        "mask": args.mask,
        "list_files": [str(path) for path in list_paths],
        "num_patches": len(ms_means),
        "mean1": mean1.tolist(),
        "std1": std1.tolist(),
        "mean2": mean2.tolist(),
        "std2": std2.tolist(),
    }
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with stats_json.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"written: {stats_json}")

    if args.update_config:
        config_path = Path(args.update_config)
        text = config_path.read_text(encoding="utf-8")
        for key, values in (
            ("mean1", mean1),
            ("std1", std1),
            ("mean2", mean2),
            ("std2", std2),
        ):
            text, count = _replace_yaml_list(text, key, values)
            print(f"  {key}: replaced {count} occurrence(s)")
            if count == 0:
                raise ValueError(f"{config_path} has no {key} entry")
        config_path.write_text(text, encoding="utf-8")

        parsed = yaml.safe_load(text)
        for part in ("train_dataset", "val_dataset"):
            transform = parsed[part]["transforms"][0]
            if (
                len(transform["mean1"]) != len(mean1)
                or len(transform["std1"]) != len(std1)
                or len(transform["mean2"]) != len(mean2)
                or len(transform["std2"]) != len(std2)
            ):
                raise ValueError(f"{part} statistics have unexpected lengths")
        print(f"updated: {config_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print grids and offsets")
    info.add_argument("--root", required=True, type=Path, help="raw MDAS root")
    info.set_defaults(func=cmd_info)

    align = sub.add_parser("align-labels", help="align land-use/water to 10 m")
    align.add_argument("--root", required=True, type=Path, help="raw MDAS root")
    align.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "MDAS_10m",
        help="output directory (default: data/MDAS_10m)",
    )
    align.add_argument(
        "--class-scheme",
        choices=("paper14", "all18"),
        default="paper14",
        help="paper14 = 14 classes used in the MDAS paper (default); "
        "all18 = all 18 OSM land-use classes",
    )
    align.add_argument(
        "--with-water",
        action="store_true",
        help="also align the water mask (off by default)",
    )
    align.add_argument(
        "--merge-water",
        action="store_true",
        help="merge the water mask into the label as class num_classes "
        "(implies --with-water)",
    )
    align.add_argument(
        "--water-policy",
        choices=("override", "unlabeled"),
        default="override",
        help="with --merge-water: water overrides land use, or only fills "
        "ignore pixels (default: override)",
    )
    align.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs",
    )
    align.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print statistics without writing files",
    )
    align.set_defaults(func=cmd_align_labels)

    build = sub.add_parser(
        "build-patches",
        help="tile the aligned data into non-overlapping patches",
    )
    build.add_argument("--root", required=True, type=Path, help="raw MDAS root")
    build.add_argument(
        "--labels",
        type=Path,
        default=Path("data") / "MDAS_10m" / "landuse_10m.tif",
        help="aligned land-use label (default: data/MDAS_10m/landuse_10m.tif)",
    )
    build.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "MDAS_64",
        help="output directory (default: data/MDAS_64)",
    )
    build.add_argument("--patch-size", type=int, default=64, help="patch size (default: 64)")
    build.add_argument(
        "--msi-bands",
        type=int,
        nargs=4,
        default=(3, 2, 1, 7),
        metavar=("B1", "B2", "B3", "B4"),
        help="0-based Sentinel-2 band indices (default: 3 2 1 7 = B4/B3/B2/B8)",
    )
    build.add_argument(
        "--sar-eps",
        type=float,
        default=1e-6,
        help="epsilon for the linear-to-dB transform (default: 1e-6)",
    )
    build.add_argument(
        "--sar-pad-db",
        type=float,
        default=0.0,
        help="dB value used in padded SAR pixels (default: 0.0)",
    )
    build.add_argument(
        "--class-scheme",
        choices=("paper14", "all18"),
        default="paper14",
        help="class scheme used by --labels (default: paper14)",
    )
    build.add_argument(
        "--no-pad",
        action="store_true",
        help="drop partial edge tiles instead of padding them",
    )
    build.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="write at most N patches (useful for a quick test)",
    )
    build.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    build.add_argument("--dry-run", action="store_true", help="print the tiling plan only")
    build.set_defaults(func=cmd_build_patches)

    split = sub.add_parser(
        "split",
        help="patch-level random train/val split (C2Seg default style)",
    )
    split.add_argument(
        "--patches",
        type=Path,
        default=Path("data") / "MDAS_64" / "patches.txt",
        help="patch list (default: data/MDAS_64/patches.txt)",
    )
    split.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: the patch list's directory)",
    )
    split.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="validation fraction (default: 0.1)",
    )
    split.add_argument("--seed", type=int, default=0, help="shuffle seed (default: 0)")
    split.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.0,
        help="drop patches below this valid-label ratio before splitting (default: 0)",
    )
    split.add_argument(
        "--class-scheme",
        choices=("paper14", "all18"),
        default="paper14",
        help="class scheme used by the labels (default: paper14)",
    )
    split.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    split.add_argument("--dry-run", action="store_true", help="print the split without writing")
    split.set_defaults(func=cmd_split)

    stats = sub.add_parser(
        "stats",
        help="compute C2Seg-style Normalize2 mean/std for MDAS patches",
    )
    stats.add_argument(
        "--lists",
        nargs="+",
        required=True,
        type=Path,
        help="patch list files to use, e.g. train.txt val.txt",
    )
    stats.add_argument(
        "--mask",
        choices=("none", "ignore"),
        default="none",
        help="none = all pixels (C2Seg data_sta.py default); "
        "ignore = drop label 255 pixels",
    )
    stats.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="output JSON (default: <patch dir>/normalize_stats_c2seg.json)",
    )
    stats.add_argument(
        "--update-config",
        type=Path,
        default=None,
        help="replace mean1/std1/mean2/std2 in this YAML config",
    )
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
