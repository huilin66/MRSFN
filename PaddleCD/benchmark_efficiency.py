# Efficiency benchmark: Params (M), FLOPs (G), and FPS for the paper's
# model tables, all measured in a single environment within one run.
#
# Motivation: the FPS values currently in the paper were logged incidentally
# during full-validation runs (`paddleseg/core/val.py`), once per experiment,
# on whatever hardware each experiment happened to use. This script instead
# benchmarks every configuration with synthetic inputs of the exact validation
# shapes, a warmup phase, and repeated timed runs, so all rows share one
# device, one paddle build, and one timing protocol. Params and FLOPs use the
# same conventions as val.py, so those columns stay comparable with the
# previously reported values.
#
# Usage (from the repository root, on the training server):
#   python PaddleCD/benchmark_efficiency.py --preset all
#   python PaddleCD/benchmark_efficiency.py --preset main_bw --warmup 5 --iters 20 --repeats 1
#
# Outputs land in <repo_root>/benchmark/: one CSV with per-run FPS values and
# one Markdown file with paper-style tables plus the environment block.

import argparse
import csv
import functools
import os
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime

import paddle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paddleseg.cvlibs import Config  # noqa: E402

# Validation-input shapes per subset: (im1 channels, im2 channels, H, W).
# BW feeds 6 MSI/SAR channels + 116 HSI channels at 256x256;
# AB feeds 6 MSI/SAR channels + 242 HSI channels at 128x128.
SUBSET_SHAPES = {
    "BW": (6, 116, 256, 256),
    "AB": (6, 242, 128, 128),
}

# Paper rows mapped to their configuration files. Configs live in
# PaddleCD/c2seg_config/. Entries marked with a comment reuse the
# architecture of a missing config file (loss-only or naming differences
# do not change Params/FLOPs/FPS).
PRESETS = {
    "main_bw": [
        ("UNet", "unet_BW", "BW"),
        ("DeepLabV3+", "deeplabv3p_BW", "BW"),
        ("OCRNet", "ocrnet_BW", "BW"),
        ("SegFormer", "segformer_BW", "BW"),
        ("HighDAN", "highdan_BW", "BW"),
        ("CMX (MiT-B4)", "cmx_4b_BW", "BW"),
        ("UPerNet (1B)", "cxup_1b_BW", "BW"),
        ("MRSN (4B2H)", "MRSN_BW", "BW"),
        ("MRSFN (full)", "cxup_4b_BW_PMRG_v2_lossV2", "BW"),
    ],
    "main_ab": [
        ("UNet", "unet_AB", "AB"),
        ("DeepLabV3+", "deeplabv3p_AB", "AB"),
        ("OCRNet", "ocrnet_AB", "AB"),
        ("SegFormer", "segformer_AB", "AB"),
        ("HighDAN", "highdan_AB", "AB"),
        ("CMX (MiT-B4)", "cmx_4b_AB", "AB"),
        ("UPerNet (1B)", "cxup_1b_AB", "AB"),
        ("MRSN (4B2H)", "MRSN_AB", "AB"),
        ("MRSFN (full)", "cxup_4b_AB_PMRG_v2_lossV2", "AB"),
    ],
    "ablation_bw": [
        ("1-branch (UPerNet ref.)", "cxup_1b_BW", "BW"),
        ("2-branch", "cxup_2b_BW", "BW"),
        ("3-branch", "cxup_3b_BW", "BW"),
        ("4-branchv0", "cxup_4b_BW_RGB_NIR", "BW"),
        ("4-branch", "cxup_4b_BW", "BW"),
        # cxup_4b_BW_PMRG_v2.yml is not in the repo; the city variant uses the
        # same CX_Uper_4B_PMRG_V2 architecture (dataset section is unused here).
        ("4-branch + PMRG", "cxup_4b_BW_PMRG_city", "BW"),
        # ML changes only the training loss; the architecture equals plain 4B.
        ("4-branch + ML", "cxup_4b_BW", "BW"),
        ("MRSFN (4-branch + PMRG + ML)", "cxup_4b_BW_PMRG_v2_lossV2", "BW"),
    ],
    "backbone_bw": [
        ("UPerNet (1 branch)", "cxup_1b_BW", "BW"),
        ("1 branch", "cxup_1b_BW_small", "BW"),
        ("2 branches", "cxup_2b_BW", "BW"),
        ("3 branches", "cxup_3b_BW", "BW"),
        ("1 branch", "cxup_1b_BW_base", "BW"),
        ("2 branches", "cxup_2b_BW_small", "BW"),
        ("4-branchv0", "cxup_4b_BW_RGB_NIR", "BW"),
        ("4-branch", "cxup_4b_BW", "BW"),
        ("1 branch", "cxup_1b_BW_large", "BW"),
    ],
    "city_bw": [
        ("UNet", "unet_BW_city", "BW"),
        ("DeepLabV3+", "deeplabv3p_BW_city", "BW"),
        ("OCRNet", "ocrnet_BW_city", "BW"),
        ("SegFormer", "segformer_BW_city", "BW"),
        ("HighDAN", "highdan_BW_city", "BW"),
        ("CMX (MiT-B4)", "cmx_4b_BW_city", "BW"),
        ("UPerNet (1B)", "cxup_1b_BW_city", "BW"),
        ("MRSN", "MRSN_BW_city", "BW"),
        ("MRSFN", "cxup_4b_BW_PMRG_v2_lossV2_city", "BW"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Params/FLOPs/FPS benchmark")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS) + ["all"],
        default="all",
        help="Which paper table(s) to benchmark. Default: all.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the FPS measurement. Default: 1, matching the "
             "validation inference protocol.")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup iterations before timing. Default: 20.")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations per repeat. Default: 100.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of timed repeats. Default: 3.")
    parser.add_argument(
        "--out",
        default="efficiency_benchmark",
        help="Output file prefix under benchmark/. Default: efficiency_benchmark.")
    parser.add_argument(
        "--flops_backend",
        choices=["auto", "slim", "paddle"],
        default="auto",
        help="FLOPs counter. 'auto' uses paddleslim (the val.py method) on "
             "Linux and paddle.flops elsewhere; paddleslim can hard-crash "
             "the interpreter on some non-Linux paddle builds. Default: auto.")
    return parser.parse_args()


def strip_pretrained(obj):
    """Recursively set `pretrained: None` inside a model spec dict.

    Benchmarking never needs trained weights, and some configs embed
    pretrained URLs that would trigger downloads at construction time.
    None (not False) is the value that skips loading cleanly: the shared
    `load_pretrained_model` helper only checks `is not None`, so False
    would fall into the URL-handling branch and crash.
    """
    if isinstance(obj, dict):
        return {
            key: (None if key == "pretrained" else strip_pretrained(val))
            for key, val in obj.items()
        }
    if isinstance(obj, list):
        return [strip_pretrained(val) for val in obj]
    return obj


def disable_pretrained_weights():
    """Force `pretrained=False` inside the ConvNeXt factory.

    CX_Uper-family constructors request pretrained ConvNeXt weights through
    the module-level `_convnext` helper regardless of the config contents
    (e.g. MRSN's `CX_Uper_4B2H` hardcodes `convnext_tiny()`), so patching
    the helper is the only config-independent way to skip weight loading.
    """
    import paddleseg.models.backbones.convnext as convnext_mod

    if getattr(convnext_mod, "_benchmark_patched", False):
        return
    original = convnext_mod._convnext

    @functools.wraps(original)
    def patched(arch, depths, *args, **kwargs):
        kwargs["pretrained"] = False
        return original(arch, depths, **kwargs)

    convnext_mod._convnext = patched
    convnext_mod._benchmark_patched = True


def build_model(config_name):
    """Build the model from a c2seg config without touching datasets."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "c2seg_config",
        "{}.yml".format(config_name))
    cfg = Config(config_path)
    # Strip pretrained keys before the lazy `cfg.model` property builds the
    # network, so no checkpoint download or local weight lookup happens.
    cfg.dic["model"] = strip_pretrained(cfg.dic.get("model", {}).copy())
    model = cfg.model
    model.eval()
    return model, type(model).__name__


def count_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(
        p.numel() for p in model.parameters() if not p.stop_gradient)
    return num_params, num_trainable


def slim_flops_available():
    """Probe paddleslim in a subprocess.

    Importing paddleslim can crash the interpreter outright on some
    paddle builds (a hard segfault cannot be caught in-process), so the
    availability check runs in a child process. When the probe fails, the
    paddle.flops fallback is used instead of risking the whole sweep.
    """
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             "from paddleslim.analysis import flops"],
            capture_output=True,
            timeout=120)
        return probe.returncode == 0
    except Exception:
        return False


def measure_flops(model, shape1, shape2, use_slim=True):
    """FLOPs with the paddleslim call used by paddleseg/core/val.py."""
    inputs = [[1] + list(shape1), [1] + list(shape2)]
    if use_slim:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='When training, we now always track global mean and variance.')
                from paddleslim.analysis import flops as slim_flops
                return float(slim_flops(model, inputs=inputs))
        except Exception:
            pass
    # Fallback mirroring val.py: concatenate the two streams into one input.
    try:
        c1 = shape1[0]
        c_total = shape1[0] + shape2[0]
        h, w = shape1[1], shape1[2]

        class _FlopsWrapper(paddle.nn.Layer):
            def __init__(self, inner):
                super().__init__()
                self.model = inner
            def forward(self, x):
                return self.model(x[:, :c1], x[:, c1:])

        wrapper = _FlopsWrapper(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='When training, we now always track global mean and variance.')
            try:
                flops = paddle.flops(
                    wrapper, [1, c_total, h, w], verbose=False)
            except TypeError:
                # Older paddle.flops signatures lack the verbose kwarg.
                flops = paddle.flops(wrapper, [1, c_total, h, w])
        return float(flops)
    except Exception:
        return None


def benchmark_fps(model, im1, im2, warmup, iters, repeats, use_cuda):
    """Return per-repeat FPS values (samples per second)."""
    sync = _cuda_sync(use_cuda)
    fps_values = []
    with paddle.no_grad():
        for _ in range(warmup):
            model(im1, im2)
        sync()
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iters):
                model(im1, im2)
            sync()
            elapsed = time.perf_counter() - start
            fps_values.append(iters * im1.shape[0] / elapsed)
    return fps_values


def _cuda_sync(use_cuda):
    """CUDA synchronize across paddle versions (API moved in 2.5)."""
    if not use_cuda:
        return lambda: None
    if hasattr(paddle.device, "synchronize"):
        return paddle.device.synchronize
    return paddle.device.cuda.synchronize


def gpu_name():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return name
    except Exception:
        pass
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=30)
        if probe.returncode == 0:
            return probe.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    return "unknown"


def environment_block(args, device, flops_backend):
    try:
        cuda_compiled = paddle.device.is_compiled_with_cuda()
    except Exception:
        cuda_compiled = False
    lines = [
        "- Paddle: {} (CUDA compiled: {})".format(
            paddle.__version__, cuda_compiled),
        "- Device: {}".format(device),
        "- GPU: {}".format(gpu_name() if device == "gpu" else "n/a"),
        "- Precision: FP32",
        "- FLOPs backend: {}".format(flops_backend),
        "- Batch size: {}".format(args.batch_size),
        "- Warmup: {} iters, timed: {} iters x {} repeats".format(
            args.warmup, args.iters, args.repeats),
        "- Date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    try:
        paddle.set_device("gpu")
        device = "gpu"
    except Exception:
        paddle.set_device("cpu")
        device = "cpu"
        print("[bench] GPU unavailable; falling back to CPU. "
              "FPS values will not be comparable with GPU measurements.")
    use_cuda = device == "gpu"
    if use_cuda and args.batch_size > 1:
        # Larger batches only make sense if they fit; keep the protocol
        # explicit rather than silently OOMing midway through the sweep.
        print("[bench] batch_size={} requested; ensure it fits all models".format(
            args.batch_size))

    disable_pretrained_weights()

    if args.flops_backend == "slim":
        slim_ok = True
    elif args.flops_backend == "paddle":
        slim_ok = False
    else:
        # 'auto': paddleslim (the val.py method) on Linux, where it is
        # stable; paddle.flops elsewhere, where importing paddleslim can
        # segfault the interpreter after CUDA initialization.
        slim_ok = sys.platform.startswith("linux") and slim_flops_available()
    flops_backend = "paddleslim" if slim_ok else "paddle.flops"
    print("[bench] FLOPs backend: {}".format(flops_backend))

    if args.preset == "all":
        presets = dict(PRESETS)
    else:
        presets = {args.preset: PRESETS[args.preset]}

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "{}.csv".format(args.out))
    md_path = os.path.join(out_dir, "{}.md".format(args.out))

    # Stream rows to disk as they complete so a crash mid-sweep keeps the
    # measurements taken so far.
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_handle)
    write_csv_header(csv_writer, args)

    # Benchmark each (config, subset) pair once, even when several presets
    # reference the same architecture.
    cache = {}
    rows = []  # (preset, row_name, config, subset, result dict)

    for preset_name, entries in presets.items():
        for row_name, config_name, subset in entries:
            key = (config_name, subset)
            if key not in cache:
                try:
                    cache[key] = benchmark_one(
                        config_name, subset, args, device, slim_ok)
                except Exception as exc:  # keep the sweep running
                    cache[key] = {"error": "{}: {}".format(type(exc).__name__, exc)}
            result = cache[key]
            rows.append((preset_name, row_name, config_name, subset, result))
            status = "OK" if "error" not in result else "ERROR"
            suffix = "" if status == "OK" else " | {}".format(result["error"])
            print("[bench] [{}] {} <- {} ({}){}".format(
                status, row_name, config_name, subset, suffix))
            append_csv_row(csv_writer, preset_name, row_name, config_name,
                           subset, result, args)
            csv_handle.flush()

    csv_handle.close()
    write_markdown(md_path, presets, rows, args, device, flops_backend)
    print("[bench] CSV: {}\n[bench] Markdown: {}".format(csv_path, md_path))


def benchmark_one(config_name, subset, args, device, slim_ok=True):
    use_cuda = device == "gpu"
    shape1_ch, shape2_ch, height, width = SUBSET_SHAPES[subset]

    model, model_type = build_model(config_name)
    num_params, num_trainable = count_params(model)

    # FPS first: paddleslim's FLOPs tracing perturbs layer state (conv2d
    # starts rejecting EagerParamBase weights afterwards), so all timed
    # forwards must happen before it. This mirrors val.py, which also
    # computes FLOPs after all inference.
    im1 = paddle.randn([args.batch_size, shape1_ch, height, width])
    im2 = paddle.randn([args.batch_size, shape2_ch, height, width])
    if use_cuda:
        im1 = im1.cuda()
        im2 = im2.cuda()

    fps_values = benchmark_fps(
        model, im1, im2, args.warmup, args.iters, args.repeats, use_cuda)

    flops = measure_flops(
        model, (shape1_ch, height, width), (shape2_ch, height, width),
        use_slim=slim_ok)

    result = {
        "model_type": model_type,
        "params_m": num_params / 1e6,
        "trainable_m": num_trainable / 1e6,
        "flops_g": flops / 1e9 if flops is not None else None,
        "fps_runs": fps_values,
        "fps_mean": statistics.mean(fps_values),
        "fps_std": statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0,
    }

    del model, im1, im2
    if use_cuda:
        paddle.device.cuda.empty_cache()

    print("[bench]   built {} | Params {:.2f}M | FLOPs {} | FPS {:.2f} +- {:.2f}".format(
        model_type, result["params_m"],
        "{:.2f}G".format(result["flops_g"]) if result["flops_g"] is not None else "N/A",
        result["fps_mean"], result["fps_std"]))
    return result


def write_csv_header(csv_writer, args):
    header = (["preset", "row", "config", "subset", "model_type",
               "params_M", "trainable_M", "flops_G"]
              + ["fps_run{}".format(i + 1) for i in range(args.repeats)]
              + ["fps_mean", "fps_std", "error"])
    csv_writer.writerow(header)


def append_csv_row(csv_writer, preset_name, row_name, config_name, subset,
                   result, args):
    fps_runs = result.get("fps_runs", [])
    fps_runs = list(fps_runs) + [""] * (args.repeats - len(fps_runs))
    csv_writer.writerow(
        [preset_name, row_name, config_name, subset,
         result.get("model_type", ""),
         "{:.2f}".format(result["params_m"]) if "params_m" in result else "",
         "{:.2f}".format(result["trainable_m"]) if "trainable_m" in result else "",
         "{:.2f}".format(result["flops_g"]) if result.get("flops_g") is not None else "",
         *["{:.2f}".format(v) if v != "" else "" for v in fps_runs],
         "{:.2f}".format(result["fps_mean"]) if "fps_mean" in result else "",
         "{:.2f}".format(result["fps_std"]) if "fps_std" in result else "",
         result.get("error", "")])


def write_markdown(md_path, presets, rows, args, device, flops_backend):
    grouped = {name: [] for name in presets}
    for entry in rows:
        if entry[0] in grouped:
            grouped[entry[0]].append(entry)

    lines = [
        "# Efficiency benchmark",
        "",
        "All rows measured in one environment within one run. Synthetic "
        "inputs of the exact validation shapes; models in eval mode; "
        "`no_grad`; CUDA-synchronized timing.",
        "",
        environment_block(args, device, flops_backend),
        "",
    ]
    for preset_name, entries in grouped.items():
        lines += ["## {}".format(preset_name), ""]
        lines += [
            "| Model | Config | Params (M) | FLOPs (G) | FPS |",
            "|---|---|---:|---:|---:|",
        ]
        for _, row_name, config_name, _, result in entries:
            if "error" in result:
                lines.append("| {} | `{}` | ERROR | ERROR | ERROR |".format(
                    row_name, config_name))
                continue
            flops_str = ("{:.2f}".format(result["flops_g"])
                         if result["flops_g"] is not None else "N/A")
            if args.repeats > 1:
                fps_str = "{:.2f} ± {:.2f}".format(
                    result["fps_mean"], result["fps_std"])
            else:
                fps_str = "{:.2f}".format(result["fps_mean"])
            lines.append("| {} | `{}` | {:.2f} | {} | {} |".format(
                row_name, config_name, result["params_m"], flops_str, fps_str))
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
