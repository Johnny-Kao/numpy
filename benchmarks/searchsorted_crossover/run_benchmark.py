import argparse
import csv
import ctypes
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE / "libbulk_search_bench.so"
CFILE = HERE / "bench_bulk_search.c"

P64 = ctypes.POINTER(ctypes.c_int64)


def compile_kernel():
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            str(CFILE),
            "-o",
            str(LIB),
        ],
        check=True,
    )


def load_merge():
    lib = ctypes.CDLL(str(LIB))
    fn = lib.merge_search_left_i64
    fn.argtypes = [P64, ctypes.c_size_t, P64, ctypes.c_size_t, P64]
    fn.restype = None
    return fn


def ptr(x):
    return x.ctypes.data_as(P64)


def timed(fn, reps, warmups):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return {
        "median_ns": statistics.median(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "mad_ns": statistics.median(
            abs(x - statistics.median(samples)) for x in samples
        ),
        "samples_ns": samples,
    }


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def environment_snapshot():
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return f"ERROR: {exc}"

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": command(["uname", "-a"]),
        "lscpu": command(["lscpu"]),
        "gcc": command(["gcc", "--version"]),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in command(["lscpu"]).splitlines()
                if line.lower().startswith("model name")
            ),
            None,
        ),
        "microcode": read_text("/sys/devices/system/cpu/cpu0/microcode/version"),
        "scaling_governor": read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "scaling_driver": read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"
        ),
        "meminfo": read_text("/proc/meminfo"),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "github": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "job": os.environ.get("GITHUB_JOB"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
        },
    }


def one(merge_fn, n, density, seed, reps, warmups):
    m = max(1, round(n * density))
    a = np.arange(n, dtype=np.int64) * 2
    rng = np.random.default_rng(seed)
    q = np.sort(rng.integers(0, 2 * n + 1, size=m, dtype=np.int64))
    out = np.empty(m, dtype=np.int64)

    pa, pq, po = ptr(a), ptr(q), ptr(out)

    def merge():
        merge_fn(pa, n, pq, m, po)

    expected = np.searchsorted(a, q, side="left")
    merge()
    if not np.array_equal(out, expected):
        raise RuntimeError("merge kernel mismatch")

    def numpy_searchsorted():
        np.searchsorted(a, q, side="left")

    return {
        "numpy": timed(numpy_searchsorted, reps, warmups),
        "merge": timed(merge, reps, warmups),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark-output")
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    compile_kernel()
    merge_fn = load_merge()

    ns = [32768, 100000, 1000000, 5000000]
    densities = [
        0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
        0.040, 0.045, 0.050, 0.055, 0.060, 0.070, 0.080,
    ]
    seeds = [20261002, 20261003, 20261004, 20261005, 20261006]

    env = environment_snapshot()
    (outdir / "environment.json").write_text(json.dumps(env, indent=2))

    rows = []
    raw = []
    for n in ns:
        theory = 1.0 / (math.log2(n) - 1.0)
        for density in densities:
            seed_results = []
            for seed in seeds:
                result = one(
                    merge_fn,
                    n,
                    density,
                    seed + n,
                    args.reps,
                    args.warmups,
                )
                seed_results.append(result)
                raw.append(
                    {
                        "n": n,
                        "density": density,
                        "seed": seed,
                        "theory_r": theory,
                        **result,
                    }
                )

            numpy_median = statistics.median(
                r["numpy"]["median_ns"] for r in seed_results
            )
            merge_median = statistics.median(
                r["merge"]["median_ns"] for r in seed_results
            )
            rows.append(
                {
                    "n": n,
                    "density": density,
                    "m": max(1, round(n * density)),
                    "theory_r": theory,
                    "numpy_median_ns": numpy_median,
                    "merge_median_ns": merge_median,
                    "merge_over_numpy": merge_median / numpy_median,
                }
            )

    with (outdir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    (outdir / "raw_samples.json").write_text(json.dumps(raw, indent=2))

    brackets = {}
    for n in ns:
        subset = [r for r in rows if r["n"] == n]
        prev = None
        bracket = None
        for row in subset:
            wins = row["merge_median_ns"] < row["numpy_median_ns"]
            if wins and prev is None:
                bracket = [0.0, row["density"]]
                break
            if wins and prev is not None and not prev["wins"]:
                bracket = [prev["density"], row["density"]]
                break
            prev = {"density": row["density"], "wins": wins}
        brackets[str(n)] = {
            "theory_r": 1.0 / (math.log2(n) - 1.0),
            "measured_bracket": bracket,
        }

    summary = {
        "environment": env,
        "parameters": {
            "reps": args.reps,
            "warmups": args.warmups,
            "seeds": seeds,
            "ns": ns,
            "densities": densities,
            "dtype": "int64",
            "side": "left",
            "query_distribution": "sorted uniform random integers",
            "query_generation_in_timed_region": False,
            "compiler_flags": "-O3 -DNDEBUG -fPIC -shared",
        },
        "crossovers": brackets,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
