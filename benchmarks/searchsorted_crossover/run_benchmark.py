import argparse
import csv
import ctypes
import json
import math
import os
import platform
import resource
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

libc = ctypes.CDLL(None, use_errno=True)
if hasattr(libc, "mlock"):
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int


def compile_kernel():
    subprocess.run(
        [
            "gcc", "-O3", "-DNDEBUG", "-fPIC", "-shared",
            str(CFILE), "-o", str(LIB),
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


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def command(args):
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def environment_snapshot():
    lscpu = command(["lscpu"])
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uname": command(["uname", "-a"]),
        "lscpu": lscpu,
        "lscpu_cache": command(["lscpu", "-C"]),
        "numactl_hardware": command(["numactl", "--hardware"]),
        "gcc": command(["gcc", "--version"]),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in lscpu.splitlines()
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
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
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


def prefault_array(x):
    # Touch one element per 4 KiB page plus the final element.
    stride = max(1, 4096 // x.itemsize)
    total = int(x[::stride].sum(dtype=np.int64))
    if x.size:
        total ^= int(x[-1])
    return total


def lock_array(x):
    if not hasattr(libc, "mlock"):
        return {"ok": False, "errno": None, "message": "mlock unavailable"}
    rc = libc.mlock(ctypes.c_void_p(x.ctypes.data), ctypes.c_size_t(x.nbytes))
    if rc == 0:
        return {"ok": True, "errno": 0, "message": "locked"}
    err = ctypes.get_errno()
    return {"ok": False, "errno": err, "message": os.strerror(err)}


def faults():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"minor": r.ru_minflt, "major": r.ru_majflt}


def timed(fn, reps, warmups):
    for _ in range(warmups):
        fn()

    before = faults()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    after = faults()

    median = statistics.median(samples)
    return {
        "median_ns": median,
        "min_ns": min(samples),
        "max_ns": max(samples),
        "mad_ns": statistics.median(abs(x - median) for x in samples),
        "minor_faults": after["minor"] - before["minor"],
        "major_faults": after["major"] - before["major"],
        "samples_ns": samples,
    }


def prepare_memory(mode, a, q, out):
    info = {"mode": mode, "prefault_checksum": None, "mlock": []}

    if mode in ("prefault", "mlock"):
        checksum = prefault_array(a)
        checksum ^= prefault_array(q)
        out.fill(0)
        checksum ^= prefault_array(out)
        info["prefault_checksum"] = checksum

    if mode == "mlock":
        for name, arr in (("a", a), ("q", q), ("out", out)):
            result = lock_array(arr)
            result["array"] = name
            result["bytes"] = arr.nbytes
            info["mlock"].append(result)

    return info


def one(merge_fn, n, density, seed, reps, warmups, memory_mode):
    m = max(1, round(n * density))
    a = np.arange(n, dtype=np.int64) * 2
    rng = np.random.default_rng(seed)
    q = np.sort(rng.integers(0, 2 * n + 1, size=m, dtype=np.int64))
    out = np.empty(m, dtype=np.int64)

    memory_info = prepare_memory(memory_mode, a, q, out)

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
        "memory": memory_info,
        "numpy": timed(numpy_searchsorted, reps, warmups),
        "merge": timed(merge, reps, warmups),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="benchmark-output")
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument(
        "--memory-mode",
        choices=("normal", "prefault", "mlock"),
        default="normal",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    compile_kernel()
    merge_fn = load_merge()

    ns = [
        1_000_000, 1_500_000, 2_000_000, 2_500_000,
        3_000_000, 3_500_000, 4_000_000, 4_500_000,
        5_000_000, 5_500_000, 6_000_000, 7_000_000,
        8_000_000,
    ]
    densities = [
        0.005, 0.0075, 0.010, 0.0125, 0.015,
        0.0175, 0.020, 0.025, 0.030, 0.040,
    ]
    seeds = [20261002, 20261003, 20261004]

    env = environment_snapshot()
    (outdir / "environment.json").write_text(json.dumps(env, indent=2))

    rows = []
    raw = []
    mlock_failures = []

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
                    args.memory_mode,
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

                for item in result["memory"]["mlock"]:
                    if not item["ok"]:
                        mlock_failures.append(
                            {
                                "n": n,
                                "density": density,
                                "seed": seed,
                                **item,
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
                    "working_set_bytes_min": (
                        n * 8
                        + max(1, round(n * density)) * 8
                        + max(1, round(n * density)) * 8
                    ),
                    "theory_r": theory,
                    "memory_mode": args.memory_mode,
                    "numpy_median_ns": numpy_median,
                    "merge_median_ns": merge_median,
                    "merge_over_numpy": merge_median / numpy_median,
                    "numpy_minor_faults": sum(
                        r["numpy"]["minor_faults"] for r in seed_results
                    ),
                    "numpy_major_faults": sum(
                        r["numpy"]["major_faults"] for r in seed_results
                    ),
                    "merge_minor_faults": sum(
                        r["merge"]["minor_faults"] for r in seed_results
                    ),
                    "merge_major_faults": sum(
                        r["merge"]["major_faults"] for r in seed_results
                    ),
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
        previous = None
        bracket = None

        for row in subset:
            wins = row["merge_median_ns"] < row["numpy_median_ns"]
            if wins and previous is None:
                bracket = [0.0, row["density"]]
                break
            if wins and previous is not None and not previous["wins"]:
                bracket = [previous["density"], row["density"]]
                break
            previous = {"density": row["density"], "wins": wins}

        brackets[str(n)] = {
            "theory_r": 1.0 / (math.log2(n) - 1.0),
            "measured_bracket": bracket,
        }

    summary = {
        "environment": env,
        "parameters": {
            "reps": args.reps,
            "warmups": args.warmups,
            "memory_mode": args.memory_mode,
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
        "mlock_failures": mlock_failures,
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
