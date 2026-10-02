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

ROOT = Path(__file__).resolve().parents[2]
CFILE = ROOT / "benchmarks/searchsorted_crossover/bench_bulk_search.c"
LIB = Path("/tmp/libbulk_search_llc.so")
P64 = ctypes.POINTER(ctypes.c_int64)


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def compile_kernel():
    subprocess.run(
        ["gcc", "-O3", "-DNDEBUG", "-fPIC", "-shared", str(CFILE), "-o", str(LIB)],
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


def prefault(x):
    stride = max(1, 4096 // x.itemsize)
    total = int(x[::stride].sum(dtype=np.int64))
    if x.size:
        total ^= int(x[-1])
    return total


def faults():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_minflt, r.ru_majflt


def timed(fn, reps=31, warmups=8):
    for _ in range(warmups):
        fn()
    bmin, bmaj = faults()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    amin, amaj = faults()
    med = statistics.median(samples)
    return {
        "median_ns": med,
        "mad_ns": statistics.median(abs(x - med) for x in samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "minor_faults": amin - bmin,
        "major_faults": amaj - bmaj,
    }


def bench(merge_fn, n, density, seed, mode):
    m = max(1, round(n * density))
    a = np.arange(n, dtype=np.int64) * 2
    rng = np.random.default_rng(seed)
    q = np.sort(rng.integers(0, 2 * n + 1, size=m, dtype=np.int64))
    out = np.empty(m, dtype=np.int64)

    if mode == "prefault":
        _ = prefault(a) ^ prefault(q)
        out.fill(0)
        _ ^= prefault(out)

    pa, pq, po = ptr(a), ptr(q), ptr(out)

    def merge():
        merge_fn(pa, n, pq, m, po)

    expected = np.searchsorted(a, q, side="left")
    merge()
    if not np.array_equal(out, expected):
        raise RuntimeError("merge mismatch")

    def numpy_search():
        np.searchsorted(a, q, side="left")

    return timed(numpy_search), timed(merge)


def env_snapshot():
    lscpu = command(["lscpu"])
    return {
        "cpu_model": next(
            (line.split(":", 1)[1].strip()
             for line in lscpu.splitlines()
             if line.lower().startswith("model name")),
            None,
        ),
        "lscpu": lscpu,
        "lscpu_cache": command(["lscpu", "-C"]),
        "numactl": command(["numactl", "--hardware"]),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "prefault"], required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compile_kernel()
    merge_fn = load_merge()

    ns = list(range(3_500_000, 4_500_001, 100_000))
    densities = [0.0075, 0.0100, 0.0125, 0.0150, 0.0175]
    seeds = [20261002, 20261003, 20261004, 20261005, 20261006]

    env = env_snapshot()
    (out / "environment.json").write_text(json.dumps(env, indent=2))

    rows = []
    for n in ns:
        theory = 1 / (math.log2(n) - 1)
        for d in densities:
            nvals, mvals = [], []
            n_faults = [0, 0]
            m_faults = [0, 0]
            n_mads, m_mads = [], []

            for seed in seeds:
                nr, mr = bench(merge_fn, n, d, seed + n, args.mode)
                nvals.append(nr["median_ns"])
                mvals.append(mr["median_ns"])
                n_mads.append(nr["mad_ns"])
                m_mads.append(mr["mad_ns"])
                n_faults[0] += nr["minor_faults"]; n_faults[1] += nr["major_faults"]
                m_faults[0] += mr["minor_faults"]; m_faults[1] += mr["major_faults"]

            nmed = statistics.median(nvals)
            mmed = statistics.median(mvals)
            rows.append({
                "n": n,
                "haystack_bytes": n * 8,
                "haystack_mib": n * 8 / (1024 * 1024),
                "density": d,
                "m": round(n * d),
                "theory_r": theory,
                "mode": args.mode,
                "numpy_median_ns": nmed,
                "merge_median_ns": mmed,
                "merge_over_numpy": mmed / nmed,
                "numpy_seed_spread": max(nvals) / min(nvals),
                "merge_seed_spread": max(mvals) / min(mvals),
                "numpy_mad_ns_median": statistics.median(n_mads),
                "merge_mad_ns_median": statistics.median(m_mads),
                "numpy_minor_faults": n_faults[0],
                "numpy_major_faults": n_faults[1],
                "merge_minor_faults": m_faults[0],
                "merge_major_faults": m_faults[1],
            })

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    crossings = {}
    for n in ns:
        sub = [r for r in rows if r["n"] == n]
        prev = None
        br = None
        for r in sub:
            win = r["merge_over_numpy"] < 1.0
            if win and prev is None:
                br = [0.0, r["density"]]
                break
            if win and prev is not None and not prev["win"]:
                br = [prev["density"], r["density"]]
                break
            prev = {"density": r["density"], "win": win}
        crossings[str(n)] = br

    summary = {
        "environment": env,
        "mode": args.mode,
        "ns": ns,
        "densities": densities,
        "crossovers": crossings,
        "all_major_faults_zero": all(
            r["numpy_major_faults"] == 0 and r["merge_major_faults"] == 0 for r in rows
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
