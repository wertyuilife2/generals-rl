"""Measure complete map generation; optionally compare a compatible Git revision."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from generals_env import MapConfig, generate_map


def baseline_generator(ref):
    """Load only the old mapgen module; keep other environment code identical."""
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
        cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    path = f"{revision}:env/src/generals_env/mapgen.py"
    source = subprocess.run(["git", "show", path], cwd=root, check=True,
                            capture_output=True, text=True).stdout
    namespace = {"__name__": "generals_env._benchmark_baseline", "__package__": "generals_env"}
    exec(compile(source, path, "exec"), namespace)
    return revision, namespace["generate_map"]


def summarize(milliseconds):
    return {"median_ms": float(np.median(milliseconds)),
            "p95_ms": float(np.percentile(milliseconds, 95)),
            "min_ms": min(milliseconds), "max_ms": max(milliseconds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0, help="first of consecutive sample seeds")
    parser.add_argument("--baseline-ref", help="optional trusted Git revision with a compatible mapgen API")
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error("samples must be positive and seed nonnegative")
    generators = {"current": generate_map}
    revision = None
    if args.baseline_ref:
        try:
            revision, generators["baseline"] = baseline_generator(args.baseline_ref)
        except (OSError, subprocess.CalledProcessError) as error:
            parser.error(f"cannot load baseline: {error}")
    rows = []
    names = list(generators)
    for size, players in ((15, 2), (25, 4), (35, 8)):
        for density in (0.2, 0.4):
            config = MapConfig(size, size, players, density, 0.1)
            for generator in generators.values():
                for seed in range(args.seed, args.seed + 3):
                    generator(config, seed)
            times = {name: [] for name in names}
            for sample in range(args.samples):
                # Alternate methods to reduce systematic order effects.
                for name in names if sample % 2 == 0 else reversed(names):
                    started = perf_counter()
                    generators[name](config, args.seed + sample)
                    times[name].append((perf_counter() - started) * 1000)
            rows.append({"config": asdict(config),
                         **{name: summarize(values) for name, values in times.items()}})
    print(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                      "platform": platform.platform(), "processor": platform.processor(),
                      "baseline_ref": revision, "samples_per_config": args.samples,
                      "first_seed": args.seed, "warmup_maps_per_method": 3,
                      "timing_scope": "complete generation, including state validation; no GUI or AI",
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
