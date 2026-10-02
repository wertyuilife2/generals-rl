"""Source-tree entry point for CPU/GPU step profiling."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generals.benchmarks.profiling import main

if __name__ == "__main__":
    raise SystemExit(main())
