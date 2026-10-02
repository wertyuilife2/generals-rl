"""Source-tree entry point for RL-friendly env.step(action) measurements."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from generals.benchmarks.step import main

if __name__ == "__main__":
    raise SystemExit(main())
