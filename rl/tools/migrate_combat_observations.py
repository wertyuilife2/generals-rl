"""Create combat weights for a fresh run with train --config ... --warm-start ... ."""
import argparse
import json
from pathlib import Path

from generals_rl.combat_migration import migrate_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="original training checkpoint or frozen snapshot")
    parser.add_argument("--config", type=Path, help="source config; defaults to run_config.json beside the checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new directory for checkpoint.pt and run_config.json")
    parser.add_argument("--king-threat-distance", action="store_true",
                        help="fold seat-P1 into P0+bias; retain current local/combat map planes")
    args = parser.parse_args(argv)
    report = migrate_checkpoint(args.checkpoint, args.output, config_path=args.config,
                                king_threat_distance=args.king_threat_distance)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
