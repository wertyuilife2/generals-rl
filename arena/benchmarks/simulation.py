"""Reproduce a fixed 8-player game; verify every action and every state digest.

  python arena/benchmarks/simulation.py
  python arena/benchmarks/simulation.py --reference 870e764

The optional reference loads only the baseline package from a Git revision into
an isolated temporary directory. It never checks out files or changes the repo.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',help='baseline Git revision to benchmark')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='arena-baseline-bench-') as temporary:
        if args.reference:
            prefix='baselines/src/'
            names=subprocess.check_output(['git','ls-tree','-r','--name-only',args.reference,'--',prefix],cwd=root,text=True).splitlines()
            for name in names:
                if not name.endswith('.py'):continue
                target=Path(temporary)/Path(name).relative_to(prefix)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(subprocess.check_output(['git','show',f'{args.reference}:{name}'],cwd=root))
            sys.path.insert(0,temporary)
        from generals_baselines import create_controller
        from generals_arena.league.config import LeagueConfig
        from generals_arena.league.game import RecordedGame
        fixture=json.loads(Path(__file__).with_name('eight_player.json').read_text())
        game=RecordedGame(fixture['plan'],LeagueConfig(max_ticks=fixture['ticks']),
                          [create_controller(kind) for kind in fixture['policies']])
        start=time.perf_counter()
        while not game.done:game.advance()
        elapsed=time.perf_counter()-start
        digest=hashlib.sha256(gzip.decompress(game.replay_blob())).hexdigest()
        assert digest==fixture['trace_sha256'], 'actions or state digests diverged from the reference trace'
        print(json.dumps({'reference':args.reference or 'working tree','seconds':elapsed,
                          'ticks':game.state.tick,'ticks_per_second':game.state.tick/elapsed,
                          'trace_sha256':digest,'identical':True},indent=2))


if __name__=='__main__':run()
