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
import statistics
import subprocess
import sys
import tempfile
import time


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',help='baseline Git revision to benchmark')
    parser.add_argument('--reference-arena',help='also load arena from this Git revision')
    parser.add_argument('--fixture',type=Path,default=Path(__file__).with_name('eight_player_35.json'))
    parser.add_argument('--repeat',type=int,default=1,help='repeat identical traces, report median')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='arena-baseline-bench-') as temporary:
        if args.repeat < 1:
            parser.error('--repeat must be positive')
        for prefix, revision in [('baselines/src/', args.reference), ('arena/src/', args.reference_arena)]:
            if not revision:
                continue
            names=subprocess.check_output(['git','ls-tree','-r','--name-only',revision,'--',prefix],cwd=root,text=True).splitlines()
            for name in names:
                if Path(name).suffix not in ('.py', '.cpp', '.h', '.hpp'):continue
                target=Path(temporary)/Path(name).relative_to(prefix)
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=root))
        sys.path.insert(0,temporary)
        from generals_baselines import create_controller
        from generals_arena.competitions.config import LeagueConfig
        from generals_arena.matches.game import RecordedGame
        fixture=json.loads(args.fixture.read_text())
        timings=[]
        for _ in range(args.repeat):
            game=RecordedGame(fixture['plan'],LeagueConfig(max_ticks=fixture['ticks']),
                              [create_controller(kind) for kind in fixture['policies']])
            start=time.perf_counter()
            while not game.done:game.advance()
            timings.append(time.perf_counter()-start)
            digest=hashlib.sha256(gzip.decompress(game.replay_blob())).hexdigest()
            assert digest==fixture['trace_sha256'], 'actions or state digests diverged from the reference trace'
        elapsed=statistics.median(timings)
        print(json.dumps({'reference':args.reference or 'working tree',
                          'arena_reference':args.reference_arena or 'working tree',
                          'fixture':str(args.fixture),'seconds':elapsed,'samples':timings,
                          'ticks':game.state.tick,'ticks_per_second':game.state.tick/elapsed,
                          'trace_sha256':digest,'identical':True},indent=2))


if __name__=='__main__':run()
