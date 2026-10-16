"""Archive a candidate and exercise it using the unmodified Arena CLI."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from generals_arena.native import source_identity

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--opponent', action='append', default=[])
    parser.add_argument('--games', type=int, default=60)
    parser.add_argument('--seed', type=int, default=261016802)
    parser.add_argument('--size', default='auto')
    args = parser.parse_args()
    source = ROOT / 'v8'
    params = json.loads((source / 'params.json').read_text())
    version, _, files = source_identity(source / 'policy.cpp', params)
    archive = ROOT / 'experiments' / version
    archive.mkdir(exist_ok=True)
    for name, data in files.items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
    (archive / 'params.json').write_text(json.dumps(params, indent=2) + '\n')
    output = PROJECT / 'outputs/analysis/razor-v8-20261016'
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'generals_arena', 'run']
    for opponent in args.opponent or ['@gpt-6-razor-v7']:
        command += ['--player', opponent]
    command += ['--player', str(archive / 'policy.cpp'), '--params', str(archive / 'params.json'),
                '--games', str(args.games), '--seed', str(args.seed), '--size', args.size]
    result = output / (args.label + '-result.json')
    with result.open('w') as out, (output / (args.label + '.log')).open('w') as err:
        completed = subprocess.run(command, cwd=PROJECT, stdout=out, stderr=err)
    if completed.returncode:
        print((output / (args.label + '.log')).read_text()[-4000:])
        raise SystemExit(completed.returncode)
    report = json.loads(result.read_text())
    player = next(p for p in report['players'] if p.get('version') == version)
    summary = dict(label=args.label, version=version, report=report['report'],
                   ok=report['ok'], completed_games=report['completed_games'],
                   metrics=player['by_players'], timing=player['timing'],
                   violations=player['candidate_violations'])
    index = ROOT / 'experiments/index-v8.json'
    with (output / 'index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(index.read_text())
        data['experiments'].append(summary)
        temporary = index.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(index)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
