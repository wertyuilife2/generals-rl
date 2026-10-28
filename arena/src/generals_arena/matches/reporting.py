"""File sinks for any schedule mode; the persistent league uses a SQLite sink."""
from pathlib import Path
import csv

from generals_arena.util import write_json


def output_path(output):
    if output is None:
        return None
    path = Path(output)
    return path if path.suffix == '.json' else path / 'report.json'


def replay_writer(output, *, enabled):
    if not enabled:
        return None
    if output is None:
        raise ValueError('recording requires an output location')
    directory = output.parent / 'replays'

    def save(result, blob):
        if blob is None:
            raise ValueError('the match executor did not record a replay')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (result['game_id'] + '.json.gz')
        path.write_bytes(blob)
        result['replay'] = str(path)
    return save


def write_report(output, report):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        if report.get('win_matrix'):
            rows = report['ranking']
            with output.with_name(output.stem + '.win_matrix.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['wins/draws/losses', *(row['name'] for row in rows)])
                for row in rows:
                    values = report['win_matrix'].get(row['entrant_id'], {})
                    writer.writerow([row['name'], *(f"{cell['wins']}/{cell['draws']}/{cell['losses']}" if cell else ''
                        for cell in (values.get(other['entrant_id']) for other in rows))])
