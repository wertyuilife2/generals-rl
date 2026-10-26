"""File sinks for any schedule mode; the persistent league uses a SQLite sink."""
from pathlib import Path

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
