"""Pinned training opponents, installed with Arena and frozen beside each run."""
import json
from pathlib import Path
import shutil
import tempfile

from generals_arena.util import process_lock
from .native import _private_cache, build_bundle, bundle_info, source_identity


BASELINE_NAMES = ('aggressive', 'expansion', 'defensive', 'random')
HEURISTIC_NAMES = ('kestrel-v7', 'razor-v8', 'aegis-v2')


def training_opponents(directory):
    """Compile each source/parameter/compiler version once; retain run-local bytes.

    No latest-version alias is resolved during a run. Normal Arena profile
    hashing includes the resulting immutable bundles and native executor.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cache = _private_cache() / 'training-references'
    opponents = [dict(name=name, kind='baseline', policy=name) for name in BASELINE_NAMES]
    for name in HEURISTIC_NAMES:
        source = Path(__file__).with_name('reference_sources') / name
        parameters = json.loads((source / 'params.json').read_text())
        snapshot = source_identity(source / 'policy.cpp', parameters)
        version = snapshot[0]
        target = directory / f'{name}-{version}.zip'
        if not target.exists():
            with process_lock(cache, version + '.lock', blocking=True):
                compiled = cache / (version + '.zip')
                if not compiled.exists():
                    with tempfile.TemporaryDirectory(dir=cache) as temporary:
                        staged = Path(temporary) / 'policy.zip'
                        build_bundle(source / 'policy.cpp', staged, parameters, snapshot=snapshot)
                        staged.replace(compiled)
                # Atomic publication also permits concurrent readers of a run.
                with tempfile.TemporaryDirectory(dir=directory) as temporary:
                    staged = Path(temporary) / 'policy.zip'
                    shutil.copyfile(compiled, staged)
                    staged.replace(target)
        if bundle_info(target)['version'] != version:
            raise ValueError(f'training opponent changed: {name}')
        opponents.append(dict(name=name, kind='cpp', bundle=str(target)))
    return opponents
