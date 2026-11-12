"""Deterministic CPU map generation and content-addressed initial-bank cache."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import numpy as np

from generals.core.mapgen import generate_map
from generals.core.state import MapLayout
from generals.protocol import STATE_DTYPES, state_shape


def pad_state(state, width, height):
    """Padding belongs to RL storage, never to the simulator's map protocol."""
    h, w = state.owner.shape
    if h > height or w > width:
        raise ValueError('sampled map exceeds the policy canvas')
    valid = np.zeros((height, width), dtype=bool)
    valid[:h, :w] = True
    padding = ((0, height - h), (0, width - w))
    result = state.copy()
    result.layout = MapLayout(np.pad(state.layout.terrain, padding, constant_values=1))
    for name, fill in (('structure', 0), ('owner', -1), ('army', 0)):
        setattr(result, name, np.pad(getattr(state, name), padding, constant_values=fill))
    for name in ('general_pos', 'spawn_pos'):
        old = getattr(state, name)
        setattr(result, name, np.where(old >= 0, old // w * width + old % w, -1).astype(np.int32))
    return result, valid.ravel()


def _empty(config, size, width, height, curriculum):
    levels = curriculum.levels if curriculum is not None else 1
    total, cells = size * levels, width * height
    arrays = {name: np.empty(state_shape(name, total, cells, config.players), dtype=dtype)
              for name, dtype in STATE_DTYPES.items()}
    return arrays, np.empty((total, cells), dtype=bool)


def _generate_rows(config, seeds, width, height, curriculum):
    arrays, valid = _empty(config, len(seeds), width, height, curriculum)
    for index, seed in enumerate(seeds):
        if curriculum is None:
            maps = (generate_map(config, int(seed)),)
        else:
            from .map_curriculum import curriculum_maps
            maps = curriculum_maps(config, int(seed), curriculum)
        for level, state in enumerate(maps):
            row = level * len(seeds) + index
            # Generated states are still validated, but padding directly into
            # owned bank storage avoids a copied GameState and singleton stacks.
            state.validate()
            h, w = state.owner.shape
            if h > height or w > width:
                raise ValueError('sampled map exceeds the policy canvas')
            valid[row] = False
            valid[row].reshape(height, width)[:h, :w] = True
            for name in arrays:
                if name in ('terrain', 'structure', 'owner', 'army'):
                    target = arrays[name][row].reshape(height, width)
                    if (h, w) != (height, width):
                        target.fill(1 if name == 'terrain' else -1 if name == 'owner' else 0)
                    target[:h, :w] = state.layout.terrain if name == 'terrain' else getattr(state, name)
                elif name in ('general_pos', 'spawn_pos'):
                    cells = getattr(state, name)
                    arrays[name][row] = np.where(cells >= 0, cells//w*width + cells%w, -1)
                elif name == 'winner':
                    arrays[name][row] = -1 if state.winner_id is None else state.winner_id
                else:
                    arrays[name][row] = getattr(state, name)
    return arrays, valid


def build_bank(config, size, seed, *, width, height, curriculum=None, workers=1):
    """Worker count and completion order cannot change map seeds or row order."""
    seeds = np.random.SeedSequence([seed, 0x4D4150]).generate_state(size, dtype=np.uint64)
    if workers == 1:
        arrays, valid = _generate_rows(config, seeds, width, height, curriculum)
    else:
        arrays, valid = _empty(config, size, width, height, curriculum)
        levels = curriculum.levels if curriculum is not None else 1
        # Bound in-flight results, including when an early chunk is slower.
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = deque()
            def receive():
                start, count, future = pending.popleft()
                values, cells = future.result()
                for level in range(levels):
                    dst, src = slice(level*size+start, level*size+start+count), slice(level*count, (level+1)*count)
                    for name in arrays:
                        arrays[name][dst] = values[name][src]
                    valid[dst] = cells[src]
            for start in range(0, size, 256):
                chunk = seeds[start:start+256]
                pending.append((start, len(chunk), pool.submit(_generate_rows, config, chunk, width, height, curriculum)))
                if len(pending) >= 2*workers:
                    receive()
            while pending:
                receive()
    stats = np.empty((len(valid), config.players, 3), dtype=np.int64)
    for pid in range(config.players):
        owned = arrays['owner'] == pid
        stats[:, pid, 0] = np.where(owned, arrays['army'], 0).sum(axis=1)
        stats[:, pid, 1] = owned.sum(axis=1)
        stats[:, pid, 2] = (owned & (arrays['structure'] == 1)).sum(axis=1)
    rng = np.random.SeedSequence([seed, 0x524E47]).generate_state(size, dtype=np.uint32)
    rng[rng == 0] = 1
    return arrays, stats, np.tile(rng, len(valid)//size), valid


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=1)
def source_digest():
    import generals.core.mapgen as mapgen
    root = Path(mapgen.__file__).parent.parent
    sources = sorted((root/'core').glob('*.py')) + sorted((root/'core').glob('*.cpp')) + sorted((root/'protocol').glob('*.py'))
    sources += [Path(__file__), Path(__file__).with_name('map_curriculum.py')]
    return hashlib.sha256(''.join(file_digest(p) for p in sources).encode()).hexdigest()


def cache_identity(config, size, seed, width, height, curriculum):
    settings = None if curriculum is None else asdict(curriculum)
    if settings is not None:
        # These select levels at reset; they never alter generated layouts.
        for key in ('warmup_transitions', 'end_transitions'):
            settings.pop(key)
    return dict(schema=1, config=asdict(config), size=size, seed=seed, width=width, height=height,
                curriculum=settings, source=source_digest(), numpy=np.__version__)


def load_bank(config, size, seed, *, width, height, curriculum=None, workers=1, cache_dir='auto', builder=build_bank):
    started = perf_counter()
    options = dict(width=width, height=height, curriculum=curriculum)
    if workers != 1:
        options['workers'] = workers
    if cache_dir is None:
        bank = builder(config, size, seed, **options)
        return bank, dict(cache_hit=False, load_seconds=perf_counter()-started, cache_key=None)
    root = (Path(os.environ.get('GENERALS_RL_MAP_CACHE',
                  str(Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'generals_rl/map_banks')))
            if cache_dir == 'auto' else Path(cache_dir)).expanduser()
    identity = cache_identity(config, size, seed, width, height, curriculum)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = root/key

    def read():
        try:
            meta = json.loads((target/'manifest.json').read_text())
            if meta['identity'] != identity or set(meta['arrays']) != set(STATE_DTYPES) | {'stats', 'rng', 'valid'}:
                return None
            values = {}
            for name, entry in meta['arrays'].items():
                path = target/(name+'.npy')
                if file_digest(path) != entry['sha256']:
                    return None
                value = np.load(path, mmap_mode='c', allow_pickle=False)
                if list(value.shape) != entry['shape'] or value.dtype.str != entry['dtype']:
                    return None
                values[name] = value
            return {n: values[n] for n in STATE_DTYPES}, values['stats'], values['rng'], values['valid']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    root.mkdir(parents=True, exist_ok=True)
    # Lock by content key: simultaneous ranks/jobs build the same bank once.
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bank = read()
        hit = bank is not None
        if not hit:
            print(f'[maps] build {size} maps/level with {workers} CPU worker(s); cache={target}', flush=True)
            bank = builder(config, size, seed, **options)
            arrays, stats, rng, valid = bank
            with tempfile.TemporaryDirectory(prefix='.'+key+'.', dir=root) as temporary:
                staged = Path(temporary)/'bank'
                staged.mkdir()
                entries = {}
                for name, value in dict(arrays, stats=stats, rng=rng, valid=valid).items():
                    path = staged/(name+'.npy')
                    np.save(path, value, allow_pickle=False)
                    entries[name] = dict(shape=list(value.shape), dtype=value.dtype.str, sha256=file_digest(path))
                (staged/'manifest.json').write_text(json.dumps(dict(identity=identity, arrays=entries)))
                if target.exists():
                    shutil.rmtree(target)  # Only this invalid, generated cache.
                staged.rename(target)
    elapsed = perf_counter()-started
    print(f'[maps] {"hit" if hit else "built"} {key[:12]} in {elapsed:.3f}s', flush=True)
    return bank, dict(cache_hit=hit, load_seconds=elapsed, cache_key=key)
