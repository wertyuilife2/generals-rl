"""Full-state fingerprints from the Python map generator at ddad4ac.

Hash every protocol field for seeds 0..255, including FFA and zero densities.
This catches changed random draw order, city fairness and map rejection rules.
"""
import hashlib

import pytest

from generals import MapConfig
from generals.core.batch import pack_states
from generals.core.mapgen import generate_map


@pytest.mark.parametrize('config,expected', [
    (MapConfig(9,9,2,.5,.5), '0013aefb005586f53d0dc9a995dc201a83df44d7c9c28fee86cf54f15f5e5e5b'),
    (MapConfig(None,None,2,.5,.5), 'f60afd1f897f3dbe3f084433fdcd7ac292d93ad364a1b5d6047e98ae0cefdea1'),
    (MapConfig(18,23,2,.25,.75), '84fb18c097ed983f16f466c78f0a0f602699fd1a8c40e3e2ad0c32f9b918d382'),
    (MapConfig(20,20,2,0,0), 'bbbc1b8793fb1abc1ec9d3bd5958e5fed7cc2e698cf8d38eab91cca6058a4098'),
    (MapConfig(25,25,4,.5,.5), '10d09ee94f9e9ff09d3b845a44fbfe13264608b5e8512b5868ceee1c7cefc71a'),
    (MapConfig(None,None,8,.75,.25), '3f0023be420404c0ecc71ab38a67ccf8b7263499275ef775406a01a5db2472e7'),
])
def test_native_generation_preserves_python_seed_distribution(config, expected):
    digest = hashlib.sha256()
    for seed in range(256):
        arrays, _, _ = pack_states([generate_map(config, seed)])
        for value in arrays.values():
            digest.update(value.tobytes())
    assert digest.hexdigest() == expected
