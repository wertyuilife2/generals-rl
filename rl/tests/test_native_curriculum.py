"""All five curriculum levels retain their pre-native full-state fingerprints."""
import hashlib

import pytest

from generals import MapConfig
from generals.core.batch import pack_states
from generals_rl.map_curriculum import curriculum_maps, MapCurriculumConfig


@pytest.mark.parametrize('config,reference,expected', [
    (MapConfig(9,9,2,.5,.5), None, '0580da8246adaa051311f4273efeb44e8a2029ab4dd9205a885a6e535770788a'),
    (MapConfig(None,None,2,.5,.5), None, '6017c3804e859a412c9f0ebca3872588a62185872bf106dc5264e239eb654cf9'),
    (MapConfig(None,None,2,.5,.5), 9, '8438da22f51eef1a3182bddc87f41ba63f6b87f2ef96d892bdbc351a1779afaf'),
])
def test_native_curriculum_preserves_all_seeded_layouts(config, reference, expected):
    settings = MapCurriculumConfig(1,2,easy_spawn_reference_size=reference)
    digest = hashlib.sha256()
    for seed in range(256):
        for state in curriculum_maps(config, seed, settings):
            arrays, _, _ = pack_states([state])
            for value in arrays.values():
                digest.update(value.tobytes())
    assert digest.hexdigest() == expected
