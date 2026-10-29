"""Online orchestration accepts ordinary opaque packages and records failures."""
import json
from dataclasses import replace

import pytest

from generals import Action, MapConfig
from generals.core.mapgen import generate_map
from generals.core.observation import observe
from generals.backends.online.backend import OnlineFrame
from generals_arena.matches.online import play_online
from policy_test_utils import make_bundle


class Server:
    replay_url = 'https://example.test/replay'

    def __init__(self, **options):
        self.obs = observe(generate_map(MapConfig(25, 25, 2), 42), 0)
        self.closed = False
        self.options = options

    def start(self):
        return OnlineFrame(1, {0: self.obs})

    def step(self, actions, *, frame_id):
        assert frame_id == 1 and isinstance(actions[0], Action)
        return OnlineFrame(2, {0: replace(self.obs, terminated=True, winner_id=0)}, True, 'won')

    def close(self):
        self.closed = True


def test_online_reuses_opaque_package_and_persists_each_game(tmp_path):
    bundle = make_bundle(tmp_path/'policy')
    output = tmp_path/'series'
    servers = []
    def backend(**options):
        server = Server(**options)
        servers.append(server)
        return server
    report = play_online({'kind': 'package', 'name': 'opaque', 'bundle': str(bundle)}, {},
                         games=2, output=output, backend_factory=backend)
    assert report['wins'] == report['completed_games'] == 2
    assert all(s.closed for s in servers)
    assert [g['seed'] for g in report['games']] == [42, 43]
    assert json.loads((output/'report.json').read_text()) == report


def test_online_rejects_unsupported_package_shape_and_keeps_failed_attempt(tmp_path):
    bundle = make_bundle(tmp_path/'policy', width=9, height=9)
    server = Server()
    output = tmp_path/'series'
    with pytest.raises(ValueError, match='capabilities'):
        play_online({'kind': 'package', 'name': 'opaque', 'bundle': str(bundle)}, {},
                    output=output, backend_factory=lambda **options: server)
    assert server.closed
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'failed' and report['completed_games'] == 0
    assert report['games'][0]['error'] == 'ValueError'
