"""CLI and HTTP present independent pools over the same recorded matches."""
from dataclasses import asdict
import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from generals_arena.cli import main
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore
from generals_arena.matches.game import close_game
from generals_arena.matches.game import RecordedGame
from generals_arena.competitions.rating import POLICY
from generals_arena.web.server import make_server
from generals_baselines import create_controller


@pytest.fixture
def league(tmp_path):
    store = CompetitionStore.create(tmp_path / 'league', LeagueConfig(
        player_counts=(2, 3), min_side=25, max_side=25, side_cap=25,
        mountain_min=0, mountain_max=0, city_min=0, city_max=0, max_ticks=1))
    entrant = store.register({'name': 'aggressive', 'kind': 'baseline', 'policy': 'aggressive'})
    random = next(entry['entrant_id'] for entry in store.entrants().values() if entry['name'] == 'random')
    for players in (2, 3):
        plan = {'game_id': f'game-{players}', 'entrant_ids': [entrant['entrant_id'], *[random] * (players - 1)],
                'instance_ids': list(range(players)), 'controller_seeds': list(range(players)),
                'map': asdict(store.config.sample_map(players, players)), 'map_seed': players,
                'human': False, 'rating_policy': POLICY, 'rating_group_size': 1}
        store.insert_plans([plan])
        game = RecordedGame(plan, store.config, [create_controller('aggressive'),
                            *[create_controller('random') for _ in range(players - 1)]])
        try:
            game.advance()
            store.complete(game)
        finally:
            close_game(game)
    return store


def test_cli_can_show_each_pool_or_both_without_an_unified_score(league, capsys):
    assert main(['league', 'leaderboard', str(league.directory), '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {'1v1', 'ffa'}
    assert report['1v1']['completed_games'] == report['ffa']['completed_games'] == 1
    assert report['1v1']['rating_model'] == 'egf-gor'
    assert report['ffa']['rating_model'] == 'codeforces'
    assert main(['league', 'leaderboard', str(league.directory), '--mode', 'ffa', '--json']) == 0
    assert set(json.loads(capsys.readouterr().out)) == {'ffa'}
    assert main(['league', 'leaderboard', str(league.directory)]) == 0
    text = capsys.readouterr().out
    assert '1v1 / egf-gor' in text and 'ffa / codeforces' in text and 'mean 1000' in text
    with pytest.raises(SystemExit) as error:
        main(['league', 'rebuild', str(league.directory), '--current-policy'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --current-policy' in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(['tournament', '--entrants', 'roster.json', '--output', 'out', '--anchor', 'random'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --anchor random' in capsys.readouterr().err


def test_api_and_web_use_explicit_rating_pools_without_changing_game_controls(league):
    server = make_server(league, port=0)
    server.arena.start(run_worker=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'

    def request(path):
        with urlopen(base + path, timeout=5) as response:
            return response.read()

    try:
        meta = json.loads(request('/api/meta'))
        assert meta['rating_modes'] == ['1v1', 'ffa'] and meta['default_rating_mode'] == '1v1'
        duel = json.loads(request('/api/leaderboard?mode=1v1'))
        ffa = json.loads(request('/api/leaderboard?mode=ffa'))
        assert duel['rating_mode'] == '1v1' and ffa['rating_mode'] == 'ffa'
        assert duel['anchor'] == league.reference and ffa['anchor'] is None
        assert json.loads(request('/api/leaderboard')) == duel
        assert duel['completed_games'] == ffa['completed_games'] == 1
        assert sum(row['elo'] for row in ffa['ranking']) / len(ffa['ranking']) == pytest.approx(1000.)
        with pytest.raises(HTTPError) as error:
            request('/api/leaderboard?mode=all')
        assert error.value.code == 400
        html = request('/').decode()
        app = request('/app.js').decode()
        assert 'id="rating-mode"' in html
        assert '/api/leaderboard?mode=${mode}' in app
        assert '1v1 / FFA 分池' in html
        assert 'MoveInput' in app and 'play-players' in html
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.arena.close()
        server.server_close()
