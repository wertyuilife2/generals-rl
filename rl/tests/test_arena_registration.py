"""Real local model submission, immutable selection and recorded Arena games."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

from generals_arena.competitions import submission as submissions
from generals_arena.competitions.private import run
from generals_arena.policies.artifacts import select_policy
from generals_arena.cli import main
from generals_arena.matches.replay import replay_frames
from generals_arena.competitions.config import LeagueConfig
from generals_arena.competitions.store import CompetitionStore
from rl_policy_test_utils import make_submission


@pytest.fixture
def submission(tmp_path):
    torch.set_num_threads(1)
    bundle, _ = make_submission(tmp_path)
    league = CompetitionStore.create(tmp_path/'league', LeagueConfig(
        player_counts=(2,), min_side=9, max_side=9, side_cap=9, max_ticks=3,
        mountain_min=0, mountain_max=0, city_min=0, city_max=0))
    return bundle, league


def test_local_commit_is_immutable_resumable_and_selectable(submission, monkeypatch):
    bundle, league = submission
    monkeypatch.setattr(submissions, 'PUBLICATION_GAMES', 2)
    report = submissions.commit(bundle,
                             league=league.directory, workers=1)
    assert report['ok'] and report['completed_games'] == 2
    assert report['name'] == 'test-ViT-3'
    assert report['policy_id'] == report['version']
    entry = league.entrants()[report['entrant_id']]
    assert entry['kind'] == 'package'
    assert all(Path(path).is_relative_to(league.directory) for path in entry['artifact_paths'].values())
    profile = report['policy_packages'][entry['entrant_id']]
    assert profile['manifest']['name'] == 'test-ViT-3'
    assert profile['decision_deadline'] is None
    assert submissions.commit(bundle,
                           league=league.directory, workers=1) == report
    with pytest.raises(ValueError, match='alias'):
        submissions.commit(bundle,
                        name='different', league=league.directory)
    with pytest.raises(ValueError, match='deployment'):
        submissions.commit(bundle,
                        league=league.directory, device='cuda:0')
    shutil.rmtree(bundle)
    private = run(['@test-ViT-3', 'baseline/aggressive'],
                           league=league.directory, games=2)
    assert private['ok'] and private['completed_games'] == 2
    by_hash = run(['policy/'+report['policy_id'], 'baseline/aggressive'],
                          league=league.directory, games=2)
    assert by_hash['run_id'] == private['run_id']
    assert league.leaderboard()['completed_games'] == 2
    private_store = CompetitionStore(private['league_directory'])
    for match in private_store.matches()['matches']:
        result, trace = private_store.replay(match['game_id'])
        assert len(replay_frames(trace, result)) == result['ticks']+1
    with pytest.raises(ValueError, match='map size'):
        run(['@test-ViT-3', 'baseline/random'], league=league.directory, size='7x9')
    with pytest.raises(ValueError, match='player count'):
        run(['@test-ViT-3', 'baseline/random', 'baseline/random'], league=league.directory)


def test_incompatible_league_rejected_without_commit_or_job(submission):
    bundle, _ = submission
    league = CompetitionStore.create(bundle.parent/'incompatible', LeagueConfig())
    before = league.jobs()
    with pytest.raises(ValueError, match='support|compatib|map|player'):
        submissions.commit(bundle, league=league.directory)
    assert league.jobs() == before
    assert not (league.directory/'commits').exists()


def test_selectors_share_the_registered_device_without_loading_cuda(submission):
    from generals_arena.policies.artifacts import snapshot_bundle
    bundle, league = submission
    spec = snapshot_bundle(league.directory, bundle, device='cuda:1')
    league.register(spec)
    by_alias = select_policy(league.directory, '@'+spec['name'], entries=league.entrants())
    by_hash = select_policy(league.directory, 'policy/'+spec['version'], entries=league.entrants())
    assert by_alias == by_hash
    assert select_policy(league.directory, bundle, device='cuda:1', entries=league.entrants()) == by_hash
    assert by_hash['device'] == 'cuda:1'
    assert select_policy(league.directory, 'policy/'+spec['version'], device='cpu', entries=league.entrants())['device'] == 'cpu'


def test_private_rl_games_require_a_compatible_league_before_recording(submission):
    from generals_arena.policies.artifacts import snapshot_bundle
    bundle, _ = submission
    league = CompetitionStore.create(bundle.parent/'private-incompatible', LeagueConfig())
    spec = snapshot_bundle(league.directory, bundle)
    with pytest.raises(ValueError, match='support'):
        run(['policy/'+spec['version'], 'baseline/random'],
                     league=league.directory, size='9x9')
    assert not (league.directory/'runs').exists()
    assert league.jobs() == []


def test_cli_complete_bundle_and_explicit_device(submission, monkeypatch, capsys):
    bundle, league = submission
    monkeypatch.setattr(submissions, 'PUBLICATION_GAMES', 1)
    assert main(['commit', str(bundle), '--league', str(league.directory),
                 '--workers', '1', '--device', 'cpu']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['ok'] and report['policy_id'] == report['version']
    assert main(['run', '--player', '@test-ViT-3', '--player', 'baseline/random',
                 '--league', str(league.directory), '--device', 'cpu']) == 0
    assert json.loads(capsys.readouterr().out)['ok']


def test_package_rejects_cpp_parameters(submission):
    bundle, league = submission
    with pytest.raises(ValueError, match='--params'):
        submissions.commit(bundle, league=league.directory, parameters={'x': 1})
    assert not (league.directory/'commits').exists()


def test_registered_matches_run_without_any_training_package(submission):
    bundle, league = submission
    script = bundle.parent/'isolated_arena.py'
    script.write_text(textwrap.dedent('''
        import importlib.abc
        import json
        import sys
        class BlockTraining(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.split('.')[0] in {'generals_rl', 'rsl_rl', 'tensordict'}:
                    raise AssertionError('training dependency: '+fullname)
        sys.meta_path.insert(0, BlockTraining())
        from generals_arena.competitions import submission as submissions
        from generals_arena.competitions.private import run
        submissions.PUBLICATION_GAMES = 1
        bundle, directory = sys.argv[1:]
        report = submissions.commit(bundle,
                                 league=directory, workers=1)
        assert report['ok'], report
        private = run(['@'+report['name'], 'baseline/aggressive'],
                               league=directory, games=2)
        assert private['ok'], private
        assert not any(name.split('.')[0] in {'generals_rl', 'rsl_rl', 'tensordict'}
                       for name in sys.modules)
        print('independent Arena publication and matches passed')
    '''))
    result = subprocess.run([sys.executable, str(script), str(bundle), str(league.directory)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'independent Arena publication and matches passed' in result.stdout


def test_registered_loader_changes_cannot_reuse_deployment_identity(submission, monkeypatch):
    import generals_arena.policies.entries as entrants
    bundle, league = submission
    entry = entrants.prepare_entrant({'name': 'test', 'kind': 'package', 'bundle': bundle})
    entrants.verify_artifacts(entry)
    file_digest = entrants.file_digest
    monkeypatch.setattr(entrants, 'file_digest', lambda path:
                        '0'*64 if Path(path).name == 'code.py' else file_digest(path))
    with pytest.raises(ValueError, match='loader changed'):
        entrants.verify_artifacts(entry)
