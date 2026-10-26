"""Best-run metrics retain their model identity when replaying an earlier model."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location(
    "analyze_iteration_tool", Path(__file__).resolve().parents[1] / "analyze_iteration.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def test_explicit_demo_does_not_replace_best_model_identity(tmp_path, monkeypatch, capsys):
    run, output = tmp_path / "run", tmp_path / "analysis"
    run.mkdir()
    (run / "run_config.json").write_text("{}")
    evaluations = []
    for updates, elo, seed in ((100, 48., 1001042), (200, 73.25, 271828)):
        directory = run / f"candidate_{updates}"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(f"distinct model weights at update {updates}".encode())
        report = {
            "status": "completed", "policy_id": f"policy_{updates}", "profile_id": "primary",
            "profile": {"seed": seed}, "elo": elo, "elo_ci": None,
            "by_opponent": {"random": {"games": 8}},
        }
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report))
        evaluations.append({"iteration": updates - 1, "completed_ppo_updates": updates,
                            "stage": "periodic", "report": str(report_path), "data": report,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest()})
    early, best = evaluations
    calls = []
    monkeypatch.setattr(analyze, "read_run", lambda directory: (evaluations, {}))
    monkeypatch.setattr(analyze, "plot", lambda *args: None)

    def record_demo(checkpoint, report, directory, *, render):
        # Use actual distinct file bytes, independently of the report metadata.
        assert sha256(checkpoint.read_bytes()).hexdigest() == early["checkpoint_sha256"]
        assert report == early["data"] and directory == output and render is False
        calls.append(checkpoint)
        return {"cases": []}

    monkeypatch.setattr(analyze, "demos", record_demo)
    assert analyze.main([str(run), "--output", str(output), "--demo",
                         "--checkpoint", early["checkpoint"],
                         "--report", early["report"]]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best"]["iteration"] == 199
    assert summary["best"]["elo"] == best["data"]["elo"]
    assert summary["best"]["by_opponent"] == best["data"]["by_opponent"]
    assert summary["best"]["checkpoint"] == best["checkpoint"]
    assert summary["best"]["checkpoint_sha256"] == best["checkpoint_sha256"]
    assert summary["demo_selection"] == {
        "checkpoint": early["checkpoint"],
        "checkpoint_sha256": early["checkpoint_sha256"],
        "report": early["report"], "seed": 1001042,
    }
    assert calls == [Path(early["checkpoint"])]
    console = json.loads(capsys.readouterr().out)
    assert console["best_checkpoint"] == best["checkpoint"]
    assert summary["best"]["checkpoint_sha256"] != summary["demo_selection"]["checkpoint_sha256"]


def test_arena_checkpoint_evaluation_replays_selected_games_exactly(tmp_path, monkeypatch):
    from contextlib import redirect_stdout
    import io
    from generals_rl.export import export_policy
    from generals_arena.competitions.evaluation import make_profile
    from generals_arena.competitions.evaluation import evaluate
    from generals_rl.train import train
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tests"))
    from test_student_ppo import student_config

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    config = student_config()
    config["evaluation"]["enabled"] = False
    # Keep the real 9x9 checkpoint, but force a two-cell battle so a short
    # integration test deterministically exercises terminal replay selection.
    import numpy as np
    from generals.core.state import GameState, MapLayout
    terrain = np.ones((9, 9), dtype=np.uint8)
    terrain[0, :2] = 0
    owner = np.full((9, 9), -1, dtype=np.int16)
    structure = np.zeros((9, 9), dtype=np.uint8)
    army = np.zeros((9, 9), dtype=np.int64)
    owner[0, :2], structure[0, :2], army[0, :2] = [0, 1], 2, [40, 1]
    initial = GameState(MapLayout(terrain), structure, owner, army,
                        np.ones(2, dtype=bool), np.array([0, 1], dtype=np.int32))
    monkeypatch.setattr("generals_arena.matches.batch.generate_map", lambda *args, **kwargs: initial.copy())
    monkeypatch.setattr("generals.core.mapgen.generate_map", lambda *args, **kwargs: initial.copy())
    run = tmp_path / "run"
    with redirect_stdout(io.StringIO()):
        train(config, run, iterations=2)
        checkpoint = run / "checkpoint.pt"
        bundle = export_policy(checkpoint, tmp_path / "bundle")
        profile = make_profile({"width": 9, "height": 9, "players": 2, "mountain_density": 0, "city_density": 0},
                               maps_per_opponent=1, seed=73, max_ticks=8)
        report = evaluate(bundle, profile, tmp_path / "original.json", batch_size=8)
        assert report["status"] == "completed" and report["games"] == 10
        demos = analyze.demos(checkpoint, report, tmp_path / "analysis", render=False)
    assert demos["cases"]
    for case in demos["cases"]:
        assert case["native_core_replay_verified"]
        original = next(game for game in report["results"] if game["game_id"] == case["game_id"])
        assert {key: value for key, value in case["match"].items() if key != "replay"} == original


@pytest.mark.parametrize('native_mode', ['wait', 'timeout'])
def test_demo_reproduction_ignores_native_latency_but_checks_game_evidence(tmp_path, monkeypatch, native_mode):
    import torch
    from generals_arena.competitions.evaluation import evaluate, make_profile
    from generals_arena.matches.replay import verified_replay
    from generals_arena.policies.artifacts import snapshot_policy
    from generals_rl.export import export_policy, prepare_policy_source
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'tests'))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from rl_policy_test_utils import make_checkpoint

    root = Path(__file__).resolve().parents[3]
    run = tmp_path / 'run'
    run.mkdir()
    checkpoint, config = make_checkpoint(run)
    source = prepare_policy_source(config, run)
    saved = torch.load(checkpoint, weights_only=True)
    saved['infos'] = {**source, 'completed_ppo_updates': 3}
    torch.save(saved, checkpoint)
    bundle = export_policy(checkpoint, tmp_path / 'bundle')
    code = (root / 'arena/examples/cpp_wait/policy.cpp').read_text()
    if native_mode == 'timeout':
        code = '#include <thread>\n#include <chrono>\n' + code.replace(
            'return 0;', 'std::this_thread::sleep_for(std::chrono::milliseconds(100)); return 0;')
    native_source = tmp_path / 'policy.cpp'
    native_source.write_text(code)
    opponent = snapshot_policy(tmp_path / 'archive', native_source)
    profile = make_profile({'width': 9, 'height': 9, 'players': 2}, opponents=[opponent],
                           max_ticks=5, maps_per_opponent=1, seed=13)
    report = evaluate(bundle, profile, tmp_path / 'original.json')
    original = json.loads(json.dumps(report))
    output = tmp_path / 'analysis'
    analyze.demos(checkpoint, report, output)
    recorded = json.loads((output / 'recorded_evaluation.json').read_text())
    assert report == original  # Comparing evidence must not rewrite archived diagnostics.
    native_games = [game for game in recorded['results'] if 'decision_timing' in game]
    assert len(native_games) == 2
    assert all(bool(game.get('forfeits')) == (native_mode == 'timeout') for game in native_games)
    for old, new in zip(report['results'], recorded['results'], strict=True):
        assert old['final_digest'] == new['final_digest'] and old['winner_seat'] == new['winner_seat']
        assert len(list(verified_replay(Path(new['replay']).read_bytes(), new))) == new['ticks'] + 1

    report['results'][0]['final_digest'] = '0' * 64
    with pytest.raises(RuntimeError, match='did not reproduce'):
        analyze.demos(checkpoint, report, output)


def test_arena_analysis_follows_confirmed_best_reference_and_preserves_periodic_curve(tmp_path, monkeypatch, capsys):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'run_config.json').write_text('{}')
    jobs = []
    for updates, stage, elo, profile_id in ((14, 'periodic', 100, 'primary'),
                                           (28, 'periodic', 150, 'primary'),
                                           (14, 'confirmation', 200, 'heldout')):
        target = run / 'evaluations' / 'jobs' / f'{updates}_{stage}'
        target.mkdir(parents=True)
        checkpoint = target / 'checkpoint.pt'
        checkpoint.write_bytes(f'full checkpoint {updates}'.encode())
        report = {'status': 'completed', 'policy_id': f'policy_{updates}', 'profile_id': profile_id,
                  'elo': elo, 'elo_ci': {'lower': elo - 20, 'upper': elo + 20},
                  'by_opponent': {'random': {'games': 2}}, 'profile': {'seed': 7}}
        path = target / 'report.json'
        path.write_text(json.dumps(report))
        jobs.append({'completed_ppo_updates': updates, 'iteration': updates - 1,
                     'status': 'completed', 'stage': stage, 'report': str(path),
                     'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
                     'report_sha256': sha256(path.read_bytes()).hexdigest()})
    (run / 'evaluations/index.json').write_text(json.dumps({'jobs': jobs}))
    (run / 'best_model.json').write_text(json.dumps({'policy_id': 'policy_14', 'profile_id': 'heldout'}))
    evaluations, _ = analyze.read_run(run)
    assert len(evaluations) == 3
    selected = analyze.select_best(run, evaluations)
    assert selected['stage'] == 'confirmation' and selected['completed_ppo_updates'] == 14
    monkeypatch.setattr(analyze, 'plot', lambda *args: None)
    assert analyze.main([str(run)]) == 0
    summary = json.loads((run / 'analysis/summary.json').read_text())
    assert summary['selection_criterion'] == 'fixed_profile_elo'
    assert summary['best']['elo'] == 200 and summary['best']['profile_id'] == 'heldout'
    capsys.readouterr()


def test_moving_run_preserves_reports_and_checkpoint_resolution(tmp_path):
    run = tmp_path / 'h200_run'
    directory = run / 'evaluations'
    target = directory / 'jobs' / 'one'
    target.mkdir(parents=True)
    checkpoint, report = target / 'checkpoint.pt', target / 'report.json'
    checkpoint.write_bytes(b'full resume checkpoint')
    report.write_text(json.dumps({'status': 'completed', 'elo': 80.}))
    job = {'status': 'completed', 'iteration': 1, 'completed_ppo_updates': 2, 'stage': 'periodic',
           'report': str(report), 'checkpoint': str(checkpoint),
           'report_sha256': sha256(report.read_bytes()).hexdigest(),
           'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest()}
    (directory / 'index.json').write_text(json.dumps({'directory': str(directory), 'jobs': [job]}))
    moved = tmp_path / 'local_run'
    run.rename(moved)
    evaluations, _ = analyze.read_run(moved)
    assert len(evaluations) == 1 and evaluations[0]['data']['elo'] == 80.
    assert analyze.checkpoint_for(evaluations[0]) == moved / checkpoint.relative_to(run)
    Path(evaluations[0]['report']).unlink()
    with pytest.raises(FileNotFoundError):
        analyze.read_run(moved)
