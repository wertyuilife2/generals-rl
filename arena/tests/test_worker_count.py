import argparse
import pytest

from generals_arena.cli import worker_argument
from generals_arena.competitions import jobs


@pytest.mark.parametrize('cpus,expected', [(1,1),(2,1),(3,1),(4,2),(8,4),(14,8),(16,9),(32,19),(64,32),(None,1)])
def test_auto_respects_available_cpus_and_limits(monkeypatch, cpus, expected):
    monkeypatch.setattr(jobs.os, 'process_cpu_count', lambda: cpus, raising=False)
    monkeypatch.setattr(jobs.os, 'cpu_count', lambda: 128)
    assert jobs.worker_count('auto') == expected


def test_auto_falls_back_on_older_python(monkeypatch):
    monkeypatch.delattr(jobs.os, 'process_cpu_count', raising=False)
    monkeypatch.setattr(jobs.os, 'cpu_count', lambda: 14)
    assert jobs.worker_count('auto') == 8
    assert jobs.worker_count(None) == 8
    assert jobs.worker_count(12) == 12


@pytest.mark.parametrize('value', ['0', '-1', '33', '1.5', 'bogus', 'AUTO'])
def test_worker_cli_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        worker_argument(value)


def test_worker_cli_preserves_auto_until_execution():
    assert worker_argument('auto') == 'auto'
    assert worker_argument('8') == 8


@pytest.mark.parametrize('argv,handler', [
    (['commit', 'policy.cpp'], 'submission_command'),
    (['run', '--player', 'baseline/random', '--player', 'baseline/aggressive'], 'submission_command'),
    (['league', 'run', '/tmp/league'], 'league_command'),
])
@pytest.mark.parametrize('options', [[], ['--workers', 'auto']])
def test_commands_accept_auto(monkeypatch, argv, handler, options):
    from generals_arena import cli
    received = []
    monkeypatch.setattr(cli, handler, lambda args: received.append(args.workers))
    cli.main([*argv, *options])
    assert received == ['auto']
