"""C ABI and CLI integration at the online Observation/Action boundary."""
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generals.backends.online import OnlineBackend
from generals.backends.online.protocol import ObservationDecoder
from generals import Action, Direction
from generals.runtime.match import run_match
from generals_arena.cli import main
from generals_arena.native import build_bundle, NativeController


class Wire:
    def __init__(self):
        board = [2, 2, 4, 0, 0, 4, 0, -1, -1, 1]
        self.events = deque([
            ('game_start', ({'playerIndex': 1, 'usernames': ['A', '[Bot] Test'], 'replay_id': 'test'},)),
            ('game_update', ({'turn': 1, 'map_diff': [0, len(board), *board],
                'cities_diff': [], 'generals': [0, 3],
                'scores': [{'i': 0, 'total': 4, 'tiles': 1}, {'i': 1, 'total': 4, 'tiles': 1}]},)),
        ])
        self.sent = []
        self.closed = False

    def connect(self, server):
        self.server = server

    def receive(self, timeout=0):
        return self.events.popleft() if self.events else None

    def emit(self, name, *args):
        self.sent.append((name, args))
        if name == 'clear_moves':
            self.events.append(('game_won', ({},)))

    def close(self):
        self.closed = True


class NativeOnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.temp.name) / 'policy.zip'
        build_bundle(Path(__file__).resolve().parents[1] / 'examples/cpp_wait/policy.cpp', cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_existing_cpp_abi_uses_server_seat_and_native_watchdog(self):
        wire = Wire()
        controller = NativeController(bundle=self.bundle)
        report = run_match(OnlineBackend(user_id='secret', room='test', transport=wire), controller)
        self.assertTrue(report['done'])
        self.assertEqual(report['winner_id'], 1)
        self.assertEqual(controller.player_id, 1)
        self.assertEqual(report['decision_timing'][1]['calls'], 1)
        self.assertEqual(report['decision_timing'][1]['timeouts'], 0)
        self.assertEqual(report['decision_timing'][1]['errors'], 0)
        self.assertEqual(report['decision_timing'][1]['limit_ns'], 10_000_000)
        self.assertIsNone(controller.instance)
        self.assertTrue(wire.closed)

    def test_fog_obstacle_reaches_cpp_without_revealing_hidden_contents(self):
        decoder = ObservationDecoder()
        decoder.start({'playerIndex': 0, 'usernames': ['A', 'B'], 'replay_id': 'fog'})
        board = [5, 2, 8, 0, 777, 888, 999, 0, 0, 0, 0, 0,
                 0, -1, -3, -4, -3, -1, -1, -3, -3, -3]
        obs = decoder.update({'turn': 1, 'map_diff': [0, len(board), *board],
                              'cities_diff': [], 'generals': [0, -1],
                              'scores': [{'i': 0, 'total': 8, 'tiles': 1},
                                         {'i': 1, 'total': 8, 'tiles': 1}]})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fog.cpp'
            source.write_text('''#include "arena_policy.h"
#include <initializer_list>
extern "C" void *arena_create(int, std::uint64_t, const char *) { return new int(0); }
extern "C" void arena_destroy(void *p) { delete static_cast<int *>(p); }
extern "C" std::int64_t arena_act(void *, const ArenaObservation *o) {
    if (o->terrain[3] != 2 || o->terrain[2] != 255) return 0;
    for (int i : {2, 3, 4})
        if (o->visible[i] || o->army[i] != -1 || o->owner[i] != -3 ||
            o->structure[i] != 255) return 0;
    return arena_move(0, 1);
}
''')
            bundle = Path(directory) / 'fog.zip'
            build_bundle(source, bundle)
            controller = NativeController(bundle=bundle)
            try:
                controller.reset(0, 42)
                self.assertEqual(controller.act(obs), Action.move(0, Direction.RIGHT))
                self.assertEqual(controller.decision_metrics['timeouts'], 0)
                self.assertEqual(controller.decision_metrics['errors'], 0)
            finally:
                controller.close()

    def test_online_cli_runs_saved_bundle_without_local_league_or_registration(self):
        wire = Wire()
        output = io.StringIO()
        with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'secret'}):
            with patch('generals.backends.online.transport.SocketIOTransport', return_value=wire):
                with redirect_stdout(output):
                    code = main(['online', str(self.bundle), '--room', 'test',
                                 '--league-directory', str(Path(self.temp.name) / 'nonexistent')])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['reason'], 'won')
        self.assertFalse((Path(self.temp.name) / 'nonexistent').exists())
        self.assertNotIn('set_username', [name for name, _ in wire.sent])

    def test_missing_identity_fails_before_connecting(self):
        with patch.dict('os.environ', {}, clear=True), redirect_stderr(io.StringIO()):
            with patch('generals.backends.online.transport.SocketIOTransport') as transport:
                with self.assertRaises(SystemExit) as result:
                    main(['online', str(self.bundle), '--room', 'test'])
                self.assertEqual(result.exception.code, 2)
                transport.assert_not_called()

    def test_main_cli_uses_main_identity_and_saved_account_infers_site(self):
        for selection in (
            ['--site', 'main'],
            ['--server', 'https://ws.generals.io'],
            ['--account', 'razor-main'],
        ):
            with self.subTest(selection=selection):
                wire = Wire()
                account = {'user_id': 'main-secret', 'server': 'https://ws.generals.io'}
                output = io.StringIO()
                with patch.dict('os.environ', {'GENERALS_MAIN_USER_ID': 'main-secret',
                                                'GENERALS_BOT_USER_ID': 'bot-secret'}, clear=True), \
                        patch('generals_arena.online_accounts.load_account', return_value=account), \
                        patch('generals.backends.online.transport.SocketIOTransport', return_value=wire), \
                        redirect_stdout(output):
                    code = main(['online', str(self.bundle), '--room', 'test', *selection])
                self.assertEqual(code, 0)
                self.assertEqual(wire.server, 'https://ws.generals.io')
                self.assertEqual(wire.sent[0][1][1], 'main-secret')
                self.assertEqual(json.loads(output.getvalue())['replay_url'], 'https://generals.io/replays/test')
                self.assertNotIn('main-secret', output.getvalue())

    def test_main_cli_does_not_fall_back_to_bot_credential(self):
        with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'bot-secret'}, clear=True), \
                redirect_stderr(io.StringIO()), \
                patch('generals.backends.online.transport.SocketIOTransport') as transport:
            with self.assertRaises(SystemExit) as result:
                main(['online', str(self.bundle), '--site', 'main', '--room', 'test'])
            self.assertEqual(result.exception.code, 2)
            transport.assert_not_called()

    def test_main_series_keeps_main_site_metadata_and_replay_links(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'series'
            wire = Wire()
            with patch.dict('os.environ', {'GENERALS_MAIN_USER_ID': 'main-secret'}), \
                    patch('generals.backends.online.transport.SocketIOTransport', return_value=wire), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = main(['online', str(self.bundle), '--site', 'main', '--room', 'test',
                             '--output', str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual((report['site'], report['server']), ('main', 'https://ws.generals.io'))
            self.assertEqual(report['games'][0]['result']['replay_url'], 'https://generals.io/replays/test')
            records = (output / 'game-001.jsonl').read_text()
            self.assertEqual(json.loads(records.splitlines()[0])['site'], 'main')
            self.assertNotIn('main-secret', records)

    def test_series_records_each_completed_game_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'series'
            wires = [Wire(), Wire()]
            with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'secret'}):
                with patch('generals.backends.online.transport.SocketIOTransport', side_effect=wires):
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = main(['online', str(self.bundle), '--room', 'test',
                                     '--games', '2', '--output', str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual((report['completed_games'], report['wins'], report['failures']), (2, 2, 0))
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(list(output.glob('game-*.jsonl'))), 2)
            self.assertTrue(all(wire.closed for wire in wires))
            self.assertNotIn('secret', json.dumps(report))
            with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'secret'}):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(['online', str(self.bundle), '--room', 'test', '--games', '2', '--output', str(output)])

    def test_series_game_protocol_failure_is_saved_and_stops_followup_games(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'series'
            wire = Wire()
            wire.events[1][1][0]['map_diff'] = []
            with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'secret'}):
                with patch('generals.backends.online.transport.SocketIOTransport', return_value=wire) as factory:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = main(['online', str(self.bundle), '--room', 'test', '--games', '2', '--output', str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(factory.call_count, 1)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual((report['completed_games'], report['attempted_games'], report['failures']), (0, 1, 1))
            self.assertEqual(report['games'][0]['replay_id'], 'test')
            self.assertEqual(report['games'][0]['error_type'], 'OnlineProtocolError')
            self.assertTrue(wire.closed)

    def test_series_pause_finishes_current_game_without_starting_another(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'series'
            wire = Wire()
            original = wire.emit

            def request_pause(name, *args):
                if name == 'clear_moves':
                    (output / 'stop-after-game').touch()
                return original(name, *args)

            wire.emit = request_pause
            with patch.dict('os.environ', {'GENERALS_BOT_USER_ID': 'secret'}):
                with patch('generals.backends.online.transport.SocketIOTransport', return_value=wire) as factory:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        code = main(['online', str(self.bundle), '--room', 'test',
                                     '--games', '2', '--output', str(output)])
            self.assertEqual(code, 1)  # The requested series is intentionally incomplete.
            self.assertEqual(factory.call_count, 1)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['status'], 'paused')
            self.assertEqual((report['completed_games'], report['wins'], report['failures']), (1, 1, 0))
            self.assertNotIn('active_game', report)
            self.assertTrue(wire.closed)


if __name__ == '__main__':
    unittest.main()
