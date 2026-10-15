"""Offline protocol/session tests; never register an account or join a real game."""
from collections import deque
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import numpy as np

from generals import Action, Direction, MapConfig, MoveMode, make_env
from generals.backends.online import OnlineBackend, OnlineProtocolError, UnsupportedGameError
from generals.backends.online.protocol import ObservationDecoder, attack_arguments, patch
from generals.backends.online.endpoints import MAIN_CLIENT_KEY, MAIN_SERVER, resolve_endpoint
from generals.protocol.match import StaleFrameError
from generals.runtime.match import run_match


def start_packet(player=0):
    return {'playerIndex': player, 'usernames': ['[Bot] A', '[Bot] B'], 'replay_id': 'abc/123'}


def update_packet(turn=1, *, armies=None):
    armies = armies or [5, 2, 30, 666, 0, 999, 0, 2, 0, 3, 4, 7]
    board = [4, 3, *armies, 0, 0, -1, -4, -2, -3, -1, 1, -1, 1, 1, 1]
    return {'turn': turn, 'map_diff': [0, len(board), *board],
            'cities_diff': [0, 3, 1, 2, 10], 'generals': [0, 11],
            'scores': [{'i': 1, 'total': 16, 'tiles': 4, 'dead': False},
                       {'i': 0, 'total': 7, 'tiles': 2, 'dead': False}]}


class FakeTransport:
    def __init__(self, events=()):
        self.events = deque(events)
        self.sent = []
        self.closed = False
        self.on_emit = lambda *_: None

    def connect(self, server):
        self.server = server

    def receive(self, timeout=0):
        return self.events.popleft() if self.events else None

    def emit(self, event, *args):
        self.sent.append((event, args))
        self.on_emit(event, args)

    def close(self):
        self.closed = True


def session(*, player=0, **options):
    transport = FakeTransport([('game_start', (start_packet(player),)),
                               ('game_update', (update_packet(),))])
    return OnlineBackend(user_id='private-credential', room='test', transport=transport,
                         update_timeout=.005, **options), transport


class DecoderTests(unittest.TestCase):
    def test_diff_runs_growth_shrink_and_malformed_bounds(self):
        self.assertEqual(patch([10, 20, 30, 40], [1, 2, 7, 8, 1]), [10, 7, 8, 40])
        self.assertEqual(patch([10, 20, 30], [1, 0]), [10])
        self.assertEqual(patch([], [0, 2, 10, 20]), [10, 20])
        self.assertEqual(patch([1], []), [])
        for diff in ([2], [-1], [0, 2, 1], [0, -1], [True], [0, 1, 1.5]):
            with self.subTest(diff=diff), self.assertRaises(OnlineProtocolError):
                patch([1], diff)

    def decode(self):
        decoder = ObservationDecoder()
        decoder.start(start_packet())
        return decoder, decoder.update(update_packet())

    def test_observation_exposes_only_fog_obstacle_outline_and_hides_city_totals(self):
        _, obs = self.decode()
        self.assertEqual((obs.width, obs.height, obs.tick, obs.general_pos), (4, 3, 1, 0))
        for cell in (3, 5):
            self.assertFalse(obs.visible.flat[cell])
            self.assertEqual((obs.structure.flat[cell], obs.owner.flat[cell], obs.army.flat[cell]),
                             (255, -3, -1))
        self.assertEqual(obs.terrain.flat[3], 2)
        self.assertEqual(obs.terrain.flat[5], 255)
        self.assertEqual(obs.terrain.flat[4], 1)
        self.assertEqual(obs.owner.flat[4], -1)
        self.assertEqual(obs.owner.flat[2], -2)
        self.assertEqual(obs.structure.flat[0], 2)
        self.assertEqual(obs.structure.flat[10], 1)
        np.testing.assert_array_equal(obs.stats.army, [7, 16])
        np.testing.assert_array_equal(obs.stats.cities, [1, -1])
        self.assertFalse(obs.army.flags.writeable)
        self.assertFalse(obs.stats.army.flags.writeable)

    def test_fog_obstacle_can_become_a_visible_city_instead_of_a_mountain(self):
        decoder, _ = self.decode()
        update = update_packet(2)
        board = update['map_diff'][2:]
        board[2 + 3] = 40
        board[2 + 12 + 3] = -1
        update['map_diff'] = [0, len(board), *board]
        update['cities_diff'] = [0, 4, 1, 2, 3, 10]
        obs = decoder.update(update)
        self.assertTrue(obs.visible.flat[3])
        self.assertEqual((obs.terrain.flat[3], obs.structure.flat[3], obs.owner.flat[3], obs.army.flat[3]),
                         (0, 1, -2, 40))

    def test_incremental_diffs_duplicate_corrections_and_atomic_validation(self):
        decoder, obs = self.decode()
        update = update_packet(4)
        update['map_diff'] = [2, 1, 9, 23]
        update['cities_diff'] = [2, 0]
        newer = decoder.update(update)
        self.assertEqual(newer.army.flat[0], 9)
        self.assertEqual(obs.army.flat[0], 5)
        self.assertEqual(decoder.cities, [1, 2])
        self.assertIsNone(decoder.update(copy.deepcopy(update)))
        corrected = copy.deepcopy(update)
        corrected['map_diff'] = [2, 1, 10, 23]
        self.assertEqual(decoder.update(corrected).army.flat[0], 10)
        with self.assertRaises(OnlineProtocolError):
            decoder.update(update_packet(3))
        bad = update_packet(5)
        bad['scores'][0]['i'] = 0
        saved = decoder.map.copy()
        with self.assertRaises(OnlineProtocolError):
            decoder.update(bad)
        self.assertEqual(decoder.map, saved)

    def test_current_map_tunnel_plane_and_following_incremental_update(self):
        decoder = ObservationDecoder()
        decoder.start(start_packet())
        update = update_packet()
        update['map_diff'][1] += 12
        update['map_diff'].extend([0] * 12)
        obs = decoder.update(update)
        self.assertEqual(obs.general_pos, 0)
        self.assertEqual(obs.owner.flat[11], 1)
        newer = update_packet(2)
        newer['map_diff'] = [2, 1, 6, 35]
        self.assertEqual(decoder.update(newer).army.flat[0], 6)
        newer = update_packet(3)
        newer['map_diff'] = [26, 1, -1, 11]
        ignored = decoder.update(newer)
        self.assertEqual(ignored.army.flat[0], 6)
        self.assertEqual(ignored.owner.flat[11], 1)
        self.assertEqual(decoder.map[26], -1)
        with self.assertRaises(UnsupportedGameError):
            ObservationDecoder().start({**start_packet(), 'options': {'modifiers': ['torus']}})

    def test_unsupported_rules_and_nonlocal_players_are_rejected(self):
        for data in ({**start_packet(), 'teams': [1, 1]},
                     {**start_packet(), 'modifiers': ['torus']},
                     {**start_packet(), 'playerIndex': -100},
                     {**start_packet(), 'usernames': ['x'] * 9}):
            with self.subTest(data=data), self.assertRaises(OnlineProtocolError):
                ObservationDecoder().start(data)
        decoder, _ = self.decode()
        with self.assertRaises(UnsupportedGameError):
            decoder.update({**update_packet(2), 'swamps': [2]})
        bad = update_packet(2)
        bad['map_diff'][2 + 2 + 12] = -5
        with self.assertRaises(UnsupportedGameError):
            decoder.update(bad)

    def test_losing_to_someone_does_not_make_them_the_ffa_winner(self):
        decoder, _ = self.decode()
        lost = decoder.finish(False)
        self.assertTrue(lost.terminated)
        self.assertFalse(lost.alive[0])
        self.assertIsNone(lost.winner_id)

    def test_move_encoding_half_wait_and_no_row_wrap(self):
        _, obs = self.decode()
        self.assertEqual(attack_arguments(Action.move(0, Direction.RIGHT, MoveMode.HALF), obs),
                         (0, 1, True))
        self.assertIsNone(attack_arguments(Action.wait(), obs))
        for action in (Action.move(0, Direction.LEFT), Action.move(0, Direction.DOWN),
                       Action.move(11, Direction.RIGHT), Action.move(7, Direction.LEFT)):
            with self.subTest(action=action), self.assertRaises(ValueError):
                attack_arguments(action, obs)


class OnlineSessionTests(unittest.TestCase):
    def test_main_custom_uses_main_wire_arguments_and_replay_site(self):
        backend, wire = session(site='main', force_start=True)
        with backend:
            backend.start()
            self.assertEqual(wire.server, 'https://ws.generals.io')
            self.assertEqual(wire.sent, [
                ('join_private', ('test', 'private-credential', MAIN_CLIENT_KEY, None)),
                ('set_force_start', ('test', True)),
            ])
            self.assertEqual(backend.replay_url, 'https://generals.io/replays/abc%2F123')
        self.assertEqual(wire.sent[-1], ('leave_game', ()))

    def test_site_public_queues_and_explicit_main_origin_use_matching_protocol(self):
        for selection in ({'site': 'main'}, {'server': MAIN_SERVER + '/'}, {'site': 'bot'}):
            for queue in ('1v1', 'ffa'):
                with self.subTest(selection=selection, queue=queue):
                    wire = FakeTransport([('game_start', (start_packet(),)),
                                          ('game_update', (update_packet(),))])
                    with OnlineBackend(user_id='secret', queue=queue, transport=wire, **selection) as backend:
                        backend.start()
                        args = ('secret',)
                        if backend.site == 'main':
                            args += (MAIN_CLIENT_KEY, None, None)
                            if queue == '1v1':
                                args += (True,)
                        self.assertEqual(wire.sent, [('join_1v1' if queue == '1v1' else 'play', args)])

    def test_main_server_rejection_is_redacted_and_closes_session(self):
        wire = FakeTransport([('gio_error', ('rejected secret',))])
        with self.assertRaisesRegex(OnlineProtocolError, r'rejected \[REDACTED\]'):
            OnlineBackend(user_id='secret', queue='1v1', site='main', transport=wire).start()
        self.assertTrue(wire.closed)
        self.assertEqual([event for event, _ in wire.sent], ['join_1v1', 'cancel'])

    def test_endpoint_selection_rejects_ambiguity_and_does_not_invent_replay_links(self):
        for options in ({'site': 'other'}, {'site': 'main', 'server': MAIN_SERVER}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_endpoint(**options)
        backend, _ = session(server='http://localhost:8000')
        with backend:
            backend.start()
            self.assertIsNone(backend.replay_url)

    def test_actions_replace_queue_and_wait_clears_it_without_fake_receipts(self):
        backend, wire = session()
        with backend:
            first = backend.start()
            wire.on_emit = lambda event, args: wire.events.append(('game_update', (update_packet(2),))) if event == 'attack' else None
            second = backend.step({0: Action.move(0, Direction.RIGHT)}, frame_id=first.frame_id)
            self.assertEqual(wire.sent[-2:], [('clear_moves', ()), ('attack', (0, 1, False))])
            self.assertEqual(second.results, ())
            wire.on_emit = lambda event, args: wire.events.append(('game_won', ({},))) if event == 'clear_moves' else None
            end = backend.step({0: Action.wait()}, frame_id=second.frame_id)
            self.assertTrue(end.done)
            self.assertEqual(end.reason, 'won')
            self.assertEqual(wire.sent[-2:], [('clear_moves', ()), ('leave_game', ())])
            self.assertEqual(backend.replay_url, 'https://bot.generals.io/replays/abc%2F123')
        self.assertTrue(wire.closed)

    def test_backlog_applies_every_diff_and_rejects_stale_decision(self):
        backend, wire = session()
        with backend:
            first = backend.start()
            for turn, army in ((2, 8), (3, 9)):
                update = update_packet(turn)
                update['map_diff'] = [2, 1, army, 23]
                wire.events.append(('game_update', (update,)))
            sent = wire.sent.copy()
            with self.assertRaises(StaleFrameError) as context:
                backend.step({0: Action.wait()}, frame_id=first.frame_id)
            frame = context.exception.frame
            self.assertEqual(frame.observations[0].army.flat[0], 9)
            self.assertEqual(frame.observations[0].tick, 3)
            self.assertEqual(wire.sent, sent)

    def test_foreign_seats_and_invalid_actions_never_send(self):
        backend, wire = session()
        with backend:
            frame = backend.start()
            sent = wire.sent.copy()
            for actions in ({1: Action.wait()}, {0: Action.move(0, Direction.LEFT)}):
                with self.assertRaises(ValueError):
                    backend.step(actions, frame_id=frame.frame_id)
                self.assertEqual(wire.sent, sent)

    def test_disconnect_is_not_retried_and_timeout_does_not_resend(self):
        backend, wire = session()
        with backend:
            frame = backend.start()
            with self.assertRaises(TimeoutError):
                backend.step({0: Action.wait()}, frame_id=frame.frame_id)
            count = len(wire.sent)
            with self.assertRaises(RuntimeError):
                backend.step({0: Action.wait()}, frame_id=frame.frame_id)
            self.assertEqual(len(wire.sent), count)
            wire.events.append(('disconnect', ('transport close',)))
            with self.assertRaises(ConnectionError):
                backend.step({0: Action.wait()}, frame_id=frame.frame_id)
        self.assertTrue(wire.closed)

    def test_username_registration_is_explicit_and_credential_never_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'online.jsonl'
            wire = FakeTransport([('error_set_username', ('',)), ('game_start', (start_packet(),)),
                                  ('game_update', (update_packet(),))])
            secret = 'private-"credential'
            with OnlineBackend(user_id=secret, room='test', username='[Bot] New',
                               transport=wire, record=path) as backend:
                backend.start()
                backend._log('test', echo=secret)
                self.assertEqual(wire.sent[0], ('set_username', (secret, '[Bot] New')))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[-1]['echo'], '[REDACTED]')
            self.assertNotIn('set_username', path.read_text())
            with self.assertRaises(FileExistsError):
                with OnlineBackend(user_id=secret, room='test', transport=FakeTransport(), record=path) as backend:
                    backend.start()

    def test_configuration_rejects_incompatible_modes_and_insecure_remote_origins(self):
        for options in ({}, {'queue': '1v1', 'room': 'a'}, {'room': 'a', 'username': 'human'},
                        {'room': 'a', 'server': 'http://botws.generals.io'},
                        {'room': 'a', 'server': 'https://user:secret@example.com'},
                        {'room': 'a', 'update_timeout': float('inf')}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                OnlineBackend(user_id='test-secret', **options)
        backend = make_env(mode='online', user_id='secret', room='test', transport=FakeTransport())
        self.assertIsInstance(backend, OnlineBackend)
        backend.close()
        with self.assertRaises(ValueError):
            make_env(MapConfig(), mode='online', user_id='secret', room='test')


class WaitController:
    def __init__(self):
        self.resets, self.ticks, self.results, self.closed = [], [], [], False

    def reset(self, player_id, seed):
        self.resets.append((player_id, seed))

    def act(self, observation):
        self.ticks.append(observation.tick)
        return Action.wait()

    def on_result(self, result):
        self.results.append(result)

    def close(self):
        self.closed = True


class MatchRunnerTests(unittest.TestCase):
    def test_same_controller_contract_runs_local_and_online(self):
        controllers = {0: WaitController(), 1: WaitController()}
        result = run_match(make_env(MapConfig(5, 5, 2), mode='match', max_ticks=3), controllers)
        self.assertEqual(result['tick'], 3)
        self.assertEqual(result['reason'], 'tick_limit')
        self.assertEqual(len(controllers[0].results), 3)
        backend, wire = session(player=1)
        wire.on_emit = lambda event, args: wire.events.append(('game_won', ({},))) if event == 'clear_moves' else None
        controller = WaitController()
        result = run_match(backend, controller, seed=42)
        self.assertEqual(result['winner_id'], 1)
        self.assertEqual(controller.resets, [(1, 43)])
        self.assertEqual(controller.results, [])
        self.assertTrue(controller.closed)

    def test_stale_actions_recompute_and_controller_error_closes_connection(self):
        backend, wire = session()
        controller = WaitController()
        original = controller.act
        def act(obs):
            if obs.tick == 1:
                wire.events.append(('game_update', (update_packet(2),)))
            return original(obs)
        controller.act = act
        wire.on_emit = lambda event, args: wire.events.append(('game_won', ({},))) if event == 'clear_moves' else None
        result = run_match(backend, controller)
        self.assertEqual(result['stale_decisions'], 1)
        self.assertEqual(controller.ticks, [1, 2])
        backend, wire = session()
        controller = WaitController()
        controller.act = lambda obs: 'not an action'
        with self.assertRaises(TypeError):
            run_match(backend, controller)
        self.assertTrue(controller.closed)
        self.assertTrue(wire.closed)

    def test_receipt_dependent_controller_is_rejected_before_deciding(self):
        backend, wire = session()
        controller = WaitController()
        controller.requires_move_results = True
        with self.assertRaisesRegex(ValueError, 'authoritative move results'):
            run_match(backend, controller)
        self.assertEqual(controller.ticks, [])
        self.assertTrue(controller.closed)
        self.assertTrue(wire.closed)


@unittest.skipUnless(importlib.util.find_spec('socketio'), 'optional online dependency unavailable')
class TransportTests(unittest.TestCase):
    def make_wire(self, **options):
        from generals.backends.online.transport import SocketIOTransport
        with mock_patch('socketio.Client') as client:
            wire = SocketIOTransport(**options)
            return wire, client.call_args.kwargs

    def test_tls_reconnect_and_positional_event_arguments(self):
        import ssl
        wire, options = self.make_wire()
        self.assertFalse(options['reconnection'])
        self.assertFalse(options['handle_sigint'])
        tls = options['websocket_extra_options']['sslopt']['context']
        self.assertEqual(tls.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(tls.check_hostname)
        wire.connect('https://botws.generals.io')
        wire.client.connect.assert_called_once_with('https://botws.generals.io',
            transports=['websocket'], wait_timeout=10.0)
        wire.emit('attack', 2, 3, False)
        wire.client.emit.assert_called_with('attack', (2, 3, False))
        wire.emit('clear_moves')
        wire.client.emit.assert_called_with('clear_moves', None)

    def test_queue_overflow_fails_instead_of_continuing_after_dropped_diff(self):
        wire, _ = self.make_wire(queue_size=1)
        wire._handler('game_update')({'turn': 1})
        wire._handler('game_update')({'turn': 2})
        with self.assertRaisesRegex(ConnectionError, 'overflow'):
            wire.receive(0)

    def test_new_error_events_and_connect_failures_are_surfaced(self):
        wire, _ = self.make_wire()
        self.assertIn('gio_error', [call.args[0] for call in wire.client.on.call_args_list])
        wire._other_event('error_join', 'bad room')
        self.assertEqual(wire.receive(0), ('error_join', ('bad room',)))
        wire.client.connect.side_effect = RuntimeError('transport failed')
        with self.assertRaises(ConnectionError):
            wire.connect('https://botws.generals.io')


if __name__ == '__main__':
    unittest.main()
