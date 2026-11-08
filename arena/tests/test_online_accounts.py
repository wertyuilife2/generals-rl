"""Account initialization must preserve identity and never expose credentials."""
from collections import deque
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from generals.backends.online.protocol import OnlineProtocolError
from generals.backends.online.endpoints import MAIN_CLIENT_KEY, MAIN_SERVER
from generals_arena.cli import main
from generals.backends.online.credentials import account_path, initialize_account, load_account


class RegistrationWire:
    def __init__(self, reply=''):
        self.events = deque([('error_set_username', (reply,))])
        self.sent, self.closed = [], False

    def connect(self, server):
        self.server = server

    def emit(self, event, *args):
        self.sent.append((event, args))

    def receive(self, timeout=0):
        return self.events.popleft() if self.events else None

    def close(self):
        self.closed = True


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'accounts'
        self.name = '[Bot] GPT6 Razor'

    def tearDown(self):
        self.temp.cleanup()

    def create(self, wire=None, **kwargs):
        return initialize_account('razor', self.name, directory=self.directory,
                                  transport=wire or RegistrationWire(), **kwargs)

    def test_secure_creation_redacted_output_and_idempotent_reuse(self):
        wire = RegistrationWire()
        result = self.create(wire)
        path = account_path('razor', self.directory)
        account = load_account('razor', self.directory)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        self.assertEqual(len(account['user_id']), 32)
        self.assertNotIn(account['user_id'], json.dumps(result))
        self.assertEqual(wire.sent, [('set_username', (account['user_id'], self.name))])
        self.assertTrue(wire.closed)
        again = RegistrationWire()
        second = self.create(again)
        self.assertTrue(second['already_registered'])
        self.assertEqual(again.sent, [])
        self.assertEqual(load_account('razor', self.directory), account)

    def test_failed_registration_keeps_original_credential_for_retry(self):
        wire = RegistrationWire('Name is already taken.')
        with self.assertRaisesRegex(OnlineProtocolError, 'already taken'):
            self.create(wire)
        path = account_path('razor', self.directory)
        pending = json.loads(path.read_text())
        self.assertEqual(pending['status'], 'pending')
        self.assertTrue(wire.closed)
        with self.assertRaisesRegex(ValueError, 'not confirmed'):
            load_account('razor', self.directory)
        self.create()
        self.assertEqual(load_account('razor', self.directory)['user_id'], pending['user_id'])

    def test_existing_identity_cannot_be_replaced_by_a_new_name_or_server(self):
        self.create()
        path = account_path('razor', self.directory)
        original = path.read_bytes()
        for name, server in (('[Bot] Different', 'https://botws.generals.io'),
                             (self.name, 'https://example.com')):
            with self.assertRaisesRegex(ValueError, 'not overwritten'):
                initialize_account('razor', name, directory=self.directory,
                                   server=server, transport=RegistrationWire())
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_names_permissions_and_symlinks_are_rejected(self):
        for profile in ('../escape', '/tmp/account', '', 'with space'):
            with self.assertRaises(ValueError):
                account_path(profile, self.directory)
        with self.assertRaises(ValueError):
            initialize_account('razor', 'human name', directory=self.directory)
        self.create()
        path = account_path('razor', self.directory)
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            load_account('razor', self.directory)
        path.chmod(0o600)
        alias = account_path('alias', self.directory)
        alias.symlink_to(path)
        with self.assertRaises(OSError):
            load_account('alias', self.directory)

    def test_cli_account_command_reports_only_public_metadata(self):
        wire = RegistrationWire()
        output = io.StringIO()
        with patch('generals.backends.online.transport.SocketIOTransport', return_value=wire):
            with redirect_stdout(output):
                code = main(['online-account', '--account', 'razor', '--username', self.name,
                             '--account-dir', str(self.directory)])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())['ok'])
        self.assertNotIn(load_account('razor', self.directory)['user_id'], output.getvalue())
        self.assertEqual([name for name, _ in wire.sent], ['set_username'])

    def test_main_account_is_separate_and_uses_main_registration_protocol(self):
        self.create()
        bot = load_account('razor', self.directory)
        wire = RegistrationWire()
        output = io.StringIO()
        with patch('generals.backends.online.transport.SocketIOTransport', return_value=wire):
            with redirect_stdout(output):
                code = main(['online-account', '--site', 'main', '--account', 'razor-main',
                             '--username', 'GPT6 Razor Bot', '--account-dir', str(self.directory)])
        self.assertEqual(code, 0)
        account = load_account('razor-main', self.directory)
        self.assertEqual(account['server'], MAIN_SERVER)
        self.assertEqual(wire.server, MAIN_SERVER)
        self.assertEqual(wire.sent, [('set_username', (account['user_id'], 'GPT6 Razor Bot', MAIN_CLIENT_KEY, None))])
        self.assertNotEqual(account['user_id'], bot['user_id'])
        self.assertEqual(load_account('razor', self.directory), bot)
        self.assertNotIn(account['user_id'], output.getvalue())
        self.assertEqual(json.loads(output.getvalue())['site'], 'main')

    def test_main_site_cannot_reuse_bot_profile_for_registration_or_play(self):
        self.create()
        original = account_path('razor', self.directory).read_bytes()
        with redirect_stderr(io.StringIO()), patch('generals.backends.online.transport.SocketIOTransport') as transport:
            for command in (
                ['online-account', '--username', 'GPT6 Razor Bot'],
                ['online', 'unused.cpp', '--room', 'test'],
            ):
                with self.subTest(command=command), self.assertRaises(SystemExit) as result:
                    main([*command, '--site', 'main', '--account', 'razor',
                          '--account-dir', str(self.directory)])
                self.assertEqual(result.exception.code, 2)
            transport.assert_not_called()
        self.assertEqual(account_path('razor', self.directory).read_bytes(), original)

    def test_main_registration_server_rejection_preserves_pending_identity(self):
        wire = RegistrationWire()
        wire.events = deque([('gio_error', ('Account Disabled',))])
        with self.assertRaisesRegex(OnlineProtocolError, 'server rejected'):
            initialize_account('razor', 'GPT6 Razor Bot', directory=self.directory,
                               transport=wire, site='main')
        self.assertTrue(wire.closed)
        pending = json.loads(account_path('razor', self.directory).read_text())
        self.assertEqual(pending['status'], 'pending')
        self.assertEqual(pending['server'], MAIN_SERVER)

    def test_main_rejects_reserved_bot_prefix_before_saving_or_connecting(self):
        wire = RegistrationWire()
        with self.assertRaisesRegex(ValueError, 'cannot start with'):
            self.create(wire, site='main')
        self.assertFalse(account_path('razor', self.directory).exists())
        self.assertEqual(wire.sent, [])



if __name__ == '__main__':
    unittest.main()
