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
from generals_arena.cli import main
from generals_arena.online_accounts import account_path, initialize_account, load_account


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

    def test_play_binds_saved_identity_to_its_registered_server(self):
        self.create()
        with redirect_stderr(io.StringIO()), patch('generals.api.make_env') as factory:
            with self.assertRaises(SystemExit) as result:
                main(['online', 'unused.cpp', '--account', 'razor', '--account-dir', str(self.directory),
                      '--server', 'https://example.com', '--room', 'test'])
            self.assertEqual(result.exception.code, 2)
            factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
