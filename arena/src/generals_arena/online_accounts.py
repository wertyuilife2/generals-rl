"""Private, durable Bot credentials in the project's ignored secret directory."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from math import isfinite
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile


def account_path(profile, directory=None):
    if not isinstance(profile, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', profile):
        raise ValueError('account profile must be 1–64 letters, digits, underscores or hyphens')
    root = Path(directory) if directory is not None else Path('config/secret')
    return root / (profile + '.json')


def _check_file(fd):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('account files must be regular, owned by this user, and have mode 0600')


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        _check_file(stream.fileno())
        try:
            value = json.load(stream)
        except (ValueError, UnicodeError):
            raise ValueError('invalid account file; existing credentials were not changed') from None
    if (not isinstance(value, dict) or value.get('schema') != 1 or
            value.get('status') not in ('pending', 'registered') or
            not isinstance(value.get('user_id'), str) or not value['user_id']):
        raise ValueError('unsupported account file; existing credentials were not changed')
    from generals.backends.online.account import validate_server, validate_username
    validate_server(value.get('server', ''))
    validate_username(value.get('username'))
    return value


def load_account(profile, directory=None):
    value = _read(account_path(profile, directory))
    if value['status'] != 'registered':
        raise ValueError('account registration is not confirmed; rerun online-account with the same profile')
    return value


@contextmanager
def _lock(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('account directory must be private (0700) and owned by this user')
    fd = os.open(path.with_suffix('.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as lock:
        _check_file(lock.fileno())
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process is initializing this account') from None
        yield


def _write(path, value, *, initial=False):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if initial:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        fd, temporary = tempfile.mkstemp(prefix='.' + path.stem + '-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    # Persist the directory entry before sending the credential to a server.
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def initialize_account(profile, username, *, directory=None,
                       server='https://botws.generals.io', timeout=15, transport=None):
    from generals.backends.online.account import register_username, validate_server, validate_username
    from generals.backends.online.transport import SocketIOTransport
    username, server = validate_username(username), validate_server(server)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError('registration timeout must be positive and finite')
    path = account_path(profile, directory)
    with _lock(path):
        if path.exists() or path.is_symlink():
            account = _read(path)
            if account['username'] != username or account['server'] != server:
                raise ValueError('existing account uses a different name or server; it was not overwritten')
        else:
            account = {'schema': 1, 'username': username, 'server': server,
                       'user_id': secrets.token_hex(16), 'status': 'pending',
                       'created_at': datetime.now(timezone.utc).isoformat()}
            _write(path, account, initial=True)
        already_registered = account['status'] == 'registered'
        if not already_registered:
            wire = transport if transport is not None else SocketIOTransport(timeout=timeout)
            try:
                wire.connect(server)
                register_username(wire, account['user_id'], username, timeout=timeout)
                account['status'] = 'registered'
                account['registered_at'] = datetime.now(timezone.utc).isoformat()
                _write(path, account)
            finally:
                wire.close()
        return {'ok': True, 'account': profile, 'username': username, 'server': server,
                'status': 'registered', 'already_registered': already_registered,
                'credential_file': str(path.resolve())}
