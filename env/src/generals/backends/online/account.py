"""One-time username registration; no matchmaking or controller is involved."""
from math import isfinite
from time import monotonic
from urllib.parse import urlsplit

from .protocol import OnlineProtocolError


DEFAULT_SERVER = 'https://botws.generals.io'


def validate_server(server):
    address = urlsplit(server)
    if (address.scheme not in ('https', 'http') or not address.hostname or address.username or
            address.password or address.query or address.fragment or address.path not in ('', '/')):
        raise ValueError('server must be an HTTP(S) origin without credentials or query parameters')
    if address.scheme != 'https' and address.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('remote online servers require HTTPS')
    return server.rstrip('/')


def validate_username(username):
    if (not isinstance(username, str) or not username.startswith('[Bot]') or
            not username[5:].strip() or any(ord(c) < 32 or ord(c) == 127 for c in username)):
        raise ValueError('a bot username must start with [Bot] followed by a name')
    return username


def register_username(transport, user_id, username, *, timeout=15.0):
    """Return only on the official empty-string success response.

    A timeout is ambiguous: retain the same user_id for subsequent recovery.
    Never generate a replacement identity merely because a reply was lost.
    """
    validate_username(username)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError('registration timeout must be positive and finite')
    transport.emit('set_username', user_id, username)
    deadline = monotonic() + timeout
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('registration reply timed out; keep the saved account credential')
        event = transport.receive(remaining)
        if event is None:
            continue
        name, args = event
        if name in ('disconnect', 'connect_error'):
            raise ConnectionError('connection ended before username registration was confirmed')
        if name == 'error_set_username':
            message = args[0] if args else None
            if message == '':
                return
            if not isinstance(message, str):
                raise OnlineProtocolError('invalid username registration response')
            raise OnlineProtocolError('username registration failed: ' + message.replace(user_id, '[REDACTED]'))
        if name == 'error' or name.startswith('error_'):
            raise OnlineProtocolError('server rejected username registration')
