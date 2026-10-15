"""One-time username registration; no matchmaking or controller is involved."""
from math import isfinite
from time import monotonic

from .protocol import OnlineProtocolError
from .endpoints import DEFAULT_SERVER, resolve_endpoint, validate_server


def validate_username(username, *, site='bot'):
    if (not isinstance(username, str) or not username.strip() or
            any(ord(c) < 32 or ord(c) == 127 for c in username)):
        raise ValueError('username must be nonempty and contain no control characters')
    if site == 'main':
        if username.startswith('[Bot]'):
            raise ValueError('main-site usernames cannot start with [Bot]; use a name such as GPT6 Razor Bot')
    elif not username.startswith('[Bot]') or not username[5:].strip():
        raise ValueError('a bot username must start with [Bot] followed by a name')
    return username


def register_username(transport, user_id, username, *, timeout=15.0, server=None, site=None):
    """Return only on the official empty-string success response.

    A timeout is ambiguous: retain the same user_id for subsequent recovery.
    Never generate a replacement identity merely because a reply was lost.
    """
    endpoint = resolve_endpoint(site=site, server=server)
    validate_username(username, site=endpoint.site)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not isfinite(timeout) or timeout <= 0:
        raise ValueError('registration timeout must be positive and finite')
    transport.emit('set_username', *endpoint.registration_args(user_id, username))
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
        if name in ('error', 'gio_error') or name.startswith('error_'):
            raise OnlineProtocolError('server rejected username registration')
