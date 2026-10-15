"""Official sites and their wire-level join/registration differences."""
from dataclasses import dataclass
from urllib.parse import urlsplit


DEFAULT_SERVER = 'https://botws.generals.io'
MAIN_SERVER = 'https://ws.generals.io'
# Public client protocol field, not an account credential. Verified against:
# https://generals.io/generals-main-prod-v31.4.3-485dbf55.js (2026-10-15).
MAIN_CLIENT_KEY = 'sd09fjdZ03i0ejwi_changeme'


def validate_server(server):
    address = urlsplit(server)
    if (address.scheme not in ('https', 'http') or not address.hostname or address.username or
            address.password or address.query or address.fragment or address.path not in ('', '/')):
        raise ValueError('server must be an HTTP(S) origin without credentials or query parameters')
    if address.scheme != 'https' and address.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('remote online servers require HTTPS')
    return server.rstrip('/')


@dataclass(frozen=True)
class OnlineEndpoint:
    site: str | None
    server: str
    website: str | None

    def registration_args(self, user_id, username):
        if self.site == 'main':
            return user_id, username, MAIN_CLIENT_KEY, None
        return user_id, username

    def join_event(self, queue, room, user_id):
        if queue == 'custom':
            args = (room, user_id)
            if self.site == 'main':
                args += (MAIN_CLIENT_KEY, None)
            return 'join_private', args
        args = (user_id,)
        if self.site == 'main':
            args += (MAIN_CLIENT_KEY, None, None)
            if queue == '1v1':
                # Match the official client's default queue_1v1_allow_bots.
                args += (True,)
        return ('join_1v1' if queue == '1v1' else 'play'), args


SITES = {
    'bot': OnlineEndpoint('bot', DEFAULT_SERVER, 'https://bot.generals.io'),
    'main': OnlineEndpoint('main', MAIN_SERVER, 'https://generals.io'),
}


def resolve_endpoint(*, site=None, server=None):
    if site is not None:
        if site not in SITES:
            raise ValueError('site must be bot or main')
        if server is not None:
            raise ValueError('choose site or server, not both')
        return SITES[site]
    origin = validate_server(DEFAULT_SERVER if server is None else server)
    for endpoint in SITES.values():
        if origin == endpoint.server:
            return endpoint
    # A custom server's replay website cannot be inferred from its socket URL.
    return OnlineEndpoint(None, origin, None)
