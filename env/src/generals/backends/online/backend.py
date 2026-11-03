"""One official online game exposed at the existing Observation/Action boundary."""
from math import isfinite
import json
from pathlib import Path
from time import monotonic, time_ns
from urllib.parse import quote

from dataclasses import dataclass
from types import MappingProxyType

from .protocol import ObservationDecoder, OnlineProtocolError, attack_arguments
from .account import register_username, validate_username
from .endpoints import resolve_endpoint


@dataclass(frozen=True)
class OnlineFrame:
    frame_id: int
    observations: dict
    done: bool = False
    reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'observations', MappingProxyType(dict(self.observations)))


class StaleFrameError(RuntimeError):
    def __init__(self, frame):
        super().__init__('a newer online frame is available; action was not sent')
        self.frame = frame


class OnlineBackend:
    supports_move_results = False

    def __init__(self, *, user_id, room=None, queue='custom', username=None,
                 server=None, site=None, force_start=False,
                 start_timeout=120.0, update_timeout=15.0,
                 record=None, transport=None):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError('user_id is required; load it from an environment variable')
        if queue not in ('custom', '1v1', 'ffa'):
            raise ValueError('queue must be custom, 1v1, or ffa')
        if queue == 'custom' and (not isinstance(room, str) or not room.strip()):
            raise ValueError('custom queue requires a room')
        if queue != 'custom' and (room is not None or force_start):
            raise ValueError('room and force_start apply only to custom games')
        if not isinstance(force_start, bool):
            raise TypeError('force_start must be boolean')
        self.endpoint = resolve_endpoint(site=site, server=server)
        if username is not None:
            validate_username(username, site=self.endpoint.site)
        for name, value in (('start_timeout', start_timeout), ('update_timeout', update_timeout)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self._user_id, self.username = user_id, username
        self.room, self.queue, self.server = room, queue, self.endpoint.server
        self.site = self.endpoint.site
        self.force_start = bool(force_start)
        self.start_timeout, self.update_timeout = start_timeout, update_timeout
        self.decoder = ObservationDecoder()
        self.transport = transport
        self.record_path = Path(record) if record is not None else None
        self._record = None
        self._started = self._closed = self._joined = self._left = False
        self.frame = None
        self._revision = 0
        self._submitted = -1

    @property
    def replay_url(self):
        if self.decoder.replay_id is None or self.endpoint.website is None:
            return None
        return self.endpoint.website + '/replays/' + quote(self.decoder.replay_id, safe='')

    def _log(self, kind, **data):
        if self._record is not None:
            # Only game packets/actions are recorded, never identity or join events.
            def redact(value):
                if isinstance(value, str):
                    return value.replace(self._user_id, '[REDACTED]')
                if isinstance(value, dict):
                    return {redact(k): redact(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [redact(v) for v in value]
                return value
            line = json.dumps(redact({'kind': kind, 'time_ns': time_ns(), **data}), ensure_ascii=False)
            self._record.write(line + '\n')
            self._record.flush()

    def _publish(self, observation=None, *, done=False, reason=None):
        self._revision += 1
        self.frame = OnlineFrame(self._revision,
            {} if observation is None else {observation.player_id: observation},
            done=done, reason=reason)

    def _leave(self):
        if self._joined and not self._left:
            self._left = True
            self.transport.emit('leave_game' if self.decoder.player_id is not None else 'cancel')

    def _event(self, event):
        name, args = event
        if name in ('disconnect', 'connect_error'):
            raise ConnectionError('online connection ended; automatic replay/reconnection is disabled')
        if name in ('error', 'queue_error', 'gio_error') or name.startswith('error_'):
            message = str(args[0] if args else name).replace(self._user_id, '[REDACTED]')
            raise OnlineProtocolError(f'online server: {message}')
        if name not in ('game_start', 'game_update', 'game_won', 'game_lost'):
            return
        if self.frame is not None and self.frame.done:
            return
        data = args[0] if args else {}
        self._log('receive', event=name, data=data)
        if name == 'game_start':
            self.decoder.start(data)
        elif name == 'game_update':
            observation = self.decoder.update(data)
            if observation is not None:
                self._publish(observation, done=observation.terminated,
                              reason='lost' if observation.terminated else None)
                if observation.terminated:
                    self._leave()
        else:
            self._publish(self.decoder.finish(name == 'game_won'), done=True,
                          reason='won' if name == 'game_won' else 'lost')
            self._leave()

    def _drain(self):
        # Relative diffs must all be applied, even when only the latest frame is used.
        for _ in range(1024):
            if self.frame is not None and self.frame.done:
                return
            event = self.transport.receive(0)
            if event is None:
                return
            self._event(event)
        raise ConnectionError('online stream cannot be caught up; refusing stale actions')

    def _wait(self, after, timeout):
        deadline = monotonic() + timeout
        while self.frame is None or self.frame.frame_id <= after:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError('timed out waiting for an online match frame')
            event = self.transport.receive(remaining)
            if event is not None:
                self._event(event)
        self._drain()
        return self.frame

    def start(self):
        if self._started or self._closed:
            raise RuntimeError('use a new OnlineBackend for each game')
        self._started = True
        try:
            if self.record_path is not None:
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
                self._record = self.record_path.open('x', encoding='utf-8')
                self._log('start', schema=1, backend='online', server=self.server, site=self.site)
            if self.transport is None:
                from .transport import SocketIOTransport
                self.transport = SocketIOTransport()
            self.transport.connect(self.server)
            if self.username is not None:
                register_username(self.transport, self._user_id, self.username,
                                  timeout=self.update_timeout, server=self.server)
            self._joined = True
            event, args = self.endpoint.join_event(self.queue, self.room, self._user_id)
            self.transport.emit(event, *args)
            if self.force_start:
                self.transport.emit('set_force_start', self.room, True)
            return self._wait(-1, self.start_timeout)
        except BaseException:
            self.close()
            raise

    def step(self, actions, *, frame_id):
        if self._closed or self.frame is None:
            raise RuntimeError('start the online backend before stepping')
        if self.frame.done:
            raise RuntimeError('online game already finished')
        self._drain()
        if self.frame.done:
            return self.frame
        if frame_id != self.frame.frame_id:
            raise StaleFrameError(self.frame)
        if frame_id == self._submitted:
            raise RuntimeError('this frame already has a submitted action')
        own = self.decoder.player_id
        if set(actions) != {own}:
            raise ValueError('online backend accepts exactly the controlled seat')
        action = actions[own]
        attack = attack_arguments(action, self.decoder.observation)
        # A decision replaces the pending queue; WAIT also clears older moves.
        # This is ordered delivery, not an acknowledgement of server execution.
        self._submitted = frame_id
        self.transport.emit('clear_moves')
        if attack is not None:
            self.transport.emit('attack', *attack)
        self._log('submit', frame_id=frame_id, turn=self.decoder.turn, action=action.encode())
        return self._wait(frame_id, self.update_timeout)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.transport is not None:
                try:
                    self._leave()
                except Exception:
                    pass  # Still release the socket if it was already disconnected.
                try:
                    self.transport.close()
                except Exception:
                    pass
        finally:
            if self._record is not None:
                self._record.close()
                self._record = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
