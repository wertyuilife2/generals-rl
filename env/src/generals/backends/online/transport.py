"""Optional Socket.IO transport. Background I/O never calls a controller."""
from queue import Empty, Full, Queue
from threading import Event


class SocketIOTransport:
    def __init__(self, *, timeout=10.0, queue_size=256):
        try:
            import socketio
            import certifi
            import ssl
            from importlib.metadata import version
        except ImportError as exc:
            raise ImportError('online mode requires: pip install "generals-env[online]"') from exc
        if version('python-socketio').split('.')[0] != '5':
            raise ImportError('online mode requires python-socketio 5.x (Engine.IO 4)')
        self.events = Queue(maxsize=queue_size)
        self.overflow = Event()
        # Some Python.org macOS installations lack a default CA bundle. Add the
        # normal Requests trust roots while retaining system/custom trust and
        # mandatory certificate/hostname verification.
        context = ssl.create_default_context()
        context.load_verify_locations(certifi.where())
        self.client = socketio.Client(reconnection=False, logger=False, engineio_logger=False,
            request_timeout=timeout, handle_sigint=False,
            websocket_extra_options={'sslopt': {'context': context}})
        self.timeout = timeout
        for name in ('game_start', 'game_update', 'game_won', 'game_lost',
                     'error_set_username', 'error', 'queue_error', 'connect_error', 'disconnect'):
            self.client.on(name, self._handler(name))
        self.client.on('*', self._other_event)

    def _other_event(self, name, *args):
        if name.startswith('error_'):
            self._handler(name)(*args)

    def _handler(self, name):
        def receive(*args):
            try:
                self.events.put_nowait((name, args))
            except Full:
                # Dropping a map diff corrupts all later observations. Fail closed.
                self.overflow.set()
        return receive

    def connect(self, server):
        try:
            self.client.connect(server, transports=['websocket'], wait_timeout=self.timeout)
        except Exception as exc:
            raise ConnectionError(f'could not establish the online Socket.IO connection to {server}') from exc

    def emit(self, name, *args):
        if not self.client.connected:
            raise ConnectionError('online connection is closed')
        try:
            self.client.emit(name, tuple(args) if args else None)
        except Exception as exc:
            raise ConnectionError('online event could not be sent; it will not be retried') from exc

    def receive(self, timeout=0):
        if self.overflow.is_set():
            raise ConnectionError('online event queue overflow; reconnect in a new match')
        try:
            return self.events.get(timeout=timeout) if timeout else self.events.get_nowait()
        except Empty:
            return None

    def close(self):
        self.client.disconnect()
