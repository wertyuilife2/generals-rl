"""Local implementation of the same match boundary used by online play."""
from generals.core.observation import observe
from generals.protocol.actions import Action
from generals.protocol.match import MatchFrame, StaleFrameError


class LocalMatchBackend:
    supports_move_results = True

    def __init__(self, config, *, seed=42, max_ticks=20000):
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
            raise ValueError('max_ticks must be a positive integer')
        self.config, self.seed, self.max_ticks = config, seed, max_ticks
        self.engine = None
        self.frame = None
        self._closed = False

    def _publish(self, results=()):
        state = self.engine.state
        done = state.terminated or state.tick >= self.max_ticks
        self.frame = MatchFrame(state.tick,
            {pid: observe(state, pid) for pid in range(state.player_count) if state.alive[pid]},
            tuple(results), done, ('capture' if state.terminated else 'tick_limit') if done else None)
        return self.frame

    def start(self):
        if self.engine is not None or self._closed:
            raise RuntimeError('use a new LocalMatchBackend for each game')
        from generals import make_engine
        from generals.core.mapgen import generate_map
        self.engine = make_engine(generate_map(self.config, self.seed))
        return self._publish()

    def step(self, actions, *, frame_id):
        if self._closed or self.frame is None or self.frame.done:
            raise RuntimeError('local match is not running')
        if frame_id != self.frame.frame_id:
            raise StaleFrameError(self.frame)
        if set(actions) != set(self.frame.observations):
            raise ValueError('provide one action for every living local seat')
        result = self.engine.step([actions.get(pid, Action.wait())
                                  for pid in range(self.engine.state.player_count)])
        return self._publish(result.moves)

    def close(self):
        if not self._closed:
            self._closed = True
            if self.engine is not None:
                self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
