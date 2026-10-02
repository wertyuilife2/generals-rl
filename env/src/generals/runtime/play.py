"""Desktop session configuration; the original GUI owns ticking and input."""
from generals.protocol import MapConfig


class PlaySession:
    """Start the existing Tk human-play/watch workflow without eager GUI imports.

    run() blocks until the window closes. This mode uses the portable reference
    engine; fast preselected-action batches use VecEnv instead.
    """
    def __init__(self, config: MapConfig, *, seed=42, watch=False, human_seat=0,
                 visibility="local", kinds=None, speed=1.0):
        if not isinstance(config, MapConfig):
            raise TypeError("config must be MapConfig")
        self.config = config
        self.seed = seed
        self.watch = watch
        self.human_seat = human_seat
        self.visibility = visibility
        self.kinds = None if kinds is None else tuple(kinds)
        self.speed = speed
        self.device = "cpu"

    def run(self):
        from generals.gui.app import launch
        return launch(self.config, seed=self.seed, mode="watch" if self.watch else "play",
                      human_seat=self.human_seat, visibility=self.visibility,
                      kinds=self.kinds, speed=self.speed)
