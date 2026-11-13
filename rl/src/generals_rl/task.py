"""Algorithm-neutral MDP composition over the existing CPU/CUDA game backend."""
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from generals import MapConfig, VecEnv

from .mdp import ActionMetrics, GameBatch, RewardConfig, ShapedReward, make_observation_encoder
from .policy.actions import make_action_space
from .policy.knowledge import KnowledgeState
from .policy.features import feature_recipe
from .policy.observations import ObservationHistory
from .mdp.observations import encode_privileged
from .opponents import make_opponent
from .map_curriculum import MapCurriculumConfig


@dataclass(frozen=True)
class TaskConfig:
    width: int = 23
    height: int = 23
    players: int = 2
    num_envs: int = 64
    backend: str = "cpu"
    device: str = "cpu"
    max_episode_steps: int = 2048
    clock_scale_ticks: int | None = None
    knowledge_age_scale: int = 2000
    time_limit_mode: str = "bootstrap"
    history_length: int = 4
    feature_set: str = "stacked"
    seed: int = 42
    threads: int = 1
    learner_seat: int = 0
    rotate_seats: bool = True
    opponent: str = "random"
    train_both_sides: bool = False
    mountain_density: float = 0.5
    city_density: float = 0.5
    combat_features: bool = False
    king_threat_distance: bool = False
    critic_potential: bool = False
    persistent_knowledge: bool = False
    critic_privileged: bool = True
    map_pool_size: int = 256
    map_cache_dir: str | None = 'auto'
    map_generation_workers: int = 2
    map_refresh_reuse: int = 0
    map_refresh_chunk_size: int = 1024
    map_prefetch_chunks: int = 8
    random_maps: bool = True
    map_curriculum: MapCurriculumConfig | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if self.clock_scale_ticks is not None and (type(self.clock_scale_ticks) is not int or self.clock_scale_ticks < 1):
            raise ValueError('clock_scale_ticks must be a positive integer or None (legacy episode clock)')
        if type(self.knowledge_age_scale) is not int or self.knowledge_age_scale < 1:
            raise ValueError('knowledge_age_scale must be a positive integer')
        if self.time_limit_mode not in ('bootstrap', 'terminal_draw'):
            raise ValueError('time_limit_mode must be bootstrap or terminal_draw')
        recipe = feature_recipe(self.feature_set)
        if self.feature_set != "stacked" and (self.history_length != recipe.history_length or not self.persistent_knowledge or not self.train_both_sides):
            raise ValueError("compact training requires matching history, persistent memory and dual current self-play")
        MapConfig(self.width, self.height, self.players, self.mountain_density, self.city_density)
        if self.players != 2:
            raise ValueError("RL training currently supports 1v1 only")
        if self.random_maps and (self.width < 23 or self.height < 23):
            raise ValueError("official random maps require a canvas of at least 23x23")
        if self.num_envs < 1 or self.max_episode_steps < 1 or self.history_length < 1:
            raise ValueError("num_envs, episode steps, and history must be positive")
        if self.backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        if self.backend != "cpu" and torch.device(self.device).type != "cuda":
            raise ValueError("GPU simulation requires a CUDA task device")
        if not 0 <= self.learner_seat < self.players:
            raise ValueError("learner seat is out of range")
        if self.opponent == "current_self_play" and self.players != 2:
            raise ValueError("current self-play requires two players")
        if not isinstance(self.train_both_sides, bool):
            raise TypeError("train_both_sides must be boolean")
        if self.train_both_sides and self.opponent != "current_self_play":
            raise ValueError("train_both_sides requires current_self_play")
        if not isinstance(self.combat_features, bool):
            raise TypeError("combat_features must be boolean")
        if self.critic_potential is not False:
            raise ValueError("analytic critic_potential inputs are retired; potential rewards remain supported")
        if not isinstance(self.persistent_knowledge, bool):
            raise TypeError("persistent_knowledge must be boolean")
        if not isinstance(self.critic_privileged, bool):
            raise TypeError("critic_privileged must be boolean")
        if self.combat_features and self.players != 2:
            raise ValueError("combat features require two players")
        if not isinstance(self.king_threat_distance, bool):
            raise TypeError("king_threat_distance must be boolean")
        if self.king_threat_distance and self.players != 2:
            raise ValueError("king threat distance requires two players")
        if isinstance(self.map_pool_size, bool) or not isinstance(self.map_pool_size, int) or self.map_pool_size < 0:
            raise ValueError("map_pool_size must be a nonnegative integer")
        if self.map_cache_dir is not None and (not isinstance(self.map_cache_dir, str) or not self.map_cache_dir):
            raise ValueError('map_cache_dir must be a path, auto, or None')
        for name in ('map_generation_workers', 'map_refresh_chunk_size', 'map_prefetch_chunks'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.map_refresh_reuse) is not int or self.map_refresh_reuse < 0 or (self.map_refresh_reuse and not self.map_pool_size):
            raise ValueError('map_refresh_reuse must be nonnegative and requires a map pool')
        if self.map_curriculum is not None and (not isinstance(self.map_curriculum, MapCurriculumConfig)
                                               or not self.map_pool_size):
            raise ValueError('map_curriculum requires MapCurriculumConfig and a positive map_pool_size')


@dataclass(frozen=True)
class TaskStep:
    observations: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    terminal_observations: dict[str, torch.Tensor] | None
    metrics: dict[str, torch.Tensor]


class GeneralsTask:
    """Physical games with one or both player trajectories exposed to PPO.

    All decisions precede ordered combat and subsequent growth. Reset is explicitly
    owned here so terminal rewards and next observations precede reset.
    """

    def __init__(self, config: TaskConfig, *, backend=None, action_space=None,
                 observation_encoder=None, reward_model=None, opponent=None):
        self.config = config
        self.device = torch.device(config.device)
        self.num_games = config.num_envs
        self.train_both_sides = config.train_both_sides
        self.num_envs = self.num_games * (2 if self.train_both_sides else 1)
        self._batch_cache = None
        self._rows = torch.arange(self.num_envs, device=self.device)
        self._game_rows = self._rows // 2 if self.train_both_sides else self._rows
        sim_device = "cpu" if config.backend == "cpu" else config.device
        options = {"threads": config.threads} if config.backend == "cpu" else {"kernel_backend": "cuda"}
        from .map_sampling import pad_state
        from generals.core.mapgen import generate_map
        map_config = MapConfig(None if config.random_maps else config.width,
                               None if config.random_maps else config.height, config.players,
                               config.mountain_density, config.city_density)
        if backend is None:
            count = 1 if config.map_pool_size else config.num_envs
            seeds = np.random.SeedSequence(config.seed).generate_state(count, dtype=np.uint64)
            initial, valid = zip(*(pad_state(generate_map(map_config, int(seed)), config.width, config.height)
                                   for seed in seeds))
            self._valid = torch.as_tensor(np.stack(valid), device=self.device).expand(config.num_envs, -1)
            self.backend = VecEnv.from_states(
                list(initial) * (config.num_envs if config.map_pool_size else 1), device=sim_device, backend=config.backend,
                seed=config.seed, auto_reset=False, episode_ticks=config.max_episode_steps, **options)
        else:
            self.backend = backend
            self._valid = torch.ones((config.num_envs, config.width * config.height), dtype=torch.bool, device=self.device)
        if (self.backend.batch_size, self.backend.players, self.backend.width, self.backend.height) != (
                config.num_envs, config.players, config.width, config.height):
            raise ValueError("injected backend dimensions do not match TaskConfig")
        if self.backend.unwrapped.auto_reset:
            raise ValueError("task requires simulator auto_reset=False")
        self.map_sampler = None
        if config.map_pool_size:
            from .map_sampling import MapSampler
            self.map_sampler = MapSampler(self.backend, map_config, config.map_pool_size, seed=config.seed,
                curriculum=config.map_curriculum, cache_dir=config.map_cache_dir,
                generation_workers=config.map_generation_workers, refresh_reuse=config.map_refresh_reuse,
                refresh_chunk_size=config.map_refresh_chunk_size, prefetch_chunks=config.map_prefetch_chunks)
            # Choose the first maps before any history/opponent/episode state.
            self.map_sampler.reset()
        self.action_space = action_space or make_action_space(config)
        self.action_metrics = ActionMetrics(self.num_envs, self.device)
        self.observation_encoder = observation_encoder or make_observation_encoder(config)
        self.reward_model = reward_model or ShapedReward(config.reward)
        self.opponent = None if self.train_both_sides else opponent or make_opponent(
            config.opponent, history_length=config.history_length, seed=config.seed,
            persistent_knowledge=config.persistent_knowledge)
        self.history = ObservationHistory(config.history_length)
        self.knowledge = None
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        if self.train_both_sides:
            # Fixed (game0/seat0, game0/seat1, game1/seat0, ...) identity across
            # every rollout and reset. Both seats are already represented.
            self.player_ids = self._rows % 2
        else:
            self.player_ids = torch.full_like(self._rows, config.learner_seat)
            if config.rotate_seats:
                self.player_ids = (self.player_ids + self._rows) % config.players
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = 0
        self.truncated_episodes = 0
        self.terminated_episodes = 0
        if config.persistent_knowledge:
            self.knowledge = KnowledgeState(config.num_envs, config.width, config.height,
                                            config.players, self.device,
                                            age_scale=config.knowledge_age_scale,
                                            remember_dynamics=feature_recipe(config.feature_set).dynamic_memory,
                                            terrain_distances=feature_recipe(config.feature_set).terrain_distances)
            self._update_knowledge(self.batch())
        bind_knowledge = getattr(self.opponent, "bind_knowledge", None)
        if bind_knowledge is not None:
            bind_knowledge(self.knowledge)
        self.history.reset(self.observation_encoder.encode(self.batch(), self.player_ids))
        self._notify_opponent("on_reset")
        self.observations = self._observations()

    def batch(self):
        if self._batch_cache is not None:
            return self._batch_cache
        valid = self.map_sampler.valid if self.map_sampler is not None else self._valid
        batch = GameBatch.from_backend(self.backend, self.device, valid)
        if not self.train_both_sides:
            return batch
        self._batch_cache = GameBatch(
            {key: value.repeat_interleave(2, 0) for key, value in batch.state.items()},
            batch.stats.repeat_interleave(2, 0), batch.width, batch.height,
            valid=batch.valid_cells.repeat_interleave(2, 0))
        return self._batch_cache

    def _update_knowledge(self, batch, mask=None, rows=None):
        players = self.player_ids if rows is None else self.player_ids[rows]
        lanes = (self._game_rows if self.train_both_sides else None) if rows is None else self._game_rows[rows]
        self.knowledge.update(batch, players, mask=mask, lane_ids=lanes)

    def _observations(self, batch=None, rows=None, frames=None):
        batch = self.batch() if batch is None else batch
        if frames is None:
            frames = self.history.frames if rows is None else self.history.frames[rows]
        players = self.player_ids if rows is None else self.player_ids[rows]
        lanes = (self._game_rows if self.train_both_sides else None) if rows is None else self._game_rows[rows]
        observations = {
            # History push/reset replace storage, so these immutable views are
            # valid until PPO has encoded the transition (including timeouts).
            "policy": frames.flatten(1),
            "single_obs": frames[:, -1],
            "action_mask": self.action_space.mask(batch, players),
            "valid_cells": batch.valid_cells.clone(),
        }
        if self.config.critic_privileged:
            observations["critic"] = encode_privileged(self.observation_encoder, batch, players)
        if self.knowledge is not None:
            observations["knowledge"] = self.knowledge.features(
                batch.state["tick"], players, lane_ids=lanes).flatten(1)
        if feature_recipe(self.config.feature_set).half_precision:
            observations["knowledge"] = observations["knowledge"].half().float()
        return observations

    def get_observations(self):
        return self.observations

    def _notify_opponent(self, method, mask=None):
        hook = getattr(self.opponent, method, None)
        if hook is not None:
            arguments = (self.batch(), self.player_ids, self.action_space, self.observation_encoder)
            if method == "on_reset":
                hook(*arguments, mask=mask)
            else:
                hook(*arguments)

    def _backend_mask(self, mask):
        return mask.cpu().numpy() if str(self.backend.device) == "cpu" else mask.to(self.backend.device)

    def reset(self, mask=None, *, rotate=False):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        rows, batch = self._reset_state(mask, rotate=rotate)
        if not rows.numel():
            return self.observations
        self.history.reset(self.observation_encoder.encode(batch, self.player_ids[rows]), rows=rows)
        self._notify_opponent("on_reset", mask)
        selected = self._observations(batch, rows)
        if len(rows) == self.num_envs:
            self.observations = selected
        else:
            # Preserve all retained pre-step/terminal tensors. Policy history
            # already owns new storage; only the other fields need a row merge.
            self.observations = {
                key: value.index_copy(0, rows, selected[key])
                for key, value in self.observations.items() if key not in ('policy', 'single_obs')}
            self.observations.update(policy=self.history.frames.flatten(1), single_obs=self.history.frames[:, -1])
        return self.observations

    def _reset_state(self, mask, *, rotate, rows=None):
        """Reset games and memories without encoding a second full observation."""
        game_mask = mask
        if self.train_both_sides:
            paired = mask.reshape(self.num_games, 2)
            if self.device.type == 'cuda':
                torch._assert_async((paired[:, 0] == paired[:, 1]).all(), 'both seats must reset together')
            elif not torch.equal(paired[:, 0], paired[:, 1]):
                raise ValueError('both seats must reset together')
            game_mask = paired[:, 0]
        rows = mask.nonzero(as_tuple=True)[0] if rows is None else rows
        if not rows.numel():
            return rows, None
        backend_mask = self._backend_mask(game_mask)
        if self.map_sampler is None:
            self.backend.reset(backend_mask)
        else:
            game_rows = rows[::2] // 2 if self.train_both_sides else rows
            if not self.map_sampler.tensor:
                game_rows = game_rows.cpu().numpy()
            self.map_sampler.reset(backend_mask, _rows=game_rows)
        self._batch_cache = None
        if rotate and self.config.rotate_seats and not self.train_both_sides:
            self.player_ids[rows] = (self.player_ids[rows] + 1) % self.config.players
        self.episode_length_buf[mask] = 0
        self.episode_return[mask] = 0
        self.action_metrics.reset(mask)
        if self.knowledge is not None:
            self.knowledge.reset(game_mask)
        batch = self.batch()
        selected = batch if len(rows) == self.num_envs else batch.select(rows)
        if self.knowledge is not None:
            self._update_knowledge(selected, rows=rows)
        return rows, selected

    def step(self, actions, *, capture_all_terminals=True):
        """Advance one tick; training may request only timeout snapshots.

        The default preserves the full pre-reset observation contract. With
        capture_all_terminals=False, terminal_observations is None unless a
        time limit needs bootstrap, and only truncated rows are defined there.
        Rewards and next observations are identical in both modes.
        """
        batch = self.batch()
        learner_actions = self.action_space.validate(actions.to(self.device), self.observations["action_mask"])
        before = batch.learner_stats(self.player_ids)
        if self.train_both_sides:
            joint = learner_actions.reshape(self.num_games, 2)
        elif hasattr(self.opponent, "joint_actions"):
            joint = self.opponent.joint_actions(batch, self.player_ids, self.action_space,
                                                self.observation_encoder)
        else:
            joint = torch.empty((self.num_envs, self.config.players), dtype=torch.long, device=self.device)
            for pid in range(self.config.players):
                seats = torch.full_like(self.player_ids, pid)
                joint[:, pid] = self.opponent.act(batch, seats, self.action_space,
                                                 self.observation_encoder, self.generator)
        if not self.train_both_sides:
            joint[self._rows, self.player_ids] = learner_actions
        submitted = joint.cpu().numpy() if str(self.backend.device) == "cpu" else joint.to(self.backend.device)
        result = self.backend.step(submitted)
        self._batch_cache = None
        # CUDA's errors are reserved status codes, not overflow protection.
        # Retain the invariant check without copying its whole buffer to CPU.
        errors = self.backend.unwrapped.errors
        if isinstance(errors, torch.Tensor) and errors.is_cuda:
            torch._assert_async((errors == 0).all(), 'GPU simulator reported a device error')
        else:
            self.backend.check_errors()
        result_hook = getattr(self.opponent, "on_result", None)
        if result_hook is not None:
            result_hook(result, submitted, self.batch(), self.player_ids)
        batch = self.batch()
        raw_done = torch.as_tensor(result.done, device=self.device)
        if self.train_both_sides:
            raw_done = raw_done.repeat_interleave(2, 0)
        else:
            raw_done = raw_done.clone()
        action_metrics = self.action_metrics.update(
            torch.as_tensor(result.moves, device=self.device)[self._game_rows, self.player_ids])
        alive = batch.state["alive"][self._rows, self.player_ids]
        terminated = raw_done[:, 0] | ~alive
        truncated = raw_done[:, 1] & ~terminated
        done = terminated | truncated
        # Compact once and reuse for metrics and both reset adapters. Repeated
        # boolean gathers each perform a synchronizing CUDA nonzero otherwise.
        ended_rows = done.nonzero(as_tuple=True)[0]
        # Keep simulator end causes for diagnostics. A finite-horizon draw
        # absorbs reward potentials, but never fabricates a winner or defeat.
        reward_terminal = terminated | truncated if self.config.time_limit_mode == 'terminal_draw' else terminated
        reward = self.reward_model.evaluate(before, batch.learner_stats(self.player_ids),
                                           reward_terminal, truncated, batch.state["winner"],
                                           alive, self.player_ids)
        self.episode_length_buf += 1
        self.episode_return += reward.total
        if self.knowledge is not None:
            self._update_knowledge(batch)
        metrics = {f"Reward/{key}": value.detach() for key, value in reward.terms.items()}
        metrics.update({f"Action/{key}": value for key, value in action_metrics.items()})
        if self.knowledge is not None:
            metrics.update({f"Knowledge/{key}": value for key, value in self.knowledge.exploration_metrics(
                self.player_ids, self._game_rows if self.train_both_sides else None).items()})
        metrics.update({"Episode/return": self.episode_return[ended_rows],
                        "Episode/length": self.episode_length_buf[ended_rows].float(),
                        "Episode/win": (batch.state["winner"][ended_rows] == self.player_ids[ended_rows]).float(),
                        "Episode/truncated": truncated[ended_rows].float()})
        stride = 2 if self.train_both_sides else 1
        completed = len(ended_rows) // stride
        timeouts = int(truncated[ended_rows[::stride]].sum()) if completed else 0
        self.completed_episodes += completed
        self.truncated_episodes += timeouts
        self.terminated_episodes += completed - timeouts
        if capture_all_terminals:
            self.history.push(self.observation_encoder.encode(batch, self.player_ids))
            self._notify_opponent("on_step")
            terminal_obs = self._observations()
            self.observations = terminal_obs
            obs = self.reset(done, rotate=True) if completed else terminal_obs
        else:
            timeout_rows = snapshots = None
            if timeouts and self.config.time_limit_mode == 'bootstrap':
                timeout_rows = truncated.nonzero(as_tuple=True)[0]
                selected = batch.select(timeout_rows)
                current = self.observation_encoder.encode(selected, self.player_ids[timeout_rows])
                frames = torch.cat((self.history.frames[timeout_rows, 1:], current[:, None]), 1)
                snapshots = self._observations(selected, timeout_rows, frames)
            self._notify_opponent("on_step")
            rows = None
            if completed:
                rows, _ = self._reset_state(done, rotate=True, rows=ended_rows)
            self.history.push(self.observation_encoder.encode(self.batch(), self.player_ids), reset_rows=rows)
            if completed:
                self._notify_opponent("on_reset", done)
            self.observations = obs = self._observations()
            # RSL's bridge expects a full batch, but only timeout rows are
            # consumed. Reuse next-state rows elsewhere; never re-encode them.
            terminal_obs = (None if snapshots is None else {
                key: value.index_copy(0, timeout_rows, snapshots[key]) for key, value in obs.items()})
        return TaskStep(obs, reward.total, terminated, truncated, terminal_obs, metrics)

    @property
    def cfg(self):
        return asdict(self.config)

    def close(self):
        self.backend.close()
        if self.map_sampler is not None:
            self.map_sampler.close()
        self.map_sampler = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
