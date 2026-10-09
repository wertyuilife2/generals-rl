"""Training lifecycle for historical opponents; the vendor runner is unchanged."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import io
from pathlib import Path

from .self_play import HistoricalPolicyPool


@dataclass(frozen=True)
class SelfPlayConfig:
    pool_size: int = 8
    snapshot_interval: int = 50
    stochastic: bool = True
    seed: int = 1000042
    sampling: str = "uniform"

    def __post_init__(self):
        for name, minimum in (("pool_size", 2), ("snapshot_interval", 1), ("seed", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"self_play.{name} must be an integer >= {minimum}")
        if not isinstance(self.stochastic, bool):
            raise TypeError("self_play.stochastic must be boolean")
        if self.sampling != "uniform":
            raise ValueError("only uniform historical sampling is implemented")


class SelfPlaySession:
    def __init__(self, pool, opponent, policy, config):
        self.pool = pool
        self.opponent = opponent
        self.policy = policy
        self.config = config
        self.publications = 0
        self.last_published_iteration = None

    def publish(self, iteration, *, source=None):
        if iteration == self.last_published_iteration:
            return
        snapshot = self.pool.publish(self.policy, iteration=iteration, source=source)
        self.publications += 1
        self.last_published_iteration = iteration
        return snapshot

    def after_update(self, iteration):
        if (iteration + 1) % self.config.snapshot_interval == 0:
            return self.publish(iteration, source={"kind": "learner_update"})

def prepare_self_play(task, runner, config, output_dir, *, resume=None, initial_iteration=-1,
                      initial_source=None, read_only=False, opponent_seed=None):
    """Bind a pool after policy loading, before the first game tick.

    The original policy constructor is used only for feature shapes; snapshot
    inference receives local policy/single_obs/action_mask, optional knowledge,
    and never critic. Actor-information critic inputs are independently chosen
    by obs_groups; enabling memory does not require an information critic.
    Resume copies the old pool and excludes snapshots newer than the selected
    learner checkpoint. Environment and opponent RNG streams start afresh.
    A weights-only warm start supplies initial_source without restoring a pool.
    Read-only ranks open the writer's completed destination after synchronization.
    """
    if task.config.opponent not in ("self_play", "current_self_play"):
        return None
    settings = SelfPlayConfig(**config.get("self_play", {}))
    policy = runner.alg.policy
    policy_options = deepcopy(config["runner"]["policy"])
    policy_options.pop("class_name", None)
    obs_groups = deepcopy(runner.cfg["obs_groups"])
    # One lane is sufficient to reconstruct dimensions, including a placeholder
    # critic shape. Do not retain a complete 10k-environment observation copy.
    example = runner.env.get_observations()[:1].clone()
    example["critic"].zero_()
    policy_type = type(policy)

    def policy_factory():
        with redirect_stdout(io.StringIO()):
            return policy_type(example, deepcopy(obs_groups), task.action_space.size,
                               **deepcopy(policy_options))

    signature = {
        "policy_class": f"{policy_type.__module__}.{policy_type.__qualname__}",
        "policy_config": policy_options,
        "obs_groups": obs_groups,
        "observation_shapes": {key: list(value.shape[1:]) for key, value in example.items()},
        "mdp": {key: getattr(task.config, key) for key in
                ("width", "height", "players", "history_length", "max_episode_steps")},
        "encoder": f"{type(task.observation_encoder).__module__}.{type(task.observation_encoder).__qualname__}",
        "action_space": f"{type(task.action_space).__module__}.{type(task.action_space).__qualname__}",
    }
    if task.config.productive_actions:
        signature["mdp"]["productive_actions"] = True
    if task.config.persistent_knowledge:
        signature["mdp"]["persistent_knowledge"] = True
        signature["knowledge_semantics"] = task.knowledge.semantics
        signature["knowledge_channels"] = list(task.knowledge.channels)
    if hasattr(task.observation_encoder, "semantics"):
        signature["encoder_semantics"] = task.observation_encoder.semantics
    resume_pool = None
    provenance = dict(initial_source) if initial_source is not None else {"kind": "random_initial_policy"}
    if resume and not read_only:
        checkpoint = Path(resume).resolve()
        provenance = {"kind": "learner_checkpoint", "path": str(checkpoint),
                      "sha256": sha256(checkpoint.read_bytes()).hexdigest()}
        candidate = checkpoint.parent / "history_pool"
        if (candidate / "manifest.json").is_file():
            resume_pool = candidate
    pool = HistoricalPolicyPool(
        Path(output_dir) / "history_pool", policy_factory=policy_factory,
        policy_spec=signature, device=task.device, max_size=settings.pool_size,
        resume_from=resume_pool,
        resume_iteration=initial_iteration if resume_pool is not None else None,
        read_only=read_only,
    )
    session = SelfPlaySession(pool, task.opponent, policy, settings)
    # Even when an old pool is restored, make the resumed learner selectable.
    if not read_only:
        session.publish(initial_iteration, source=provenance)
    seed = settings.seed if opponent_seed is None else opponent_seed
    if task.config.opponent == "current_self_play":
        # Snapshots are evaluation/export artifacts only. Every training
        # opponent uses the live learner, which changes between PPO updates.
        task.opponent.attach_policy(policy, stochastic=settings.stochastic, seed=seed)
    else:
        task.opponent.attach(pool, stochastic=settings.stochastic, seed=seed)
    return session
