"""Optional stopping and bounded checkpoint retention outside the RSL runner."""
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainingControlConfig:
    max_seconds: float | None = None
    min_seconds: float = 0.0
    plateau_evaluations: int | None = None
    min_improvement: float = 0.02
    keep_checkpoints: int | None = None

    def __post_init__(self):
        for name in ("max_seconds", "min_seconds", "min_improvement"):
            value = getattr(self, name)
            if value is None and name == "max_seconds":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"training_control.{name} must be finite")
            if value < 0 or (name == "max_seconds" and value == 0):
                raise ValueError(f"training_control.{name} must be positive or nonnegative as appropriate")
        if self.max_seconds is not None and self.min_seconds > self.max_seconds:
            raise ValueError("training_control.min_seconds must not exceed max_seconds")
        for name in ("plateau_evaluations", "keep_checkpoints"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"training_control.{name} must be a positive integer or null")


class TrainingStopped(Exception):
    """Raised only after a complete rollout and optimizer update."""


class TrainingController:
    def __init__(self, settings):
        self.settings = settings
        self.completed_evaluations = 0
        self.evaluations_without_improvement = 0
        self.reference = None

    def on_evaluation(self, point):
        worst, aggressive = point["selection_score"][:2]
        if not all(math.isfinite(value) for value in (worst, aggressive)):
            raise ValueError("evaluation stopping requires finite win rates")
        self.completed_evaluations += 1
        margin = self.settings.min_improvement
        improved = (self.reference is None or worst > self.reference[0] + margin
                    or (worst >= self.reference[0] - margin and aggressive > self.reference[1] + margin))
        if improved:
            self.reference = (worst, aggressive)
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1

    def stop_reason(self, elapsed_seconds, *, evaluation_healthy=True):
        if self.settings.max_seconds is not None and elapsed_seconds >= self.settings.max_seconds:
            return "max_seconds"
        patience = self.settings.plateau_evaluations
        if (patience is not None and evaluation_healthy
                and elapsed_seconds >= self.settings.min_seconds
                and self.completed_evaluations >= patience + 1
                and self.evaluations_without_improvement >= patience):
            return "evaluation_plateau"
        return None


def prune_checkpoints(directory, keep):
    """Keep at most ``keep`` full runner checkpoints, including the final one."""
    if keep is None:
        return
    directory = Path(directory)
    periodic = sorted(directory.glob("model_*.pt"), key=lambda path: int(path.stem.removeprefix("model_")))
    slots = keep - int((directory / "checkpoint.pt").exists())
    for path in periodic[:max(0, len(periodic) - slots)]:
        path.unlink()
