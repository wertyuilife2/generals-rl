"""Plot a training run and optionally replay one loss/win per built-in AI.

Example: python rl/tools/analyze_iteration.py outputs/rl_iterations/v1/train
Add --demo to record the selected checkpoint's evaluated games, and --render
to turn the recordings into MP4 files. Selection uses worst-AI win rate first.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


KINDS = ("aggressive", "expansion", "defensive", "random")


def read_json(path):
    return json.loads(Path(path).read_text())


def score(report):
    rates = report["by_ai"]
    return (min(rates[k]["win_rate"] for k in KINDS),
            rates["aggressive"]["win_rate"], report["overall"]["win_rate"])


def read_run(directory):
    """Use reports as the source of outcomes; index/request supply run coordinates."""
    index = directory / "evaluations/index.json"
    jobs = read_json(index).get("jobs", []) if index.exists() else []
    by_path = {Path(job["report"]).resolve(): job for job in jobs if job.get("report")}
    evaluations = []
    for path in sorted((directory / "evaluations").glob("snapshot_*/report.json")):
        report = read_json(path)
        if set(KINDS) - report.get("by_ai", {}).keys():
            continue
        metadata = report.get("training_evaluation") or by_path.get(path.resolve(), {})
        request = path.with_name("request.json")
        if not metadata and request.exists():
            metadata = read_json(request)
        evaluations.append({"report": str(path.resolve()), "data": report,
                            "iteration": metadata.get("iteration"),
                            "steps_per_environment": metadata.get("steps_per_environment"),
                            "snapshot_id": metadata.get("snapshot_id", int(path.parent.name.split("_")[-1]))})
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    scalars = {tag: [(entry.step, entry.value) for entry in events.Scalars(tag)]
               for tag in tags if not tag.endswith("/time")}
    return evaluations, scalars


def checkpoint_for(directory, evaluation):
    candidates = [Path(evaluation["data"]["checkpoint"]),
                  Path(evaluation["report"]).with_name("snapshot.pt"),
                  directory / "evaluations/best.pt"]
    expected = evaluation["data"].get("checkpoint_sha256")
    for candidate in candidates:
        if candidate.exists() and (not expected or sha256(candidate.read_bytes()).hexdigest() == expected):
            return candidate.resolve()
    return None


def plot(scalars, evaluations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for kind in KINDS:
        points = [(e["iteration"], 100 * e["data"]["by_ai"][kind]["win_rate"])
                  for e in evaluations if e["iteration"] is not None]
        if not points:
            points = [(step, 100 * value) for step, value in scalars.get(f"Eval/{kind}/win_rate", [])]
        if points:
            x, y = zip(*sorted(points))
            axes[0, 0].plot(x, y, label=kind)
    axes[0, 0].axhline(80, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Student vs built-in AIs", ylabel="Wins / all games (%)", ylim=(0, 100))
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend()
    panels = [((0, 1), "Loss", ["Loss/value", "Loss/surrogate"]),
              ((0, 2), "Entropy", ["Loss/entropy"]),
              ((1, 0), "Training throughput", ["Perf/total_fps"]),
              ((1, 1), "Learning rate", ["Loss/learning_rate"]),
              ((1, 2), "Teacher/student return", ["Train/mean_teacher_reward", "Train/mean_student_reward"])]
    for (row, column), title, tags in panels:
        axis = axes[row, column]
        axis.set_title(title)
        for tag in tags:
            values = scalars.get(tag, [])
            if not values:
                continue
            data = np.array(values)
            axis.plot(data[:, 0], data[:, 1], alpha=.2, linewidth=.4)
            window = min(100, max(1, len(data) // 20))
            smooth = np.convolve(data[:, 1], np.ones(window) / window, mode="valid")
            axis.plot(data[window - 1:, 0], smooth, label=tag.split("/")[-1])
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    if scalars.get("Loss/learning_rate"):
        axes[1, 1].set_yscale("log")
    axes[1, 0].set_ylabel("Environment transitions / second")
    for axis in axes.flat:
        axis.set_xlabel("Learner iteration")
        axis.grid(alpha=.15)
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def demos(checkpoint, report, output, *, render=False):
    """Reuse evaluation's policy loader, map seeds, battle loop and recording sink."""
    import torch
    from tensordict import TensorDict
    from generals import MapConfig
    from generals.core.mapgen import generate_map
    from generals_rl.config import seed_everything, task_config
    from generals_rl.evaluate_builtin import GameSpec, _load_student, battle_batch
    from generals_rl.task import GeneralsTask
    from recordings import MatchRecorder

    if report.get("stochastic"):
        raise ValueError("demo replay requires a deterministic evaluation report")
    if sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("demo report and checkpoint differ; provide the matching --report")
    config = read_json(checkpoint.with_name("run_config.json"))
    task = task_config(config["task"])
    seed_everything(report["seed"])
    torch.set_num_threads(1)
    chosen = []
    for kind in KINDS:
        for outcome in ("loss", "win"):
            candidates = sorted([game for game in report["games"]
                                 if game["opponent"] == kind and game["outcome"] == outcome],
                                key=lambda game: (game["ticks"], game["game_id"]))
            if candidates:
                chosen.append(candidates[len(candidates) // 2])
    if not chosen:
        return {"cases": [], "reason": "evaluation contains only draws"}
    specs = [GameSpec(**{key: game[key] for key in GameSpec.__dataclass_fields__}) for game in chosen]
    map_config = MapConfig(task.width, task.height, 2, task.mountain_density, task.city_density)
    states = [generate_map(map_config, spec.map_seed) for spec in specs]
    # Obtain feature shapes through the task composition, including future MDP changes.
    with GeneralsTask(replace(task, backend="cpu", device="cpu", num_envs=1,
                              threads=1, opponent="wait")) as example:
        observations = TensorDict(example.get_observations(), batch_size=[1], device="cpu")
        policy = _load_student(checkpoint, config, observations, "cpu")
    names = {spec.game_id: f"{spec.opponent}_{game['outcome']}" for spec, game in zip(specs, chosen)}
    with MatchRecorder(output / "recordings", names=names,
                       metadata={"checkpoint_sha256": report["checkpoint_sha256"],
                                 "evaluation_seed": report["seed"]}) as recorder:
        completed, timing = battle_batch(specs, states, policy, task,
                                          max_ticks=report["max_ticks"], device="cpu",
                                          cpu_threads=1, recorder=recorder)
        cases = recorder.finish()
    for case, expected, actual in zip(cases, chosen, completed, strict=True):
        if (actual["outcome"], actual["ticks"]) != (expected["outcome"], expected["ticks"]):
            raise RuntimeError("demo did not reproduce the evaluated game's outcome and length")
        case.update(opponent=actual["opponent"], outcome=actual["outcome"],
                    learner_seat=actual["learner_seat"], match=actual)
        import gzip
        with gzip.open(case["frames"], "rt") as stream:
            frames = json.load(stream)["frames"][1:]
        seat = actual["learner_seat"]
        waits = sum(frame["actions"][seat][0] == 0 for frame in frames)
        zero_moves = sum(frame["actions"][seat][0] != 0 and frame["moves"][seat][1]
                         and frame["moves"][seat][4] == 0 for frame in frames)
        case["action_diagnostics"] = {"wait_actions": waits, "zero_troop_moves": int(zero_moves),
                                      "zero_troop_move_fraction": zero_moves / len(frames)}
        if render:
            from render_recording import render_recording
            case["video"] = render_recording(case["frames"], output / "videos" / f"{case['name']}.mp4")
    return {"cases": cases, "timing": timing,
            "selection": "one median-duration loss then win per AI when available",
            "examples_are_not_an_unbiased_win_rate_estimate": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="optional matching demo checkpoint")
    parser.add_argument("--report", type=Path, help="optional matching demo evaluation report")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--render", action="store_true", help="also render selected demos as MP4")
    args = parser.parse_args(argv)
    directory = args.run.resolve()
    if not (directory / "run_config.json").exists() and (directory / "train/run_config.json").exists():
        directory /= "train"
    output = (args.output or directory / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluations, scalars = read_run(directory)
    best = max(evaluations, key=lambda entry: score(entry["data"])) if evaluations else None
    checkpoint = args.checkpoint.resolve() if args.checkpoint else (checkpoint_for(directory, best) if best else None)
    summary = {"run": str(directory), "evaluations": len(evaluations),
               "selection_criterion": ["worst_ai_win_rate", "aggressive_win_rate", "overall_win_rate"],
               "best": ({key: value for key, value in best.items() if key != "data"} |
                        {"by_ai": best["data"]["by_ai"], "overall": best["data"]["overall"],
                         "checkpoint": str(checkpoint) if checkpoint else None}) if best else None,
               "latest": evaluations[-1]["data"]["by_ai"] if evaluations else None,
               "scalar_summary": {tag: {"count": len(values), "last": values[-1][1],
                                         "last_100_mean": float(np.mean([value for _, value in values[-100:]]))}
                                  for tag, values in scalars.items() if values}}
    if (directory / "summary.json").exists():
        summary["training"] = read_json(directory / "summary.json")
    plot(scalars, evaluations, output)
    if args.demo or args.render:
        if checkpoint is None:
            raise ValueError("selected weights are no longer retained; provide --checkpoint and --report")
        if args.report is None and best is None:
            raise ValueError("no evaluation report found; provide --report for demo replay")
        report = read_json(args.report) if args.report else best["data"]
        summary["demos"] = demos(checkpoint, report, output, render=args.render)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"curves": str(output / "curves.png"),
                      "summary": str(output / "summary.json"),
                      "best_checkpoint": str(checkpoint) if checkpoint else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
