"""Plot current Arena evaluation jobs and render their verified selected games."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_json(path):
    return json.loads(Path(path).read_text())


def read_run(directory):
    """Read persisted job coordinates and reports without loading any policy."""
    index = directory / "evaluations/index.json"
    saved = read_json(index) if index.exists() else {}
    recorded = Path(saved.get("directory", index.parent)).resolve()
    by_path = {}
    for job in saved.get("jobs", []):
        if job.get("status") != "completed":
            continue
        job = dict(job)
        for key in ("report", "checkpoint", "bundle"):
            if job.get(key):
                path = Path(job[key])
                relative = path.relative_to(recorded) if path.is_absolute() else path
                job[key] = str((index.parent / relative).resolve())
        by_path[Path(job["report"])] = job
    evaluations = []
    for path, metadata in sorted(by_path.items()):
        report = read_json(path)
        if report.get("status") != "completed" or "elo" not in report:
            raise ValueError("completed job requires a current Arena Elo report; use archived tools for historical reports")
        if metadata.get("report_sha256") and sha256(path.read_bytes()).hexdigest() != metadata["report_sha256"]:
            raise ValueError("evaluation report changed after training accepted it")
        evaluations.append({"report": str(path), "data": report, "iteration": metadata["iteration"],
                            "completed_ppo_updates": metadata["completed_ppo_updates"], "stage": metadata["stage"],
                            "checkpoint": metadata["checkpoint"],
                            "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "bundle": metadata.get("bundle"),
                            "steps_per_environment": metadata.get("steps_per_environment")})
    confirmation = saved.get('confirmation')
    if confirmation and confirmation['status'] == 'completed':
        path = (index.parent / Path(confirmation['report']).relative_to(recorded)).resolve()
        if sha256(path.read_bytes()).hexdigest() != confirmation['report_sha256']:
            raise ValueError('joint confirmation report changed after training accepted it')
        report = read_json(path)
        periodic = {entry['data']['policy_id']: entry for entry in evaluations if entry['stage'] == 'periodic'}
        bundles = [periodic[policy]['bundle'] for policy in confirmation['policies']]
        for row in report['candidates']:
            metadata = periodic[row['policy_id']]
            evaluations.append({**metadata, 'report': str(path), 'stage': 'confirmation',
                                'data': {**report, **row, 'candidate_bundles': bundles}})
    evaluations.sort(key=lambda entry: (entry["completed_ppo_updates"] or 0, entry["stage"]))
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    scalars = {tag: [(entry.step, entry.value) for entry in events.Scalars(tag)]
               for tag in events.Tags()["scalars"] if not tag.endswith("/time")}
    return evaluations, scalars


def checkpoint_for(evaluation):
    candidate = Path(evaluation["checkpoint"])
    expected = evaluation["checkpoint_sha256"]
    return candidate.resolve() if candidate.exists() and sha256(candidate.read_bytes()).hexdigest() == expected else None


def select_best(directory, evaluations):
    selected = directory / "best_model.json"
    if selected.exists() and evaluations:
        best = read_json(selected)
        matching = [entry for entry in evaluations if (entry["data"]["policy_id"], entry["data"]["profile_id"])
                    == (best["policy_id"], best["profile_id"])]
        if matching:
            return matching[0]
    eligible = [entry for entry in evaluations if entry["stage"] == "periodic"] or evaluations
    return max(eligible, key=lambda entry: (entry["data"]["elo"], -entry["completed_ppo_updates"])) if eligible else None


def plot(scalars, evaluations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for stage, marker in (("periodic", "o-"), ("confirmation", "x")):
        rows = sorted((entry for entry in evaluations if entry["stage"] == stage),
                      key=lambda entry: entry["completed_ppo_updates"])
        if rows:
            x = [entry["completed_ppo_updates"] for entry in rows]
            y = [entry["data"]["elo"] for entry in rows]
            axes[0, 0].plot(x, y, marker, label=stage)
            intervals = [entry["data"].get("elo_ci") for entry in rows]
            if all(interval is not None for interval in intervals):
                axes[0, 0].fill_between(x, [ci["lower"] for ci in intervals],
                                       [ci["upper"] for ci in intervals], alpha=.15)
    axes[0, 0].set(title="Fixed-profile Arena Elo", ylabel="Elo (95% approximate interval)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=7)
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
            axis.plot(data[:, 0] + 1, data[:, 1], alpha=.2, linewidth=.4)
            window = min(100, max(1, len(data) // 20))
            smooth = np.convolve(data[:, 1], np.ones(window) / window, mode="valid")
            axis.plot(data[window - 1:, 0] + 1, smooth, label=tag.split("/")[-1])
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    if scalars.get("Loss/learning_rate"):
        axes[1, 1].set_yscale("log")
    axes[1, 0].set_ylabel("Environment transitions / second")
    for axis in axes.flat:
        axis.set_xlabel("Completed PPO updates")
        axis.grid(alpha=.15)
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)


def _match_evidence(game):
    """Keep match behavior and identities, excluding paths and measured latency."""
    value = {key: item for key, item in game.items() if key not in ('replay', 'decision_timing')}
    if 'forfeits' in value:
        value['forfeits'] = [{key: item for key, item in event.items() if key != 'elapsed_ns'}
                             for event in value['forfeits']]
    return value


def demos(checkpoint, report, output, *, render=False):
    """Use Arena for re-evaluation and its verified replay for presentation."""
    from generals_arena.competitions.evaluation import evaluate
    from generals_arena.matches.results import seat_outcome
    from generals_arena.policies.artifacts import validate_bundle
    from generals_rl.export import export_policy
    from recordings import replay_arena_game

    bundle = export_policy(checkpoint, output / "demo_bundle")
    if validate_bundle(bundle)["policy_id"] != report["policy_id"]:
        raise ValueError("demo checkpoint does not identify the evaluated policy")
    games = report["results"]
    if not all(game.get("replay") and Path(game["replay"]).is_file() for game in games):
        joint = 'candidate_bundles' in report
        recorded = evaluate(report['candidate_bundles'] if joint else bundle,
                            report["profile"], output / "recorded_evaluation.json",
                            device=report["device"], backend=report["backend"],
                            batch_size=report["batch_size"], cpu_threads=report["cpu_threads"],
                            record_replays=True)
        if ((not joint and recorded["policy_id"] != report["policy_id"]) or recorded["profile_id"] != report["profile_id"]
                or [_match_evidence(game) for game in recorded["results"]]
                != [_match_evidence(game) for game in games]):
            raise RuntimeError("Arena re-evaluation did not reproduce the evaluated games; retain the new report separately")
        games = recorded["results"]
    if 'candidate_bundles' in report:
        candidate = next(row['entrant_id'] for row in report['candidates'] if row['policy_id'] == report['policy_id'])
        names = {row['entrant_id']: row['name'] for row in report['ranking']}
        games = [{**game, 'candidate_seat': game['entrant_ids'].index(candidate),
                  'opponent': names[next(key for key in game['entrant_ids'] if key != candidate)],
                  'outcome': seat_outcome(game, game['entrant_ids'].index(candidate))}
                 for game in games if candidate in game['entrant_ids']]
    chosen = []
    for opponent in sorted({game["opponent"] for game in games}):
        for outcome in ("loss", "win"):
            matches = sorted((game for game in games if game["opponent"] == opponent and game["outcome"] == outcome),
                             key=lambda game: (game["ticks"], game["game_id"]))
            if matches:
                chosen.append(matches[len(matches) // 2])
    cases = []
    for game in chosen:
        case = replay_arena_game(game, output / "recordings",
                                 metadata={"policy_id": report["policy_id"], "profile_id": report["profile_id"]})
        if render:
            from render_recording import render_recording
            case["video"] = render_recording(case["frames"], output / "videos" / f"{case['name']}.mp4")
        cases.append(case)
    return {"cases": cases, "selection": "one median-duration loss and win per opponent when available",
            "reason": "evaluation contains only draws" if not cases else None,
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
    output = (args.output or directory / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluations, scalars = read_run(directory)
    best = select_best(directory, evaluations)
    best_checkpoint = checkpoint_for(best) if best else None
    summary = {"run": str(directory), "evaluations": len(evaluations),
               "selection_criterion": "fixed_profile_elo",
               "best": ({key: value for key, value in best.items() if key != "data"} |
                        {**{key: best["data"][key] for key in ("elo", "elo_ci", "policy_id", "profile_id", "by_opponent")},
                         "checkpoint": str(best_checkpoint) if best_checkpoint else None,
                         "checkpoint_sha256": best["checkpoint_sha256"]}) if best else None,
               "latest": ({key: evaluations[-1]["data"][key] for key in ("elo", "elo_ci", "by_opponent")}
                          if evaluations else None),
               "scalar_summary": {tag: {"count": len(values), "last": values[-1][1],
                                         "last_100_mean": float(np.mean([value for _, value in values[-100:]]))}
                                  for tag, values in scalars.items() if values}}
    if (directory / "summary.json").exists():
        summary["training"] = read_json(directory / "summary.json")
    plot(scalars, evaluations, output)
    if args.demo or args.render:
        demo_checkpoint = args.checkpoint.resolve() if args.checkpoint else best_checkpoint
        if demo_checkpoint is None:
            raise ValueError("selected weights are no longer retained; provide --checkpoint and --report")
        if args.report is None and best is None:
            raise ValueError("no evaluation report found; provide --report for demo replay")
        demo_report_path = args.report.resolve() if args.report else Path(best["report"]).resolve()
        report = read_json(args.report) if args.report else best["data"]
        summary["demos"] = demos(demo_checkpoint, report, output, render=args.render)
        summary["demo_selection"] = {
            "checkpoint": str(demo_checkpoint),
            "checkpoint_sha256": sha256(demo_checkpoint.read_bytes()).hexdigest(),
            "report": str(demo_report_path), "seed": report["profile"]["seed"],
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"curves": str(output / "curves.png"),
                      "summary": str(output / "summary.json"),
                      "best_checkpoint": str(best_checkpoint) if best_checkpoint else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
