"""Best-run metrics retain their model identity when replaying an earlier model."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "analyze_iteration_tool", Path(__file__).resolve().parents[1] / "analyze_iteration.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def test_explicit_demo_does_not_replace_best_model_identity(tmp_path, monkeypatch, capsys):
    run, output = tmp_path / "run", tmp_path / "analysis"
    run.mkdir()
    (run / "run_config.json").write_text("{}")
    evaluations = []
    for iteration, rate, seed in ((99, 0.48, 1001042), (199, 0.7325, 271828)):
        directory = run / f"snapshot_{iteration}"
        directory.mkdir()
        checkpoint = directory / "snapshot.pt"
        checkpoint.write_bytes(f"distinct model weights at update {iteration}".encode())
        report = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": seed,
            "by_ai": {kind: {"win_rate": rate} for kind in analyze.KINDS},
            "overall": {"win_rate": rate},
        }
        report_path = directory / "report.json"
        report_path.write_text(json.dumps(report))
        evaluations.append({"iteration": iteration, "report": str(report_path), "data": report})
    early, best = evaluations
    calls = []
    monkeypatch.setattr(analyze, "read_run", lambda directory: (evaluations, {}))
    monkeypatch.setattr(analyze, "plot", lambda *args: None)

    def record_demo(checkpoint, report, directory, *, render):
        # Use actual distinct file bytes, independently of the report metadata.
        assert sha256(checkpoint.read_bytes()).hexdigest() == early["data"]["checkpoint_sha256"]
        assert report == early["data"] and directory == output and render is False
        calls.append(checkpoint)
        return {"cases": []}

    monkeypatch.setattr(analyze, "demos", record_demo)
    assert analyze.main([str(run), "--output", str(output), "--demo",
                         "--checkpoint", early["data"]["checkpoint"],
                         "--report", early["report"]]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["best"]["iteration"] == 199
    assert summary["best"]["by_ai"] == best["data"]["by_ai"]
    assert summary["best"]["overall"] == best["data"]["overall"]
    assert summary["best"]["checkpoint"] == best["data"]["checkpoint"]
    assert summary["best"]["checkpoint_sha256"] == best["data"]["checkpoint_sha256"]
    assert summary["demo_selection"] == {
        "checkpoint": early["data"]["checkpoint"],
        "checkpoint_sha256": early["data"]["checkpoint_sha256"],
        "report": early["report"], "seed": 1001042,
    }
    assert calls == [Path(early["data"]["checkpoint"])]
    console = json.loads(capsys.readouterr().out)
    assert console["best_checkpoint"] == best["data"]["checkpoint"]
    assert summary["best"]["checkpoint_sha256"] != summary["demo_selection"]["checkpoint_sha256"]
