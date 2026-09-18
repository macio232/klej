import csv
import dawgz
import json
import pytest
import subprocess

from pathlib import Path

from klej.__main__ import main


def _plan(
    records: Path,
    workflow_id: str,
    job_id: str,
    backend: str = "slurm",
) -> Path:
    path = records / workflow_id
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "pipeline": f"{workflow_id}/dvc.yaml",
            "backend": backend,
            "jobs": {job_id: "job"},
        })
    )
    (path / "job.log").write_text("output\n")
    return path


def _index(records: Path, *workflow_ids: str) -> None:
    with (records / "workflows.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        for workflow_id in workflow_ids:
            writer.writerow((
                "pipeline",
                workflow_id,
                "date",
                "slurm",
                "1",
                "0",
            ))
        writer.writerow(("malformed",))


def test_clear_workflow_confirms_and_removes_directory_and_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "finished", "10")
    _plan(records, "other", "11")
    _index(records, "finished", "other")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n11|RUNNING\n"
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    main(["logs", "--clear", "finished"])

    assert not (records / "finished").exists()
    assert (records / "other").is_dir()
    rows = list(csv.reader((records / "workflows.csv").open()))
    assert [row[1] for row in rows if len(row) > 1] == ["other"]
    assert ["malformed"] in rows
    assert "Removed workflow finished." in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["n", "", "anything"])
def test_clear_workflow_decline_keeps_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "finished", "10")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n"
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    main(["logs", "--clear", "finished"])

    assert (records / "finished").is_dir()
    assert "No workflow records removed." in capsys.readouterr().out


def test_clear_force_skips_confirmation_but_never_overrides_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "finished", "10")
    _plan(records, "active", "11")
    _plan(records, "unknown", "12")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n11|RUNNING\n"
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("--force unexpectedly prompted"),
    )

    main(["logs", "--clear", "finished", "--force"])
    assert not (records / "finished").exists()

    for workflow_id in ("active", "unknown"):
        with pytest.raises(SystemExit):
            main(["logs", "--clear", workflow_id, "--force"])
        assert "not known to be terminal" in capsys.readouterr().err
        assert (records / workflow_id).is_dir()


def test_clear_all_removes_terminal_records_and_keeps_active_and_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    for workflow_id, job_id in (
        ("success", "10"),
        ("failed", "11"),
        ("active", "12"),
        ("unknown", "13"),
    ):
        _plan(records, workflow_id, job_id)
    local_path = _plan(records, "local", "ignored", "local")
    local_job = dawgz.job(lambda: None, name="job")()
    scheduler = dawgz.schedulers.AsyncScheduler("test")
    scheduler.path = local_path
    scheduler.uid = "local"
    scheduler.order = {local_job: 0}
    scheduler.results = {local_job: None}
    scheduler.traces = {}
    scheduler.dump()
    _index(records, "success", "failed", "active", "unknown", "local")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="10|COMPLETED\n11|FAILED\n12|PENDING\n",
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")

    main(["logs", "--clear-all"])

    assert not (records / "success").exists()
    assert not (records / "failed").exists()
    assert not (records / "local").exists()
    assert (records / "active").is_dir()
    assert (records / "unknown").is_dir()
    assert "Removed 3 workflow records." in capsys.readouterr().out


def test_cleanup_flags_reject_log_options_without_removing_anything(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "finished", "10")
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["logs", "--clear", "finished", "--follow"])

    assert "cannot be combined" in capsys.readouterr().err
    assert (records / "finished").is_dir()


def test_clear_treats_confirmation_eof_as_decline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "finished", "10")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n"
        ),
    )

    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)

    main(["logs", "--clear", "finished"])

    assert (records / "finished").is_dir()
    assert "No workflow records removed." in capsys.readouterr().out


def test_clear_unknown_workflow_is_actionable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dawgz.set_dawgz_dir(tmp_path / ".dawgz")

    with pytest.raises(SystemExit):
        main(["logs", "--clear", "missing", "--force"])

    assert "Unknown klej workflow 'missing'" in capsys.readouterr().err


def test_clear_all_force_skips_confirmation_and_handles_no_finished_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _plan(records, "active", "10")
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|RUNNING\n"
        ),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: pytest.fail("--force unexpectedly prompted"),
    )

    main(["logs", "--clear-all", "--force"])

    assert (records / "active").is_dir()
    assert capsys.readouterr().out == "No finished workflow records to clear.\n"


def test_clear_checks_hidden_physical_jobs_in_version_two_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "alternatives"
    path.mkdir(parents=True)
    logical_id = "a" * 64
    physical = [
        {
            "name": "winner",
            "role": "attempt",
            "logical_id": logical_id,
            "profile": "small",
            "slurm_job_id": "10",
        },
        {
            "name": "loser",
            "role": "attempt",
            "logical_id": logical_id,
            "profile": "large",
            "slurm_job_id": "11",
        },
        {
            "name": "gate",
            "role": "gate",
            "logical_id": logical_id,
            "profile": "control",
            "slurm_job_id": "12",
        },
    ]
    (path / "plan.json").write_text(
        json.dumps({
            "plan_version": 2,
            "pipeline": "dvc.yaml",
            "backend": "slurm",
            "jobs": {job["slurm_job_id"]: job["name"] for job in physical},
            "stages": [
                {
                    "id": logical_id,
                    "target": "dvc.yaml:train",
                    "profiles": ["small", "large"],
                }
            ],
            "physical_jobs": physical,
        })
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="10|COMPLETED||0:0\n11|PENDING||\n12|PENDING||\n",
        ),
    )

    with pytest.raises(SystemExit):
        main(["logs", "--clear", "alternatives", "--force"])

    assert "not known to be terminal" in capsys.readouterr().err
    assert path.is_dir()
