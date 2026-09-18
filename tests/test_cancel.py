import dawgz
import json
import pytest
import subprocess

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from klej import runner
from klej.__main__ import _cancel
from klej.runner import (
    cleanup_reservation,
    lock_documents,
    register_reservation_jobs,
    reserve_targets,
)


class _Job(NamedTuple):
    name: str


def test_cancel_only_submits_cancellable_jobs_and_prints_a_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scheduler = SimpleNamespace(
        backend="slurm",
        results={
            _Job("pending"): "10",
            _Job("running"): "11",
            _Job("complete"): "12",
            _Job("failed"): "13",
            _Job("completing"): "14",
            _Job("klej_reservation_cleanup"): "15",
        },
    )
    monkeypatch.setattr(
        "klej.__main__.dawgz.Scheduler.load", lambda _path: scheduler
    )
    calls = []

    def run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[0] == "sacct":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "10|PENDING\n"
                    "11|RUNNING\n"
                    "12|COMPLETED\n"
                    "13|FAILED\n"
                    "14|COMPLETING\n"
                ),
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)

    _cancel("workflow", dry_run=False)

    assert calls[1:] == [
        (
            ["scancel", "10"],
            {"capture_output": True, "check": True, "text": True},
        ),
        (
            ["scancel", "11"],
            {"capture_output": True, "check": True, "text": True},
        ),
    ]
    assert capsys.readouterr().out == (
        "Workflow: workflow\n"
        "Cancelling 2 of 5 Slurm jobs (3 not cancellable).\n"
    )


def test_cancel_reports_when_all_jobs_are_already_inactive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scheduler = SimpleNamespace(
        backend="slurm", results={_Job("complete"): "10"}
    )
    monkeypatch.setattr(
        "klej.__main__.dawgz.Scheduler.load", lambda _path: scheduler
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n"
        ),
    )

    _cancel("workflow", dry_run=False)

    assert capsys.readouterr().out == (
        "Workflow: workflow\nNo cancellable Slurm jobs.\n"
    )


def test_cancel_preserves_cleanup_by_plan_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tmp_path / "records"
    path = records / "workflow"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "plan_version": 2,
            "physical_jobs": [
                {"name": "contender", "role": "attempt", "slurm_job_id": "20"},
                {"name": "success", "role": "gate", "slurm_job_id": "21"},
                {
                    "name": "not_the_old_name",
                    "role": "cleanup",
                    "slurm_job_id": "22",
                },
            ],
        })
    )
    dawgz.set_dawgz_dir(records)
    scheduler = SimpleNamespace(
        backend="slurm",
        results={
            _Job("scheduler_only"): "23",
            _Job("klej_reservation_cleanup"): "24",
        },
    )
    monkeypatch.setattr(
        "klej.__main__.dawgz.Scheduler.load", lambda _path: scheduler
    )
    calls = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = (
            "20|PENDING\n21|RUNNING\n22|PENDING\n23|PENDING\n24|PENDING\n"
            if command[0] == "sacct"
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)

    _cancel("workflow", dry_run=False)

    assert calls[0][2] == "20,21,23"
    assert calls[1:] == [
        ["scancel", "20"],
        ["scancel", "21"],
        ["scancel", "23"],
    ]


def test_cancelled_workflow_cleanup_waits_for_active_intermediate_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = [
        {"name": "first", "role": "stage", "slurm_job_id": "20"},
        {"name": "commit_first", "role": "commit", "slurm_job_id": "21"},
        {"name": "second", "role": "stage", "slurm_job_id": "22"},
        {"name": "cleanup", "role": "cleanup", "slurm_job_id": "23"},
    ]
    records = tmp_path / "records"
    path = records / "workflow"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({"plan_version": 2, "physical_jobs": physical})
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        "klej.__main__.dawgz.Scheduler.load",
        lambda _path: SimpleNamespace(backend="slurm", results={}),
    )
    reservation = reserve_targets(tmp_path, ["stage"])
    register_reservation_jobs(tmp_path, *reservation, physical)
    squeue_outputs = iter(["21\n", ""])
    calls = []
    waits = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "sacct":
            output = "20|COMPLETED\n21|RUNNING\n22|CANCELLED\n23|PENDING\n"
        elif command[0] == "squeue":
            output = next(squeue_outputs)
        elif command[:3] == ["scontrol", "show", "config"]:
            output = "KillWait = 90 sec\n"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(runner.time, "sleep", waits.append)
    monkeypatch.setenv("SLURM_JOB_ID", "23")

    _cancel("workflow", dry_run=False)
    cleanup_reservation(tmp_path, *reservation)

    assert calls.count(["scancel", "21"]) == 2
    assert ["scontrol", "show", "config"] in calls
    assert waits == [95]
    assert lock_documents(tmp_path) == {}
