import dawgz
import json
import pytest
import subprocess

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from klej.__main__ import main
from klej.status import (
    _plans_accounting_start,
    _read_plan,
    _slurm_details,
    _slurm_queue,
    _slurm_states,
    _WorkflowStatus,
)


def _write_plan(
    records: Path,
    workflow_id: str,
    pipeline: str,
    jobs: dict[str, str],
    backend: str = "slurm",
) -> None:
    path = records / workflow_id
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({"pipeline": pipeline, "backend": backend, "jobs": jobs})
    )


def _write_alternative_plan(
    records: Path,
    workflow_id: str = "alternatives",
    winner: dict[str, object] | None = None,
) -> Path:
    path = records / workflow_id
    path.mkdir(parents=True)
    logical_id = "a" * 64
    physical = [
        {
            "name": "small_attempt",
            "role": "attempt",
            "logical_id": logical_id,
            "target": "dvc.yaml:train",
            "profile": "small",
            "slurm_job_id": "10",
        },
        {
            "name": "large_attempt",
            "role": "attempt",
            "logical_id": logical_id,
            "target": "dvc.yaml:train",
            "profile": "large",
            "slurm_job_id": "11",
        },
        {
            "name": "train_gate",
            "role": "gate",
            "logical_id": logical_id,
            "target": "dvc.yaml:train",
            "profile": "control",
            "slurm_job_id": "12",
        },
        {
            "name": "cleanup",
            "role": "cleanup",
            "logical_id": None,
            "target": None,
            "profile": "control",
            "slurm_job_id": "13",
        },
    ]
    (path / "plan.json").write_text(
        json.dumps({
            "plan_version": 2,
            "pipeline": "dvc.yaml",
            "backend": "slurm",
            "jobs": {
                str(job["slurm_job_id"]): str(job["name"]) for job in physical
            },
            "stages": [
                {
                    "id": logical_id,
                    "name": "train",
                    "target": "dvc.yaml:train",
                    "profiles": ["small", "large"],
                }
            ],
            "physical_jobs": physical,
        })
    )
    if winner is not None:
        race = path / "races"
        race.mkdir()
        (race / f"{logical_id}.json").write_text(json.dumps(winner))
    return path


def test_status_aggregates_successful_alternatives_and_physical_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={
            "claiming_job_id": "10",
            "profile": "small",
            "outcome": "succeeded",
            "payload_exit_status": 0,
        },
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED|2026-09-17T12:00:00|0:0\n"
                "11|FAILED|2026-09-17T12:00:01|75:0\n"
                "12|COMPLETED|2026-09-17T12:00:02|0:0\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives", "--physical"])

    output = capsys.readouterr().out
    logical = output.partition("Physical jobs:")[0]
    assert "Counts: pending=0  running=0  succeeded=2" in output
    assert logical.count("dvc.yaml:train") == 1
    assert "winner=10 profile=small" in logical
    assert "superseded" in output
    assert "75:0" in output
    assert "gate" in output


def test_inspect_failed_filters_logical_and_physical_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={
            "claiming_job_id": "10",
            "outcome": "failed",
            "payload_exit_status": 23,
        },
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|FAILED|2026-09-17T12:00:00|23:0\n"
                "11|FAILED|2026-09-17T12:00:01|1:0\n"
                "12|CANCELLED|2026-09-17T12:00:02|0:15\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives", "--failed", "--physical"])

    output = capsys.readouterr().out
    logical, physical = output.split("Physical jobs:")
    assert "State: failed" in logical
    assert "Counts: pending=0  running=0  succeeded=1  failed=1" in logical
    assert "dvc.yaml:train" in logical
    physical_rows = physical.strip().splitlines()
    assert len(physical_rows) == 2
    assert "  10  " in physical_rows[1]
    assert "  11  " not in physical_rows[1]
    assert "superseded" not in physical


def test_inspect_failed_keeps_headers_when_no_jobs_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={
            "claiming_job_id": "10",
            "outcome": "succeeded",
            "payload_exit_status": 0,
        },
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED|2026-09-17T12:00:00|0:0\n"
                "11|CANCELLED|2026-09-17T12:00:01|0:15\n"
                "12|COMPLETED|2026-09-17T12:00:02|0:0\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives", "--failed", "--physical"])

    output = capsys.readouterr().out
    logical, physical = output.split("Physical jobs:")
    assert "Counts: pending=0  running=0  succeeded=2  failed=0" in logical
    assert "INDEX  JOB  SLURM ID  START TIME  STATE  DETAIL" in logical
    assert "INDEX  JOB  ROLE  PROFILE  SLURM ID" in physical
    assert "dvc.yaml:train" not in logical
    assert len(physical.strip().splitlines()) == 1


def test_status_shows_persisted_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = _write_alternative_plan(records)
    document = json.loads((path / "plan.json").read_text())
    document.update({
        "execution_root": str(tmp_path),
        "reservation_id": "abc123",
    })
    (path / "plan.json").write_text(json.dumps(document))
    owner = tmp_path / ".klej/locks/runs/abc123"
    owner.mkdir(parents=True)
    (owner / "owner.json").write_text(
        json.dumps({
            "cleanup_failure": {
                "operation": "confirm",
                "job_ids": ["21"],
                "recorded_at": "2026-09-18T12:00:00+00:00",
            }
        })
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|CANCELLED|2026-09-18T11:00:00|0:15\n"
                "11|CANCELLED|2026-09-18T11:00:01|0:15\n"
                "12|CANCELLED|2026-09-18T11:00:02|0:15\n"
                "13|FAILED|2026-09-18T11:00:03|1:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives", "--physical"])

    cleanup_rows = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "cleanup" in line
    ]
    assert cleanup_rows
    assert all(
        "cleanup confirm failed for Slurm jobs 21" in row
        for row in cleanup_rows
    )


def test_recorded_payload_failure_dominates_cancelled_slurm_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={
            "claiming_job_id": "10",
            "outcome": "failed",
            "payload_exit_status": 23,
        },
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|CANCELLED|2026-09-17T12:00:00|0:15\n"
                "11|PENDING||0:0\n"
                "12|CANCELLED||0:0\n"
                "13|PENDING||0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives"])

    output = capsys.readouterr().out
    assert "Terminal: no" in output
    assert "dvc.yaml:train" in output
    assert "failed" in output


def test_cancelled_running_claimant_is_logically_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={"claiming_job_id": "10", "outcome": "running"},
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|CANCELLED|2026-09-17T12:00:00|0:15\n"
                "11|CANCELLED|2026-09-17T12:00:01|0:15\n"
                "12|CANCELLED|2026-09-17T12:00:02|0:15\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives"])

    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if "dvc.yaml:train" in line)
    assert "State: cancelled" in output
    assert "cancelled" in row
    assert "failed" not in row


def test_cancelled_gate_after_successful_payload_is_logically_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={
            "claiming_job_id": "10",
            "outcome": "succeeded",
            "payload_exit_status": 0,
        },
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED|2026-09-17T12:00:00|0:0\n"
                "11|CANCELLED|2026-09-17T12:00:01|0:15\n"
                "12|CANCELLED|2026-09-17T12:00:02|0:15\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives"])

    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if "dvc.yaml:train" in line)
    assert "State: cancelled" in output
    assert "cancelled" in row
    assert "failed" not in row


def test_failed_claimant_remains_failed_when_gate_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(
        records,
        winner={"claiming_job_id": "10", "outcome": "running"},
    )
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|FAILED|2026-09-17T12:00:00|23:0\n"
                "11|CANCELLED|2026-09-17T12:00:01|0:15\n"
                "12|CANCELLED|2026-09-17T12:00:02|0:15\n"
                "13|COMPLETED|2026-09-17T12:00:03|0:0\n"
            ),
        ),
    )

    main(["inspect", "alternatives"])

    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if "dvc.yaml:train" in line)
    assert "State: failed" in output
    assert "failed" in row


def test_status_rejects_unsupported_plan_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "future"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "plan_version": 99,
            "pipeline": "dvc.yaml",
            "backend": "slurm",
            "jobs": {},
        })
    )
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["status"])

    assert (
        "Unsupported klej workflow plan version 99" in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("accounting", "expected"),
    [
        (
            "10|PENDING||0:0\n11|PENDING||0:0\n12|PENDING||0:0\n13|PENDING||0:0\n",
            "pending",
        ),
        (
            "10|CANCELLED||0:0\n11|CANCELLED||0:0\n12|CANCELLED||0:0\n13|COMPLETED||0:0\n",
            "cancelled",
        ),
        (
            "10|FAILED||2:0\n11|CANCELLED||0:0\n12|CANCELLED||0:0\n13|COMPLETED||0:0\n",
            "failed",
        ),
    ],
)
def test_alternative_status_without_winner_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    accounting: str,
    expected: str,
) -> None:
    records = tmp_path / ".dawgz"
    _write_alternative_plan(records)
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=accounting
        ),
    )

    main(["inspect", "alternatives"])

    row = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "dvc.yaml:train" in line
    )
    assert expected in row


def test_inspect_requires_workflow_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dawgz.set_dawgz_dir(tmp_path / "records")

    with pytest.raises(SystemExit):
        main(["inspect", "--physical"])

    assert (
        "the following arguments are required: workflow_id"
        in capsys.readouterr().err
    )


def test_status_rejects_physical(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["status", "--physical"])

    assert "unrecognized arguments: --physical" in capsys.readouterr().err


def test_fast_status_shows_queue_progress_for_active_slurm_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(
        records,
        "active_two",
        "pipelines/two/dvc.yaml",
        {"20": "done", "21": "running", "22": "failed", "23": "unknown"},
    )
    _write_plan(
        records,
        "active_one",
        "dvc.yaml",
        {"10": "queued", "11": "configuring", "12": "done"},
    )
    _write_plan(records, "complete", "old/dvc.yaml", {"30": "done"})
    _write_plan(
        records, "local", "local/dvc.yaml", {"ignored": "done"}, "local"
    )
    (records / "unrelated").mkdir()
    dawgz.set_dawgz_dir(records)

    def run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "check": True, "text": True}
        assert command == [
            "squeue",
            "--noheader",
            "--me",
            "--format=%A|%T",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=("10|PENDING\n11|CONFIGURING\n20|COMPLETING\n21|RUNNING\n"),
        )

    monkeypatch.setattr(subprocess, "run", run)

    main(["status", "--fast"])

    output = capsys.readouterr().out
    assert "WORKFLOW" in output
    assert "PIPELINE" in output
    assert "TOTAL" in output
    assert "RUNNING" in output
    assert "PENDING" in output
    assert "NOT_IN_QUEUE" in output
    assert "active_one" in output
    assert "dvc.yaml" in output
    assert "active_two" in output
    assert "pipelines/two/dvc.yaml" in output
    assert "complete" not in output
    assert "local" not in output
    assert "unrelated" not in output
    rows = [line.split() for line in output.splitlines()]
    assert rows[1] == [
        "active_one",
        "dvc.yaml",
        "3",
        "1",
        "1",
        "1",
    ]
    assert rows[2] == [
        "active_two",
        "pipelines/two/dvc.yaml",
        "4",
        "0",
        "2",
        "2",
    ]


def test_status_uses_accounting_for_queue_selected_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(
        records,
        "active",
        "dvc.yaml",
        {"10": "queued", "11": "done", "12": "failed"},
    )
    _write_plan(records, "inactive", "old/dvc.yaml", {"20": "done"})
    dawgz.set_dawgz_dir(records)
    commands = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output = (
            "10|PENDING\n"
            if command[0] == "squeue"
            else "10|PENDING|\n11|COMPLETED|0:0\n12|FAILED|1:0\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)

    main(["status"])

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == [
        "WORKFLOW",
        "PIPELINE",
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "UNKNOWN",
    ]
    assert rows[1] == ["active", "dvc.yaml", "1", "0", "1", "1", "0", "0"]
    assert [command[0] for command in commands] == ["squeue", "sacct"]
    assert commands[1][2] == "10,11,12"


def test_status_omits_workflow_that_finishes_after_queue_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "workflow", "dvc.yaml", {"10": "job"})
    dawgz.set_dawgz_dir(records)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = (
            "10|RUNNING\n" if command[0] == "squeue" else "10|COMPLETED|0:0\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)

    main(["status"])

    assert capsys.readouterr().out == "No active klej workflows.\n"


def test_fast_status_summary_does_not_load_submitted_workflow_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = _write_alternative_plan(records)
    (path / "dump.pkl").write_bytes(b"not needed for summary")
    dawgz.set_dawgz_dir(records)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == "squeue"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="10|PENDING\n11|RUNNING\n12|PENDING\n13|PENDING\n",
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        dawgz.Scheduler,
        "load",
        lambda _path: pytest.fail("summary loaded the DAWGZ dump"),
    )

    main(["status", "--fast"])

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[1] == ["alternatives", "dvc.yaml", "2", "1", "1", "0"]


@pytest.mark.parametrize(
    ("state", "category"),
    [
        ("COMPLETED+", "succeeded"),
        ("CANCELLED by 12345", "cancelled"),
        ("PENDING", "pending"),
        ("REQUEUED", "pending"),
        ("REQUEUE_FED", "pending"),
        ("REQUEUE_HOLD", "pending"),
        ("RESV_DEL_HOLD", "pending"),
        ("SPECIAL_EXIT", "pending"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("CONFIGURING", "running"),
        ("EXPEDITING", "running"),
        ("POWER_UP_NODE", "running"),
        ("RESIZING", "running"),
        ("SIGNALING", "running"),
        ("STAGE_OUT", "running"),
        ("STOPPED", "running"),
        ("SUSPENDED", "running"),
        ("UPDATE_DB", "running"),
        ("BOOT_FAIL", "failed"),
        ("DEADLINE", "failed"),
        ("FAILED", "failed"),
        ("LAUNCH_FAILED", "failed"),
        ("NODE_FAIL", "failed"),
        ("OUT_OF_MEMORY", "failed"),
        ("PREEMPTED", "failed"),
        ("RECONFIG_FAIL", "failed"),
        ("REVOKED", "failed"),
        ("TIMEOUT", "failed"),
        ("FUTURE_STATE", "unknown"),
        (None, "unknown"),
    ],
)
def test_slurm_state_normalization(state: str | None, category: str) -> None:
    from klej.status import _category

    assert _category(state) == category


@pytest.mark.parametrize(
    ("states", "aggregate", "terminal"),
    [
        (("running", "failed"), "running", False),
        (("pending", "failed"), "pending", False),
        (("failed", "cancelled"), "failed", True),
        (("cancelled", "succeeded"), "cancelled", True),
        (("succeeded", "succeeded"), "succeeded", True),
        (("succeeded", "unknown"), "unknown", False),
    ],
)
def test_aggregate_state_precedence_and_terminal_detection(
    states: tuple[str, ...], aggregate: str, terminal: bool
) -> None:
    workflow = _WorkflowStatus(
        SimpleNamespace(),
        tuple(SimpleNamespace(state=state) for state in states),
    )

    assert workflow.state == aggregate
    assert workflow.terminal is terminal


def test_status_and_failed_listing_use_terminal_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "failed", "failed/dvc.yaml", {"1": "bad"})
    _write_plan(records, "cancelled", "cancel/dvc.yaml", {"2": "gone"})
    _write_plan(records, "unknown", "old/dvc.yaml", {"3": "lost"})
    dawgz.set_dawgz_dir(records)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[4] == "JobID,State,ExitCode"
        return subprocess.CompletedProcess(
            command, 0, stdout="1|TIMEOUT|1:0\n2|CANCELLED by 42|0:0\n"
        )

    monkeypatch.setattr(subprocess, "run", run)

    main(["status", "--failed"])

    output = capsys.readouterr().out
    assert "failed" in output
    assert "cancelled" in output
    assert "unknown" not in output


def test_status_reports_when_no_workflows_are_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "complete", "dvc.yaml", {"10": "done"})
    dawgz.set_dawgz_dir(records)
    commands = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(
        subprocess,
        "run",
        run,
    )

    main(["status"])

    assert capsys.readouterr().out == "No active klej workflows.\n"
    assert [command[0] for command in commands] == ["squeue"]


def test_status_reports_an_actionable_sacct_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "workflow", "dvc.yaml", {"10": "stage"})
    dawgz.set_dawgz_dir(records)

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("sacct")

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(SystemExit):
        main(["inspect", "workflow"])

    assert (
        "Cannot query Slurm status: sacct was not found"
        in capsys.readouterr().err
    )


def test_status_reports_an_actionable_squeue_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "workflow", "dvc.yaml", {"10": "stage"})
    dawgz.set_dawgz_dir(records)

    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("squeue")

    monkeypatch.setattr(subprocess, "run", missing)

    with pytest.raises(SystemExit):
        main(["status"])

    assert (
        "Cannot query Slurm queue: squeue was not found"
        in capsys.readouterr().err
    )


def test_status_with_no_records_does_not_query_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dawgz.set_dawgz_dir(tmp_path / ".dawgz")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("queried Slurm without jobs"),
    )

    main(["status"])

    assert capsys.readouterr().out == "No active klej workflows.\n"


def test_slurm_queue_batches_job_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)

    assert _slurm_queue([str(index) for index in range(1_001)]) == set()
    assert [len(command[3].split(",")) for command in commands] == [1_000, 1]


def test_status_rejects_a_malformed_klej_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "broken"
    path.mkdir(parents=True)
    (path / "plan.json").write_text("{}")
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["status"])

    assert "Malformed klej workflow plan" in capsys.readouterr().err


def test_status_reads_planned_version_two_record_without_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = _write_alternative_plan(records, workflow_id="planned")
    document = json.loads((path / "plan.json").read_text())
    document.pop("jobs")
    document["submission_state"] = "planned"
    for job in document["physical_jobs"]:
        job["slurm_job_id"] = None
    (path / "plan.json").write_text(json.dumps(document))
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("planned record queried Slurm"),
    )

    main(["inspect", "planned"])

    output = capsys.readouterr().out
    assert "State: unknown" in output
    assert "Terminal: no" in output


def test_status_rejects_nonplanned_version_two_record_without_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / ".dawgz"
    path = _write_alternative_plan(records, workflow_id="failed")
    document = json.loads((path / "plan.json").read_text())
    document.pop("jobs")
    document["submission_state"] = "failed"
    (path / "plan.json").write_text(json.dumps(document))
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["inspect", "failed"])

    assert "Malformed klej workflow plan" in capsys.readouterr().err


def test_status_rejects_malformed_accounting_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / ".dawgz"
    path = _write_alternative_plan(records)
    document = json.loads((path / "plan.json").read_text())
    document["accounting_start"] = "not-a-date"
    (path / "plan.json").write_text(json.dumps(document))
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["inspect", "alternatives"])

    assert "Malformed accounting start time" in capsys.readouterr().err


def test_detailed_status_uses_dawgz_order_targets_and_traces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "workflow"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "pipeline": "dvc.yaml",
            "backend": "slurm",
            "jobs": {"10": "simulate_x", "11": "simulate_x"},
        })
    )
    first = dawgz.job(lambda: None, name="simulate_x")()
    second = dawgz.job(lambda: None, name="simulate_x")()
    skipped = dawgz.job(lambda: None, name="commit_simulate_x")()
    first.klej_target = "dvc.yaml:simulate@{'x': 1}"
    second.klej_target = "dvc.yaml:simulate@{'x': 2}"
    scheduler = dawgz.schedulers.SlurmScheduler("test")
    scheduler.date = datetime.fromisoformat("2026-09-17T12:00:00")
    scheduler.path = path
    scheduler.uid = "workflow"
    scheduler.order = {second: 0, first: 1, skipped: 2}
    scheduler.results = {first: "10", second: "11"}
    scheduler.traces = {
        skipped: "Traceback\nJobNeverSatisfiedError: dependency failed"
    }
    scheduler.dump()
    dawgz.set_dawgz_dir(records)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[-2:] == ["--starttime", "2026-09-17T11:55:00"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED+|2026-09-17T12:05:00\n"
                "11|OUT_OF_MEMORY|2026-09-17T12:10:00\n"
            ),
        )

    monkeypatch.setattr(subprocess, "run", run)

    main(["inspect", "workflow"])

    output = capsys.readouterr().out
    assert "State: failed" in output
    assert "Terminal: yes" in output
    assert "dvc.yaml:simulate@{'x': 2}" in output
    assert "dvc.yaml:simulate@{'x': 1}" in output
    assert output.index("dvc.yaml:simulate@{'x': 1}") < output.index(
        "dvc.yaml:simulate@{'x': 2}"
    )
    assert "COMPLETED+" in output
    assert "OUT_OF_MEMORY" in output
    assert "JobNeverSatisfiedError: dependency failed" in output
    assert "START TIME" in output
    assert "2026-09-17T12:05:00" in output
    assert "2026-09-17T12:10:00" in output
    skipped_row = next(
        line for line in output.splitlines() if "commit_simulate_x" in line
    )
    assert "  -  " in skipped_row


def test_sacct_queries_large_job_sets_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        ids = command[2].split(",")
        return subprocess.CompletedProcess(
            command, 0, stdout="".join(f"{job_id}|RUNNING\n" for job_id in ids)
        )

    monkeypatch.setattr(subprocess, "run", run)

    states = _slurm_states([str(index) for index in range(1_001)])

    assert len(calls) == 2
    assert len(calls[0][2].split(",")) == 1_000
    assert states["1000"] == "RUNNING"


def test_sacct_query_uses_accounting_start_for_every_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)

    _slurm_states(
        [str(index) for index in range(1_001)],
        datetime.fromisoformat("2026-09-17T11:55:00"),
    )

    assert len(calls) == 2
    assert all(
        command[-2:] == ["--starttime", "2026-09-17T11:55:00"]
        for command in calls
    )


def test_multi_workflow_accounting_uses_earliest_complete_bound(
    tmp_path: Path,
) -> None:
    records = tmp_path / ".dawgz"
    paths = [
        _write_alternative_plan(records, workflow_id=workflow_id)
        for workflow_id in ("first", "second")
    ]
    for path, value in zip(
        paths,
        ("2026-09-17T12:00:00", "2026-09-17T10:00:00"),
        strict=True,
    ):
        document = json.loads((path / "plan.json").read_text())
        document["accounting_start"] = value
        (path / "plan.json").write_text(json.dumps(document))

    plans = [_read_plan(path / "plan.json") for path in paths]

    assert _plans_accounting_start(plans) == datetime.fromisoformat(
        "2026-09-17T09:55:00"
    )
    assert (
        _plans_accounting_start([
            plans[0],
            _read_plan(
                _write_alternative_plan(records, "unbounded") / "plan.json"
            ),
        ])
        is None
    )


def test_detailed_sacct_queries_start_time_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        ids = command[2].split(",")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="".join(
                f"{job_id}|RUNNING|2026-09-17T12:00:00\n" for job_id in ids
            ),
        )

    monkeypatch.setattr(subprocess, "run", run)

    states, start_times, exit_codes = _slurm_details([
        str(index) for index in range(1_001)
    ])

    assert len(calls) == 2
    assert calls[0][4] == "JobID,State,Start,ExitCode"
    assert len(calls[0][2].split(",")) == 1_000
    assert states["1000"] == "RUNNING"
    assert start_times["1000"] == "2026-09-17T12:00:00"
    assert exit_codes == {}


def test_detailed_status_rejects_local_workflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "local", "dvc.yaml", {"ignored": "job"}, "local")
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["inspect", "local"])

    assert "only supported for Slurm" in capsys.readouterr().err


def test_detailed_status_reads_legacy_plan_without_target_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "legacy", "dvc.yaml", {"10": "recorded_name"})
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|COMPLETED\n"
        ),
    )

    main(["inspect", "legacy"])

    output = capsys.readouterr().out
    assert "recorded_name" in output
    assert "succeeded" in output


def test_finished_status_lists_terminal_slurm_and_local_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "success", "success/dvc.yaml", {"10": "done"})
    _write_plan(records, "failure", "failure/dvc.yaml", {"11": "bad"})
    _write_plan(records, "cancelled", "cancel/dvc.yaml", {"12": "gone"})
    _write_plan(records, "active", "active/dvc.yaml", {"13": "running"})
    _write_plan(records, "unknown", "unknown/dvc.yaml", {"14": "lost"})

    local_path = records / "local"
    local_path.mkdir(parents=True)
    (local_path / "plan.json").write_text(
        json.dumps({
            "pipeline": "local/dvc.yaml",
            "backend": "local",
            "jobs": {},
        })
    )
    local_job = dawgz.job(lambda: None, name="local_job")()
    scheduler = dawgz.schedulers.AsyncScheduler("test")
    scheduler.path = local_path
    scheduler.uid = "local"
    scheduler.date = datetime.fromisoformat("2026-09-17T12:00:00")
    scheduler.order = {local_job: 0}
    scheduler.results = {local_job: None}
    scheduler.traces = {}
    scheduler.dump()
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED\n11|TIMEOUT\n12|CANCELLED by 42\n13|RUNNING\n"
            ),
        ),
    )

    main(["status", "--finished"])

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["WORKFLOW", "PIPELINE", "SUBMISSION", "STATUS"]
    assert ["cancelled", "cancel/dvc.yaml", "-", "cancelled"] in rows
    assert ["failure", "failure/dvc.yaml", "-", "failed"] in rows
    assert [
        "local",
        "local/dvc.yaml",
        "2026-09-17T12:00:00",
        "succeeded",
    ] in rows
    assert ["success", "success/dvc.yaml", "-", "succeeded"] in rows
    assert not any(row[0] in {"active", "unknown"} for row in rows[1:])


def test_status_rejects_workflow_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["status", "workflow"])
    assert "unrecognized arguments: workflow" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--failed", "--fast"],
        ["--finished", "--fast"],
        ["--all", "--fast"],
        ["--all", "--failed"],
        ["--all", "--finished"],
        ["--all", "--all-after", "2026-09-22T12:30:00"],
        ["--all-after", "2026-09-22T12:30:00", "--failed"],
    ],
)
def test_fast_status_is_mutually_exclusive(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["status", *arguments])

    assert "not allowed" in capsys.readouterr().err


def test_finished_status_reports_local_trace_as_failed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "local_failure"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({"pipeline": "dvc.yaml", "backend": "local", "jobs": {}})
    )
    job = dawgz.job(lambda: None, name="job")()
    scheduler = dawgz.schedulers.AsyncScheduler("test")
    scheduler.path = path
    scheduler.uid = "local_failure"
    scheduler.date = datetime.fromisoformat("2026-09-17T12:00:00")
    scheduler.order = {job: 0}
    scheduler.results = {}
    scheduler.traces = {job: "JobSubmissionError: failed"}
    scheduler.dump()
    dawgz.set_dawgz_dir(records)

    main(["status", "--finished"])

    assert capsys.readouterr().out.splitlines()[1].split() == [
        "local_failure",
        "dvc.yaml",
        "2026-09-17T12:00:00",
        "failed",
    ]


def test_finished_status_sorts_workflows_by_submission_time(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    for workflow_id, submitted in (
        ("later", "2026-09-17T14:00:00"),
        ("earlier", "2026-09-17T10:00:00"),
    ):
        path = records / workflow_id
        path.mkdir(parents=True)
        (path / "plan.json").write_text(
            json.dumps({
                "pipeline": "dvc.yaml",
                "backend": "local",
                "jobs": {},
            })
        )
        job = dawgz.job(lambda: None, name="job")()
        scheduler = dawgz.schedulers.AsyncScheduler("test")
        scheduler.path = path
        scheduler.uid = workflow_id
        scheduler.date = datetime.fromisoformat(submitted)
        scheduler.order = {job: 0}
        scheduler.results = {job: None}
        scheduler.traces = {}
        scheduler.dump()
    dawgz.set_dawgz_dir(records)

    main(["status", "--finished"])

    rows = [
        line.split()[0] for line in capsys.readouterr().out.splitlines()[1:]
    ]
    assert rows == ["earlier", "later"]


def test_all_status_lists_slurm_and_local_workflows_with_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    _write_plan(records, "slurm", "slurm/dvc.yaml", {"10": "stage"})
    slurm_job = dawgz.job(lambda: None, name="stage")()
    slurm_scheduler = dawgz.schedulers.AsyncScheduler("test")
    slurm_scheduler.path = records / "slurm"
    slurm_scheduler.uid = "slurm"
    slurm_scheduler.date = datetime.fromisoformat("2026-09-17T10:00:00")
    slurm_scheduler.order = {slurm_job: 0}
    slurm_scheduler.results = {slurm_job: "10"}
    slurm_scheduler.traces = {}
    slurm_scheduler.dump()

    local_path = records / "local"
    local_path.mkdir(parents=True)
    (local_path / "plan.json").write_text(
        json.dumps({
            "pipeline": "local/dvc.yaml",
            "backend": "local",
            "jobs": {},
        })
    )
    local_job = dawgz.job(lambda: None, name="local_job")()
    local_scheduler = dawgz.schedulers.AsyncScheduler("test")
    local_scheduler.path = local_path
    local_scheduler.uid = "local"
    local_scheduler.date = datetime.fromisoformat("2026-09-17T11:00:00")
    local_scheduler.order = {local_job: 0}
    local_scheduler.results = {local_job: None}
    local_scheduler.traces = {}
    local_scheduler.dump()
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="10|RUNNING|0:0\n"
        ),
    )

    main(["status", "--all"])

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == [
        "WORKFLOW",
        "PIPELINE",
        "SUBMISSION",
        "STATUS",
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "UNKNOWN",
    ]
    assert rows[1] == [
        "slurm",
        "slurm/dvc.yaml",
        "2026-09-17T10:00:00",
        "running",
        "0",
        "1",
        "0",
        "0",
        "0",
        "0",
    ]
    assert rows[2] == [
        "local",
        "local/dvc.yaml",
        "2026-09-17T11:00:00",
        "succeeded",
        "0",
        "0",
        "1",
        "0",
        "0",
        "0",
    ]


def test_all_after_is_inclusive_and_keeps_unknown_submission_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = tmp_path / ".dawgz"
    for workflow_id, submitted in (
        ("older", "2026-09-22T12:29:59"),
        ("boundary", "2026-09-22T12:30:00"),
        ("later", "2026-09-22T12:30:01"),
        ("legacy", None),
    ):
        path = records / workflow_id
        path.mkdir(parents=True)
        (path / "plan.json").write_text(
            json.dumps({
                "pipeline": "dvc.yaml",
                "backend": "local",
                "jobs": {},
            })
        )
        job = dawgz.job(lambda: None, name="job")()
        scheduler = dawgz.schedulers.AsyncScheduler("test")
        scheduler.path = path
        scheduler.uid = workflow_id
        scheduler.date = (
            datetime.fromisoformat(submitted) if submitted is not None else None
        )
        scheduler.order = {job: 0}
        scheduler.results = {job: None}
        scheduler.traces = {}
        scheduler.dump()
    dawgz.set_dawgz_dir(records)

    main(["status", "--all-after", "2026-09-22T12:30:00"])

    rows = [
        line.split()[0] for line in capsys.readouterr().out.splitlines()[1:]
    ]
    assert rows == ["boundary", "later", "legacy"]


@pytest.mark.parametrize(
    "value",
    [
        "2026-09-22",
        "2026-9-22T12:30:00",
        "2026-09-22 12:30:00",
        "2026-09-22T12:30",
        "2026-09-22T12:30:00.000000",
        "2026-09-22T12:30:00+02:00",
        "2026-02-30T12:30:00",
    ],
)
def test_all_after_rejects_datetime_outside_exact_format(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["status", "--all-after", value])

    assert "YYYY-MM-DDTHH:MM:SS" in capsys.readouterr().err


def test_all_status_reports_when_no_workflows_are_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dawgz.set_dawgz_dir(tmp_path / ".dawgz")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("queried Slurm without jobs"),
    )

    main(["status", "--all"])

    assert capsys.readouterr().out == "No klej workflows.\n"
