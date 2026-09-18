import pytest
import subprocess
import sys
import threading

from pathlib import Path

from klej import runner
from klej.runner import (
    CLEANUP_CONFIRMATION_GRACE,
    DEFAULT_KILL_WAIT,
    RACE_LOST_EXIT_CODE,
    claim_and_run,
    cleanup_reservation,
    commit_lock,
    lock_documents,
    register_reservation_jobs,
    release_reservation,
    reserve_targets,
)


def _register_cleanup_jobs(
    tmp_path: Path,
    reservation: tuple[str, str],
) -> None:
    register_reservation_jobs(
        tmp_path,
        *reservation,
        [
            {
                "role": "stage",
                "logical_id": "a" * 64,
                "target": "first",
                "name": "first",
                "profile": "compute",
                "slurm_job_id": "20",
            },
            {
                "role": "commit",
                "logical_id": None,
                "target": "first",
                "name": "commit_first",
                "profile": "commit",
                "slurm_job_id": "21",
            },
            {
                "role": "stage",
                "logical_id": "b" * 64,
                "target": "second",
                "name": "second",
                "profile": "compute",
                "slurm_job_id": "22",
            },
            {
                "role": "cleanup",
                "logical_id": None,
                "target": None,
                "name": "cleanup",
                "profile": "control",
                "slurm_job_id": "23",
            },
        ],
    )


def test_registered_winner_cancels_only_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    logical_id = "c" * 64
    register_reservation_jobs(
        tmp_path,
        *reservation,
        [
            {
                "role": "attempt",
                "logical_id": logical_id,
                "target": "stage",
                "name": "a",
                "profile": "small",
                "slurm_job_id": "10",
            },
            {
                "role": "attempt",
                "logical_id": logical_id,
                "target": "stage",
                "name": "b",
                "profile": "large",
                "slurm_job_id": "11",
            },
            {
                "role": "gate",
                "logical_id": logical_id,
                "target": "stage",
                "name": "gate",
                "profile": "control",
                "slurm_job_id": "12",
            },
        ],
    )
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "10")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: (
            calls.append((command, kwargs))
            or subprocess.CompletedProcess(command, 0)
        ),
    )

    assert (
        claim_and_run(
            tmp_path,
            tmp_path / "records",
            reservation[0],
            logical_id,
            "stage",
            "small",
            [sys.executable, "-c", "pass"],
        )
        == 0
    )

    assert calls[0][0] == ["scancel", "11"]
    assert calls[1][0] == [sys.executable, "-c", "pass"]


def test_cleanup_cancels_active_intermediate_commit_then_releases_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    _register_cleanup_jobs(tmp_path, reservation)
    calls = []
    waits = []
    squeue_outputs = iter(["21\n", ""])
    monkeypatch.setenv("SLURM_JOB_ID", "23")

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[0] == "squeue":
            output = next(squeue_outputs)
        elif command[:3] == ["scontrol", "show", "config"]:
            output = "  KillWait = 90 sec\n"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(runner.time, "sleep", waits.append)

    cleanup_reservation(tmp_path, *reservation)

    assert calls == [
        [
            "squeue",
            "--noheader",
            "--jobs",
            "20,21,22",
            "--format=%A",
        ],
        ["scancel", "21"],
        ["scontrol", "show", "config"],
        [
            "squeue",
            "--noheader",
            "--jobs",
            "20,21,22",
            "--format=%A",
        ],
    ]
    assert waits == [95]
    assert lock_documents(tmp_path) == {}


@pytest.mark.parametrize(
    "failure",
    ["missing_command", "command_failure", "missing_value", "malformed_value"],
)
def test_slurm_kill_wait_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if failure == "missing_command":
            raise FileNotFoundError
        if failure == "command_failure":
            raise subprocess.CalledProcessError(1, command)
        output = (
            "CompleteWait = 0 sec\n"
            if failure == "missing_value"
            else "KillWait = later sec\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(subprocess, "run", run)

    assert runner._slurm_kill_wait() == DEFAULT_KILL_WAIT
    assert "using default KillWait of 30 seconds" in capsys.readouterr().err


def test_cleanup_does_not_query_kill_wait_when_no_jobs_are_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    _register_cleanup_jobs(tmp_path, reservation)
    calls = []
    waits = []
    monkeypatch.setenv("SLURM_JOB_ID", "23")

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(runner.time, "sleep", waits.append)

    cleanup_reservation(tmp_path, *reservation)

    assert [command[0] for command in calls] == ["squeue"]
    assert waits == []
    assert lock_documents(tmp_path) == {}


def test_cleanup_fallback_wait_includes_confirmation_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    _register_cleanup_jobs(tmp_path, reservation)
    waits = []
    queries = iter(["21\n", ""])
    monkeypatch.setenv("SLURM_JOB_ID", "23")

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "squeue":
            return subprocess.CompletedProcess(command, 0, stdout=next(queries))
        if command[0] == "scontrol":
            raise FileNotFoundError
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(runner.time, "sleep", waits.append)

    cleanup_reservation(tmp_path, *reservation)

    assert waits == [DEFAULT_KILL_WAIT + CLEANUP_CONFIRMATION_GRACE]


@pytest.mark.parametrize(
    ("failure", "operation", "job_ids"),
    [
        ("query", "query", ["20", "21", "22"]),
        ("cancel", "cancel", ["21"]),
        ("confirm_query", "confirm", ["20", "21", "22"]),
        ("still_active", "confirm", ["21"]),
    ],
)
def test_cleanup_failure_retains_reservation_and_records_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    operation: str,
    job_ids: list[str],
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    _register_cleanup_jobs(tmp_path, reservation)
    query = 0
    monkeypatch.setenv("SLURM_JOB_ID", "23")
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal query
        if command[0] == "squeue":
            query += 1
            if failure == "query" and query == 1:
                raise subprocess.CalledProcessError(1, command)
            if failure == "confirm_query" and query == 2:
                raise subprocess.CalledProcessError(1, command)
            output = "21\n" if query == 1 or failure == "still_active" else ""
            return subprocess.CompletedProcess(command, 0, stdout=output)
        if failure == "cancel":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError, match="reservation retained"):
        cleanup_reservation(tmp_path, *reservation)

    owner = lock_documents(tmp_path)[f"run:{reservation[0]}"]
    assert owner["cleanup_failure"]["operation"] == operation
    assert owner["cleanup_failure"]["job_ids"] == job_ids
    assert owner["cleanup_failure"]["recorded_at"]


def test_cleanup_can_be_retried_after_a_query_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    _register_cleanup_jobs(tmp_path, reservation)
    monkeypatch.setenv("SLURM_JOB_ID", "23")
    attempts = 0

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(RuntimeError):
        cleanup_reservation(tmp_path, *reservation)
    cleanup_reservation(tmp_path, *reservation)

    assert lock_documents(tmp_path) == {}


def test_concurrent_race_claim_invokes_payload_once(tmp_path: Path) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    logical_id = "a" * 64
    output = tmp_path / "payload.txt"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('run')",
    ]
    barrier = threading.Barrier(3)
    results: list[int] = []

    def contender(profile: str) -> None:
        barrier.wait()
        results.append(
            claim_and_run(
                tmp_path,
                tmp_path / "records",
                reservation[0],
                logical_id,
                "dvc.yaml:stage",
                profile,
                command,
            )
        )

    threads = [
        threading.Thread(target=contender, args=(profile,))
        for profile in ("small", "large")
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(results) == [0, RACE_LOST_EXIT_CODE]
    assert output.read_text() == "run"


def test_failed_winner_does_not_allow_failover(tmp_path: Path) -> None:
    reservation = reserve_targets(tmp_path, ["stage"])
    logical_id = "b" * 64
    command = [sys.executable, "-c", "raise SystemExit(23)"]

    winner = claim_and_run(
        tmp_path,
        tmp_path / "records",
        reservation[0],
        logical_id,
        "dvc.yaml:stage",
        "small",
        command,
    )
    loser = claim_and_run(
        tmp_path,
        tmp_path / "records",
        reservation[0],
        logical_id,
        "dvc.yaml:stage",
        "large",
        command,
    )

    assert winner == 23
    assert loser == RACE_LOST_EXIT_CODE


def test_overlapping_reservation_preserves_original(tmp_path: Path) -> None:
    original = reserve_targets(tmp_path, ["shared"])

    with pytest.raises(
        RuntimeError, match="DVC target 'shared' is already reserved"
    ):
        reserve_targets(tmp_path, ["shared", "unused"])

    assert list(lock_documents(tmp_path)) == [f"run:{original[0]}"]
    release_reservation(tmp_path, *original)
    assert lock_documents(tmp_path) == {}


def test_disjoint_reservations_are_released_independently(
    tmp_path: Path,
) -> None:
    first = reserve_targets(tmp_path, ["first"])
    second = reserve_targets(tmp_path, ["second"])

    assert set(lock_documents(tmp_path)) == {
        f"run:{first[0]}",
        f"run:{second[0]}",
    }

    release_reservation(tmp_path, *first)
    assert list(lock_documents(tmp_path)) == [f"run:{second[0]}"]
    release_reservation(tmp_path, *second)
    assert lock_documents(tmp_path) == {}


def test_partial_reservation_failure_rolls_back_only_new_locks(
    tmp_path: Path,
) -> None:
    original = reserve_targets(tmp_path, ["z-occupied"])

    with pytest.raises(RuntimeError, match="z-occupied"):
        reserve_targets(tmp_path, ["a-free", "z-occupied"])

    documents = lock_documents(tmp_path)
    assert list(documents) == [f"run:{original[0]}"]
    assert documents[f"run:{original[0]}"]["targets"] == ["z-occupied"]
    release_reservation(tmp_path, *original)
    assert lock_documents(tmp_path) == {}


def test_commit_lock_excludes_a_second_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retrying = threading.Event()
    retry = threading.Event()
    entered = threading.Event()
    errors: list[Exception] = []

    def wait_for_retry(_seconds: float) -> None:
        retrying.set()
        if not retry.wait(2):
            raise TimeoutError("test did not permit the commit-lock retry")

    monkeypatch.setattr(runner.time, "sleep", wait_for_retry)

    def enter_lock() -> None:
        try:
            with commit_lock(tmp_path, "second"):
                entered.set()
        except Exception as error:  # noqa: BLE001 - report worker failures
            errors.append(error)

    thread = threading.Thread(target=enter_lock)
    with commit_lock(tmp_path, "first"):
        thread.start()
        assert retrying.wait(2), "second thread did not encounter the lock"
        assert not entered.is_set()

    retry.set()
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert entered.is_set()
    assert lock_documents(tmp_path) == {}
