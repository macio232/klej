import dawgz
import json
import pytest
import subprocess

from pathlib import Path

from klej.__main__ import main


def _record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    records = tmp_path / ".dawgz"
    path = records / "workflow"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "pipeline": "dvc.yaml",
            "backend": "slurm",
            "jobs": {"100": "first", "101": "second"},
        })
    )
    first = dawgz.job(lambda: None, name="first")()
    second = dawgz.job(lambda: None, name="second")()
    first.klej_target = "dvc.yaml:simulate@one"
    scheduler = dawgz.schedulers.SlurmScheduler("test")
    scheduler.path = path
    scheduler.uid = "workflow"
    scheduler.order = {first: 0, second: 1}
    scheduler.results = {first: "100", second: "101"}
    scheduler.traces = {}
    scheduler.dump()
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout="100|COMPLETED\n101|RUNNING\n"
        ),
    )
    return path


def _alternative_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: dict[str, object] | None = None,
    execution_root: Path | None = None,
    accounting: str = (
        "10|COMPLETED||0:0\n"
        "11|FAILED||75:0\n"
        "12|COMPLETED||0:0\n"
        "13|COMPLETED||0:0\n"
    ),
    separate_commit: bool = False,
) -> tuple[Path, str]:
    records = tmp_path / ".dawgz"
    path = records / "alternatives"
    path.mkdir(parents=True)
    logical_id = "a" * 64
    physical = [
        ("small_attempt", "attempt", "small", "10"),
        ("large_attempt", "attempt", "large", "11"),
        ("train_gate", "gate", "control", "12"),
    ]
    if separate_commit:
        physical.append(("commit_train", "commit", "control", "13"))
    physical.append((
        "cleanup",
        "cleanup",
        "control",
        "14" if separate_commit else "13",
    ))
    document = {
        "plan_version": 2,
        "pipeline": "dvc.yaml",
        "backend": "slurm",
        "jobs": {job_id: name for name, _, _, job_id in physical},
        "reservation_id": "b" * 32,
        "stages": [
            {
                "id": logical_id,
                "name": "train",
                "target": "dvc.yaml:train",
                "profiles": ["small", "large"],
            }
        ],
        "physical_jobs": [
            {
                "name": name,
                "role": role,
                "logical_id": logical_id
                if role in {"attempt", "gate"}
                else None,
                "target": "dvc.yaml:train" if role != "cleanup" else None,
                "profile": profile,
                "slurm_job_id": job_id,
            }
            for name, role, profile, job_id in physical
        ],
    }
    if execution_root is not None:
        document["execution_root"] = str(execution_root)
    (path / "plan.json").write_text(json.dumps(document))
    jobs = [dawgz.job(lambda: None, name=name)() for name, *_ in physical]
    scheduler = dawgz.schedulers.SlurmScheduler("test")
    scheduler.path = path
    scheduler.uid = "alternatives"
    scheduler.order = {job: index for index, job in enumerate(jobs)}
    scheduler.results = {
        job: physical[index][3] for index, job in enumerate(jobs)
    }
    scheduler.traces = {}
    scheduler.dump()
    if winner is not None:
        race = path / "races"
        race.mkdir()
        (race / f"{logical_id}.json").write_text(json.dumps(winner))
    dawgz.set_dawgz_dir(records)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=accounting
        ),
    )
    return path, logical_id


def test_logs_select_by_index_id_and_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    (path / "0000_first.log").write_text("one\ntwo\nthree\n")

    main(["logs", "workflow", "0", "--tail", "2"])
    assert capsys.readouterr().out == "two\nthree\n"
    main(["logs", "workflow", "100"])
    assert capsys.readouterr().out == "one\ntwo\nthree\n"
    main(["logs", "workflow", "dvc.yaml:simulate@one", "--tail", "0"])
    assert capsys.readouterr().out == ""


def test_logs_reports_not_created_yet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(tmp_path, monkeypatch)

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "1"])

    assert "not been created yet" in capsys.readouterr().err


def test_follow_emits_appended_content_and_stops_at_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    log = path / "0001_second.log"
    log.write_text("existing\n")
    states = iter(("running", "succeeded"))
    monkeypatch.setattr("klej.status._refresh", lambda _job: next(states))
    monkeypatch.setattr(
        "klej.status.time.sleep",
        lambda _interval: log.open("a").write("appended\n"),
    )

    main(["logs", "workflow", "1", "--follow"])

    assert capsys.readouterr().out == "existing\nappended\n"


def test_follow_waits_once_for_delayed_log_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    log = path / "0001_second.log"
    states = iter(("running", "succeeded"))
    monkeypatch.setattr("klej.status._refresh", lambda _job: next(states))
    monkeypatch.setattr(
        "klej.status.time.sleep",
        lambda _interval: log.write_text("late\n"),
    )

    main(["logs", "workflow", "1", "--follow"])

    captured = capsys.readouterr()
    assert captured.out == "late\n"
    assert captured.err.count("Waiting for log") == 1


def test_logs_prints_stored_trace_when_submission_has_no_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    scheduler = dawgz.Scheduler.load(path)
    second = next(job for job in scheduler.order if scheduler.order[job] == 1)
    del scheduler.results[second]
    scheduler.traces[second] = "Traceback\nJobSubmissionError: sbatch rejected"
    scheduler.dump()

    main(["logs", "workflow", "1"])

    assert capsys.readouterr().out == (
        "Traceback\nJobSubmissionError: sbatch rejected\n"
    )


def test_logs_accepts_an_existing_empty_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    (path / "0000_first.log").touch()

    main(["logs", "workflow", "0"])

    assert capsys.readouterr().out == ""


def test_logs_rejects_ambiguous_numeric_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    scheduler = dawgz.Scheduler.load(path)
    first = next(job for job in scheduler.order if scheduler.order[job] == 0)
    scheduler.results[first] = "1"
    scheduler.dump()
    document = json.loads((path / "plan.json").read_text())
    document["jobs"] = {"1": "first", "101": "second"}
    (path / "plan.json").write_text(json.dumps(document))

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "1"])

    assert "matching job indices: 0, 1" in capsys.readouterr().err


def test_logs_rejects_missing_selector_and_negative_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(tmp_path, monkeypatch)

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "missing"])
    assert "Unknown job" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "0", "--tail", "-1"])
    assert "must be non-negative" in capsys.readouterr().err


def test_logs_rejects_ambiguous_legacy_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _record(tmp_path, monkeypatch)
    scheduler = dawgz.Scheduler.load(path)
    for job in scheduler.order:
        job.__dict__.pop("klej_target", None)
        job.name = "same"
    scheduler.dump()

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "same"])

    assert "matching job indices: 0, 1" in capsys.readouterr().err


def test_logs_rejects_terminal_job_without_log_or_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _record(tmp_path, monkeypatch)
    monkeypatch.setattr("klej.status._refresh", lambda _job: "failed")

    with pytest.raises(SystemExit):
        main(["logs", "workflow", "1"])

    assert "no Slurm log or stored submission trace" in capsys.readouterr().err


def test_logs_rejects_local_workflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = tmp_path / ".dawgz"
    path = records / "local"
    path.mkdir(parents=True)
    (path / "plan.json").write_text(
        json.dumps({
            "pipeline": "dvc.yaml",
            "backend": "local",
            "jobs": {"ignored": "job"},
        })
    )
    dawgz.set_dawgz_dir(records)

    with pytest.raises(SystemExit):
        main(["logs", "local", "0"])

    assert "only supported for Slurm" in capsys.readouterr().err


@pytest.mark.parametrize(
    "winner",
    [
        {"claiming_job_id": "10", "outcome": "succeeded"},
        {
            "claiming_job_id": "10",
            "outcome": "failed",
            "payload_exit_status": 23,
        },
    ],
)
def test_logs_logical_alternative_selects_recorded_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    winner: dict[str, object],
) -> None:
    path, _ = _alternative_record(tmp_path, monkeypatch, winner)
    (path / "0000_small_attempt.log").write_text("winner\n")

    main(["logs", "alternatives", "dvc.yaml:train"])
    assert capsys.readouterr().out == "winner\n"
    main(["logs", "alternatives", "0"])
    assert capsys.readouterr().out == "winner\n"


def test_logs_logical_alternative_with_separate_commit_is_unambiguous(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _alternative_record(
        tmp_path,
        monkeypatch,
        {"claiming_job_id": "10", "outcome": "succeeded"},
        accounting=(
            "10|COMPLETED||0:0\n"
            "11|FAILED||75:0\n"
            "12|COMPLETED||0:0\n"
            "13|COMPLETED||0:0\n"
            "14|COMPLETED||0:0\n"
        ),
        separate_commit=True,
    )
    (path / "0000_small_attempt.log").write_text("winner\n")
    (path / "0003_commit_train.log").write_text("commit\n")

    main(["logs", "alternatives", "dvc.yaml:train"])
    assert capsys.readouterr().out == "winner\n"
    main(["logs", "alternatives", "commit_train"])
    assert capsys.readouterr().out == "commit\n"
    main(["logs", "alternatives", "3"])
    assert capsys.readouterr().out == "commit\n"
    main(["logs", "alternatives", "13"])
    assert capsys.readouterr().out == "commit\n"


def test_logs_exact_slurm_id_selects_superseded_attempt_and_internal_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _alternative_record(
        tmp_path,
        monkeypatch,
        {"claiming_job_id": "10", "outcome": "succeeded"},
    )
    (path / "0001_large_attempt.log").write_text("loser\n")
    (path / "0002_train_gate.log").write_text("gate\n")

    main(["logs", "alternatives", "11"])
    assert capsys.readouterr().out == "loser\n"
    main(["logs", "alternatives", "12"])
    assert capsys.readouterr().out == "gate\n"


def test_logs_exact_slurm_id_takes_precedence_over_logical_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _alternative_record(
        tmp_path,
        monkeypatch,
        {"claiming_job_id": "11", "outcome": "succeeded"},
    )
    document = json.loads((path / "plan.json").read_text())
    document["jobs"]["0"] = document["jobs"].pop("10")
    document["physical_jobs"][0]["slurm_job_id"] = "0"
    (path / "plan.json").write_text(json.dumps(document))
    scheduler = dawgz.Scheduler.load(path)
    first = next(job for job, index in scheduler.order.items() if index == 0)
    scheduler.results[first] = "0"
    scheduler.dump()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "0|FAILED||75:0\n11|COMPLETED||0:0\n"
                "12|COMPLETED||0:0\n13|COMPLETED||0:0\n"
            ),
        ),
    )
    (path / "0000_small_attempt.log").write_text("job zero\n")

    main(["logs", "alternatives", "0"])

    assert capsys.readouterr().out == "job zero\n"


def test_logs_unresolved_alternative_lists_attempts_without_using_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _alternative_record(
        tmp_path,
        monkeypatch,
        accounting="10|PENDING||\n11|RUNNING||\n12|PENDING||\n13|PENDING||\n",
    )
    (path / "0001_large_attempt.log").write_text("first log\n")

    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train"])

    error = capsys.readouterr().err
    assert "unresolved" in error
    assert "10 (small, pending)" in error
    assert "11 (large, running)" in error


def test_follow_rejects_terminal_canceled_race_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _alternative_record(
        tmp_path,
        monkeypatch,
        accounting=(
            "10|CANCELLED||0:0\n11|CANCELLED||0:0\n"
            "12|CANCELLED||0:0\n13|COMPLETED||0:0\n"
        ),
    )
    monkeypatch.setattr(
        "klej.status.time.sleep",
        lambda _interval: pytest.fail("terminal race unexpectedly waited"),
    )

    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train", "--follow"])

    error = capsys.readouterr().err
    assert "unresolved" in error
    assert "cancelled" in error


def test_follow_rejects_unknown_race_absent_from_queue_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _alternative_record(tmp_path, monkeypatch, accounting="")
    commands = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        "klej.status.time.sleep",
        lambda _interval: pytest.fail(
            "inactive unknown race unexpectedly waited"
        ),
    )

    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train", "--follow"])

    error = capsys.readouterr().err
    assert "unresolved" in error
    queue_commands = [command for command in commands if command[0] == "squeue"]
    assert queue_commands == [
        [
            "squeue",
            "--noheader",
            "--jobs",
            "10,11",
            "--format=%A",
        ]
    ]


def test_follow_waits_for_unknown_attempt_present_in_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, logical_id = _alternative_record(tmp_path, monkeypatch, accounting="")
    log = path / "0000_small_attempt.log"

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        output = "10\n" if command[0] == "squeue" else ""
        return subprocess.CompletedProcess(command, 0, stdout=output)

    def resolve(_interval: float) -> None:
        race = path / "races"
        race.mkdir(exist_ok=True)
        (race / f"{logical_id}.json").write_text(
            json.dumps({"claiming_job_id": "10", "outcome": "succeeded"})
        )
        log.write_text("winner\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("klej.status.time.sleep", resolve)
    monkeypatch.setattr("klej.status._refresh", lambda _job: "succeeded")

    main(["logs", "alternatives", "dvc.yaml:train", "--follow"])

    captured = capsys.readouterr()
    assert captured.out == "winner\n"
    assert captured.err.count("Waiting for winner") == 1


def test_follow_reports_queue_query_failure_for_unknown_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _alternative_record(tmp_path, monkeypatch, accounting="")

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "squeue":
            raise subprocess.CalledProcessError(
                1, command, stderr="controller unavailable"
            )
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        "klej.status.time.sleep",
        lambda _interval: pytest.fail("failed queue query unexpectedly waited"),
    )

    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train", "--follow"])

    assert (
        "Cannot query Slurm queue: controller unavailable"
        in capsys.readouterr().err
    )


def test_logs_uses_live_claim_when_durable_record_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    execution_root = tmp_path / "repo"
    path, logical_id = _alternative_record(
        tmp_path,
        monkeypatch,
        execution_root=execution_root,
        accounting="",
    )
    claim = (
        execution_root
        / ".klej/locks/runs"
        / ("b" * 32)
        / "races"
        / logical_id
        / "claim"
    )
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(
        json.dumps({"logical_id": logical_id, "claiming_job_id": "10"})
    )
    (path / "0000_small_attempt.log").write_text("live\n")

    main(["logs", "alternatives", "dvc.yaml:train"])

    assert capsys.readouterr().out == "live\n"


def test_logs_rejects_conflicting_durable_and_live_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    execution_root = tmp_path / "repo"
    _, logical_id = _alternative_record(
        tmp_path,
        monkeypatch,
        {"claiming_job_id": "10"},
        execution_root,
    )
    claim = (
        execution_root
        / ".klej/locks/runs"
        / ("b" * 32)
        / "races"
        / logical_id
        / "claim"
    )
    claim.mkdir(parents=True)
    (claim / "owner.json").write_text(
        json.dumps({"logical_id": logical_id, "claiming_job_id": "11"})
    )

    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train"])

    assert "Conflicting winner metadata" in capsys.readouterr().err


def test_logs_infers_only_one_successful_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, _ = _alternative_record(tmp_path, monkeypatch)
    (path / "0000_small_attempt.log").write_text("inferred\n")

    main(["logs", "alternatives", "dvc.yaml:train"])
    assert capsys.readouterr().out == "inferred\n"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "10|COMPLETED||0:0\n11|COMPLETED||0:0\n"
                "12|COMPLETED||0:0\n13|COMPLETED||0:0\n"
            ),
        ),
    )
    with pytest.raises(SystemExit):
        main(["logs", "alternatives", "dvc.yaml:train"])
    assert "ambiguous" in capsys.readouterr().err


def test_follow_waits_for_winner_then_follows_its_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, logical_id = _alternative_record(
        tmp_path,
        monkeypatch,
        accounting="10|RUNNING||\n11|PENDING||\n12|PENDING||\n13|PENDING||\n",
    )
    log = path / "0000_small_attempt.log"

    def resolve(_interval: float) -> None:
        race = path / "races"
        race.mkdir(exist_ok=True)
        (race / f"{logical_id}.json").write_text(
            json.dumps({"claiming_job_id": "10", "outcome": "succeeded"})
        )
        log.write_text("winner\n")

    monkeypatch.setattr("klej.status.time.sleep", resolve)
    monkeypatch.setattr("klej.status._refresh", lambda _job: "succeeded")

    main(["logs", "alternatives", "dvc.yaml:train", "--follow"])

    captured = capsys.readouterr()
    assert captured.out == "winner\n"
    assert captured.err.count("Waiting for winner") == 1
