"""Standalone compute-side DVC runner. Keep imports package-independent."""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

RACE_LOST_EXIT_CODE = 75
DEFAULT_KILL_WAIT = 30
CLEANUP_CONFIRMATION_GRACE = 5


def _slurm_job_id(value: object) -> str:
    job_id = str(value).strip()
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError(f"Invalid Slurm job ID {value!r}")
    return job_id


def _warning(message: str) -> None:
    print(f"klej: warning: {message}", file=sys.stderr, flush=True)


def cancel_slurm_jobs(job_ids: list[str]) -> list[str]:
    failed = []
    for value in dict.fromkeys(job_ids):
        try:
            job_id = _slurm_job_id(value)
            subprocess.run(
                ["scancel", job_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except (ValueError, FileNotFoundError, subprocess.CalledProcessError):
            failed.append(str(value))
    return failed


def _slurm_kill_wait() -> int:
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None
    if result is not None:
        match = re.search(
            r"(?m)^\s*KillWait\s*=\s*([0-9]+)\s+sec\s*$",
            result.stdout,
        )
        if match is not None:
            return int(match.group(1))
    _warning(
        f"could not read Slurm KillWait; using default KillWait of "
        f"{DEFAULT_KILL_WAIT} seconds"
    )
    return DEFAULT_KILL_WAIT


def state_root(repo_root: Path) -> Path:
    return repo_root / ".klej" / "locks"


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def claim_and_run(
    execution_root: Path,
    record_root: Path,
    run_id: str,
    logical_id: str,
    target: str,
    profile: str,
    command: list[str],
) -> int:
    if not re.fullmatch(r"[0-9a-f]{64}", logical_id):
        raise ValueError(f"Invalid logical stage ID {logical_id!r}")
    if not re.fullmatch(r"[0-9a-f]+", run_id):
        raise ValueError(f"Invalid reservation ID {run_id!r}")
    race = state_root(execution_root) / "runs" / run_id / "races" / logical_id
    race.mkdir(parents=True, exist_ok=True)
    claim = race / "claim"
    job_id = os.environ.get("SLURM_JOB_ID")
    started = datetime.now(UTC).isoformat()
    try:
        claim.mkdir()
    except FileExistsError:
        return RACE_LOST_EXIT_CODE
    metadata = {
        "logical_id": logical_id,
        "target": target,
        "profile": profile,
        "slurm_job_id": job_id,
        "claiming_job_id": job_id,
        "start_time": started,
        "outcome": "running",
    }
    _atomic_json(claim / "owner.json", metadata)
    record = record_root / "races" / f"{logical_id}.json"
    _atomic_json(record, metadata)
    if job_id is not None:
        cancel_registered_siblings(execution_root, run_id, logical_id, job_id)
    result = subprocess.run(command, check=False)
    metadata.update({
        "finish_time": datetime.now(UTC).isoformat(),
        "payload_exit_status": result.returncode,
        "outcome": "succeeded" if result.returncode == 0 else "failed",
    })
    _atomic_json(claim / "owner.json", metadata)
    _atomic_json(record, metadata)
    return result.returncode


def _owner(kind: str, token: str, **values: object) -> dict:
    return {
        "kind": kind,
        "token": token,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "acquired_at": datetime.now(UTC).isoformat(),
        **values,
    }


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _remove_owned(path: Path, token: str) -> bool:
    owner = _read_json(path / "owner.json")
    if owner is None or owner.get("token") != token:
        return False
    (path / "owner.json").unlink(missing_ok=True)
    try:
        path.rmdir()
    except FileNotFoundError:
        return False
    return True


def _force_remove(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            raise RuntimeError(
                f"Refusing to remove unexpected directory {child}"
            )
        child.unlink()
    path.rmdir()


def validate_dvc_lock(repo_root: Path) -> None:
    path = repo_root / ".dvc" / "tmp" / "rwlock"
    if path.exists():
        try:
            json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"DVC lock metadata is corrupted at {path}"
            ) from error


@contextmanager
def commit_lock(repo_root: Path, target: str) -> Iterator[None]:
    root = state_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "commit.lock"
    token = uuid.uuid4().hex
    owner = _owner("commit", token, target=target)
    while True:
        try:
            path.mkdir()
        except FileExistsError:
            time.sleep(random.uniform(1.0, 3.0))
        else:
            _atomic_json(path / "owner.json", owner)
            break
    handlers = {}
    if threading.current_thread() is threading.main_thread():

        def interrupt(signum: int, _frame: object) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupt)
    try:
        validate_dvc_lock(repo_root)
        yield
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        _remove_owned(path, token)


def reserve_targets(repo_root: Path, targets: list[str]) -> tuple[str, str]:
    root = state_root(repo_root)
    target_root, run_root = root / "targets", root / "runs"
    target_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    run_id, token = uuid.uuid4().hex, uuid.uuid4().hex
    run_path = run_root / run_id
    run_path.mkdir()
    acquired = []
    try:
        for target in sorted(targets):
            path = (
                target_root
                / f"{hashlib.sha256(target.encode()).hexdigest()}.lock"
            )
            try:
                path.mkdir()
            except FileExistsError as error:
                raise RuntimeError(
                    f"DVC target {target!r} is already reserved"
                ) from error
            _atomic_json(
                path / "owner.json",
                _owner("target", token, target=target, reservation_id=run_id),
            )
            acquired.append(path)
        _atomic_json(
            run_path / "owner.json",
            _owner(
                "run",
                token,
                reservation_id=run_id,
                targets=sorted(targets),
                slurm_job_ids=[],
            ),
        )
    except BaseException:
        for path in reversed(acquired):
            _remove_owned(path, token)
        (run_path / "owner.json").unlink(missing_ok=True)
        run_path.rmdir()
        raise
    return run_id, token


def update_reservation_jobs(
    repo_root: Path, run_id: str, token: str, job_ids: list[str]
) -> None:
    path = state_root(repo_root) / "runs" / run_id / "owner.json"
    owner = _read_json(path)
    if owner is not None and owner.get("token") == token:
        owner["slurm_job_ids"] = list(dict.fromkeys(job_ids))
        _atomic_json(path, owner)


def register_reservation_jobs(
    repo_root: Path,
    run_id: str,
    token: str,
    physical_jobs: list[dict],
) -> None:
    path = state_root(repo_root) / "runs" / run_id / "owner.json"
    owner = _read_json(path)
    if owner is None or owner.get("token") != token:
        raise RuntimeError(
            f"Reservation {run_id!r} is missing or ownership changed"
        )
    submitted = [job for job in physical_jobs if job.get("slurm_job_id")]
    job_ids = [_slurm_job_id(job["slurm_job_id"]) for job in submitted]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Submitted physical jobs contain duplicate Slurm IDs")
    races: dict[str, dict] = {}
    for job in submitted:
        logical_id = job.get("logical_id")
        if job.get("role") not in {"attempt", "gate"} or not logical_id:
            continue
        race = races.setdefault(
            str(logical_id),
            {"target": job.get("target"), "attempts": [], "gate_job_id": None},
        )
        if job["role"] == "attempt":
            race["attempts"].append({
                "name": job.get("name"),
                "profile": job.get("profile"),
                "job_id": _slurm_job_id(job["slurm_job_id"]),
            })
        else:
            race["gate_job_id"] = _slurm_job_id(job["slurm_job_id"])
    for logical_id, race in races.items():
        if len(race["attempts"]) < 2 or race["gate_job_id"] is None:
            raise ValueError(f"Incomplete submitted race {logical_id}")
    cleanup_ids = [
        _slurm_job_id(job["slurm_job_id"])
        for job in submitted
        if job.get("role") == "cleanup"
    ]
    owner.update({
        "slurm_job_ids": job_ids,
        "races": races,
        "cleanup_job_id": cleanup_ids[0] if len(cleanup_ids) == 1 else None,
    })
    _atomic_json(path, owner)


def cancel_registered_siblings(
    repo_root: Path, run_id: str, logical_id: str, current_job_id: str
) -> None:
    try:
        current = _slurm_job_id(current_job_id)
    except ValueError as error:
        _warning(str(error))
        return
    owner = _read_json(state_root(repo_root) / "runs" / run_id / "owner.json")
    race = (owner or {}).get("races", {}).get(logical_id)
    if not isinstance(race, dict):
        _warning(
            f"race {logical_id} is not registered; siblings were not cancelled"
        )
        return
    siblings = [
        str(attempt.get("job_id"))
        for attempt in race.get("attempts", [])
        if str(attempt.get("job_id")) != current
    ]
    failed = cancel_slurm_jobs(siblings)
    if failed:
        _warning(f"could not cancel sibling Slurm jobs: {', '.join(failed)}")


def cleanup_reservation(repo_root: Path, run_id: str, token: str) -> None:
    path = state_root(repo_root) / "runs" / run_id / "owner.json"
    owner = _read_json(path)
    if owner is None or owner.get("token") != token:
        raise RuntimeError(
            f"Reservation {run_id!r} is missing or ownership changed"
        )
    try:
        registered = [
            _slurm_job_id(value) for value in owner.get("slurm_job_ids", [])
        ]
    except ValueError as error:
        _cleanup_failure(path, token, "query", [], str(error))
    current = os.environ.get("SLURM_JOB_ID")
    job_ids = [job_id for job_id in registered if job_id != current]
    if not job_ids:
        release_reservation(repo_root, run_id, token)
        return
    active = _active_jobs(job_ids)
    if active is None:
        _cleanup_failure(
            path,
            token,
            "query",
            job_ids,
            "cleanup could not query active Slurm jobs",
        )
    cancellable = [job_id for job_id in job_ids if job_id in active]
    failed = cancel_slurm_jobs(cancellable)
    if failed:
        _cleanup_failure(
            path,
            token,
            "cancel",
            failed,
            "cleanup could not cancel Slurm jobs",
        )
    if cancellable:
        time.sleep(_slurm_kill_wait() + CLEANUP_CONFIRMATION_GRACE)
        remaining = _active_jobs(job_ids)
        if remaining is None:
            _cleanup_failure(
                path,
                token,
                "confirm",
                job_ids,
                "cleanup could not confirm Slurm jobs stopped",
            )
        if remaining:
            active_ids = [job_id for job_id in job_ids if job_id in remaining]
            _cleanup_failure(
                path,
                token,
                "confirm",
                active_ids,
                "cleanup found Slurm jobs still active",
            )
    release_reservation(repo_root, run_id, token)


def _cleanup_failure(
    path: Path,
    token: str,
    operation: str,
    job_ids: list[str],
    message: str,
) -> None:
    owner = _read_json(path)
    if owner is not None and owner.get("token") == token:
        owner["cleanup_failure"] = {
            "operation": operation,
            "job_ids": job_ids,
            "message": message,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        _atomic_json(path, owner)
    detail = f"{message}: {', '.join(job_ids)}" if job_ids else message
    raise RuntimeError(f"{detail}; reservation retained")


def release_reservation(repo_root: Path, run_id: str, token: str) -> None:
    root = state_root(repo_root)
    run_path = root / "runs" / run_id
    owner = _read_json(run_path / "owner.json")
    if owner is None or owner.get("token") != token:
        return
    for target in owner.get("targets", []):
        key = hashlib.sha256(str(target).encode()).hexdigest()
        _remove_owned(root / "targets" / f"{key}.lock", token)
    races = run_path / "races"
    if races.is_dir():
        for race in races.iterdir():
            claim = race / "claim"
            if claim.is_dir():
                (claim / "owner.json").unlink(missing_ok=True)
                claim.rmdir()
            race.rmdir()
        races.rmdir()
    _remove_owned(run_path, token)


def lock_documents(repo_root: Path) -> dict[str, dict]:
    root = state_root(repo_root)
    documents = {}
    commit = root / "commit.lock"
    if commit.is_dir():
        documents["commit"] = _read_json(commit / "owner.json") or {
            "metadata_error": "owner.json is missing or malformed"
        }
    runs = root / "runs"
    if runs.is_dir():
        for path in sorted(runs.iterdir()):
            if path.is_dir():
                documents[f"run:{path.name}"] = _read_json(
                    path / "owner.json"
                ) or {"metadata_error": "owner.json is missing or malformed"}
    known = {
        key.removeprefix("run:") for key in documents if key.startswith("run:")
    }
    targets = root / "targets"
    if targets.is_dir():
        for path in sorted(targets.iterdir()):
            owner = _read_json(path / "owner.json")
            if path.is_dir() and (
                owner is None or owner.get("reservation_id") not in known
            ):
                documents[f"target:{path.stem}"] = owner or {
                    "metadata_error": "owner.json is missing or malformed"
                }
    return documents


def _active_jobs(job_ids: list[str]) -> set[str] | None:
    if not job_ids:
        return None
    try:
        result = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(job_ids),
                "--format=%A",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def clear_lock(repo_root: Path, lock_id: str, force: bool = False) -> None:
    documents = lock_documents(repo_root)
    if lock_id not in documents:
        raise ValueError(f"Unknown lock {lock_id!r}")
    owner = documents[lock_id]
    job_ids = owner.get("slurm_job_ids") or [owner.get("slurm_job_id")]
    active = _active_jobs([str(value) for value in job_ids if value])
    if active:
        raise RuntimeError(
            f"Refusing to clear {lock_id}: active Slurm jobs {sorted(active)}"
        )
    if active is None and not force:
        raise RuntimeError(
            f"Cannot verify {lock_id}; pass --force after inspection"
        )
    token = owner.get("token")
    if not token:
        if not force:
            raise RuntimeError(
                f"Cannot safely clear malformed lock {lock_id!r}"
            )
        if lock_id == "commit":
            path = state_root(repo_root) / "commit.lock"
        elif lock_id.startswith("run:"):
            path = state_root(repo_root) / "runs" / lock_id.removeprefix("run:")
        else:
            name = f"{lock_id.removeprefix('target:')}.lock"
            path = state_root(repo_root) / "targets" / name
        _force_remove(path)
        return
    if lock_id == "commit":
        path = state_root(repo_root) / "commit.lock"
        if not _remove_owned(path, str(token)):
            raise RuntimeError(
                "Commit lock ownership changed while clearing it"
            )
    elif lock_id.startswith("run:"):
        release_reservation(repo_root, lock_id.removeprefix("run:"), str(token))
    else:
        path = (
            state_root(repo_root)
            / "targets"
            / f"{lock_id.removeprefix('target:')}.lock"
        )
        if not _remove_owned(path, str(token)):
            raise RuntimeError(
                f"Target lock {lock_id} changed while clearing it"
            )


def absolute_target(repo_root: Path, target: str) -> str:
    path, separator, stage = target.rpartition(":")
    return (
        f"{repo_root / path}:{stage}" if separator else str(repo_root / target)
    )


def _verbose(message: str, enabled: bool) -> None:
    if enabled:
        print(f"klej: {message}", file=sys.stderr, flush=True)


def _enable_verbose_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s: %(message)s",
        )
    root.setLevel(logging.DEBUG)
    for handler in root.handlers:
        handler.setLevel(logging.DEBUG)


def run_stage(
    repo_root: Path,
    execution_root: Path,
    target: str,
    *,
    force: bool = False,
    allow_missing: bool = False,
    run_cache: bool = True,
    metadata: str = "deferred",
    no_commit: bool = False,
    verbose: bool = False,
) -> None:
    from dvc.repo import Repo

    repo = None
    try:
        _verbose(f"loading DVC target {target}", verbose)
        with commit_lock(execution_root, f"read:{target}"):
            repo = Repo(str(repo_root))
            stages = repo.stage.collect(absolute_target(repo_root, target))
        if verbose:
            _enable_verbose_logging()
        if len(stages) != 1:
            raise ValueError(
                f"Expected one expanded DVC stage for {target!r}, found {len(stages)}"
            )
        stage = stages[0]
        cache_action = (
            "bypassed by --force"
            if force
            else "bypassed by --no-run-cache"
            if not run_cache
            else "enabled"
        )
        _verbose(f"run-cache lookup for {target}: {cache_action}", verbose)
        with repo.lock:
            reproduced = stage.reproduce(
                force=force,
                allow_missing=allow_missing,
                no_commit=True,
                run_cache=run_cache,
            )
        if reproduced is None:
            _verbose(f"skipped unchanged DVC target {target}", verbose)
            return
        if metadata == "none":
            _verbose(f"leaving {target} without DVC metadata", verbose)
            return
        if metadata == "inline":
            _verbose(f"acquiring metadata lock for {target}", verbose)
            with commit_lock(execution_root, target), repo.lock:
                if no_commit:
                    stage.dump(update_pipeline=False)
                else:
                    repo.stage_cache.save(stage)
                    stage.commit(allow_missing=allow_missing)
                    stage.dump(update_pipeline=False)
            _verbose(f"updated DVC metadata for {target}", verbose)
        elif not no_commit:
            with repo.lock:
                repo.stage_cache.save(stage)
    finally:
        if repo is not None:
            repo.close()


def commit_targets(
    repo_root: Path,
    execution_root: Path,
    targets: list[str],
    *,
    no_commit: bool = False,
    allow_missing: bool = False,
    verbose: bool = False,
) -> None:
    from dvc.repo import Repo

    label = ", ".join(targets)
    _verbose(f"acquiring metadata lock for {label}", verbose)
    with commit_lock(execution_root, label):
        if not no_commit:
            for target in targets:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dvc",
                        "--wait-for-lock",
                        "commit",
                        "--force",
                        target,
                    ],
                    cwd=repo_root,
                    check=True,
                )
        else:
            with Repo(str(repo_root)) as repo, repo.lock:
                for target in targets:
                    stages = repo.stage.collect(
                        absolute_target(repo_root, target)
                    )
                    for stage in stages:
                        stage.save(allow_missing=allow_missing, run_cache=False)
                        stage.dump(update_pipeline=False)
    _verbose(f"updated DVC metadata for {label}", verbose)


def commit_target(repo_root: Path, execution_root: Path, target: str) -> None:
    commit_targets(repo_root, execution_root, [target])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("execution_root", type=Path)
    parser.add_argument(
        "command",
        choices=("stage", "commit", "release", "cleanup", "race", "gate"),
    )
    parser.add_argument("values", nargs="+")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-run-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        _enable_verbose_logging()
    if args.command == "race":
        if len(args.values) != 6:
            parser.error(
                "race requires RUN LOGICAL_ID TARGET PROFILE RECORD COMMAND_JSON"
            )
        command = json.loads(args.values[5])
        if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command
        ):
            parser.error("race command must be a JSON list of strings")
        raise SystemExit(
            claim_and_run(
                args.execution_root,
                Path(args.values[4]),
                args.values[0],
                args.values[1],
                args.values[2],
                args.values[3],
                command,
            )
        )
    if args.command == "gate":
        return
    if args.command == "cleanup":
        cleanup_reservation(args.execution_root, args.values[0], args.values[1])
        return
    if args.command == "stage":
        metadata = (
            "none"
            if args.no_metadata
            else "inline"
            if args.commit
            else "deferred"
        )
        run_stage(
            args.repo_root,
            args.execution_root,
            args.values[0],
            force=args.force,
            allow_missing=args.allow_missing,
            run_cache=not args.no_run_cache,
            metadata=metadata,
            no_commit=args.no_commit,
            verbose=args.verbose,
        )
    elif args.command == "commit":
        commit_targets(
            args.repo_root,
            args.execution_root,
            args.values,
            no_commit=args.no_commit,
            allow_missing=args.allow_missing,
            verbose=args.verbose,
        )
    else:
        release_reservation(args.execution_root, args.values[0], args.values[1])


if __name__ == "__main__":
    main()
