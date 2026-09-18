import csv
import dawgz
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_PENDING = {
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SPECIAL_EXIT",
}
_RUNNING = {
    "COMPLETING",
    "CONFIGURING",
    "EXPEDITING",
    "POWER_UP_NODE",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
    "UPDATE_DB",
}
_FAILED = {
    "BOOT_FAIL",
    "DEADLINE",
    "FAILED",
    "LAUNCH_FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "RECONFIG_FAIL",
    "REVOKED",
    "TIMEOUT",
}
_TERMINAL = {"succeeded", "failed", "cancelled"}
_CATEGORIES = (
    "pending",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "unknown",
)
_SUMMARY_COLUMNS = ("WORKFLOW", "PIPELINE", *map(str.upper, _CATEGORIES))
_SACCT_BATCH_SIZE = 1_000
_FOLLOW_INTERVAL = 1.0
_PHYSICAL_ROLES = {"stage", "attempt", "gate", "commit", "cleanup", "internal"}


@dataclass(frozen=True)
class _Plan:
    workflow: str
    path: Path
    pipeline: str
    backend: str
    jobs: dict[str, str]
    version: int | None = None
    stages: tuple[dict[str, Any], ...] = ()
    physical_jobs: tuple[dict[str, Any], ...] = ()
    reservation_id: str | None = None
    execution_root: Path | None = None


@dataclass(frozen=True)
class _JobStatus:
    index: int
    identity: str
    name: str
    slurm_id: str | None
    start_time: str | None
    state: str
    detail: str
    trace: str | None
    log: Path | None
    role: str = "job"
    profile: str | None = None
    exit_code: str | None = None
    logical_id: str | None = None


@dataclass(frozen=True)
class _WorkflowStatus:
    plan: _Plan
    jobs: tuple[_JobStatus, ...]
    submission: datetime | None = None
    physical_jobs: tuple[_JobStatus, ...] | None = None

    @property
    def counts(self) -> dict[str, int]:
        result = dict.fromkeys(_CATEGORIES, 0)
        for job in self.jobs:
            result[job.state] += 1
        return result

    @property
    def terminal(self) -> bool:
        jobs = (
            self.physical_jobs if self.physical_jobs is not None else self.jobs
        )
        return bool(jobs) and all(job.state in _TERMINAL for job in jobs)

    @property
    def state(self) -> str:
        states = {job.state for job in self.jobs}
        for state in ("running", "pending", "failed", "cancelled"):
            if state in states:
                return state
        if self.jobs and states == {"succeeded"}:
            return "succeeded"
        return "unknown"


def _read_plan(path: Path) -> _Plan:
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed klej workflow plan: {path}") from error
    if not isinstance(document, dict):
        raise TypeError(f"Malformed klej workflow plan: {path}")
    pipeline = document.get("pipeline")
    backend = document.get("backend")
    jobs = document.get("jobs")
    version = document.get("plan_version")
    if (
        version == 2
        and jobs is None
        and document.get("submission_state") == "planned"
    ):
        jobs = {}
    if (
        not isinstance(pipeline, str)
        or not isinstance(backend, str)
        or not isinstance(jobs, dict)
        or not all(
            isinstance(job_id, str) and isinstance(name, str)
            for job_id, name in jobs.items()
        )
    ):
        raise TypeError(f"Malformed klej workflow plan: {path}")
    if version is None:
        return _Plan(path.parent.name, path.parent, pipeline, backend, jobs)
    if version != 2:
        raise ValueError(
            f"Unsupported klej workflow plan version {version!r}: {path}"
        )
    stages = document.get("stages")
    physical = document.get("physical_jobs")
    reservation_id = document.get("reservation_id")
    execution_root = document.get("execution_root")
    if not isinstance(stages, list) or not isinstance(physical, list):
        raise TypeError(f"Malformed version 2 klej workflow plan: {path}")
    if reservation_id is not None and not isinstance(reservation_id, str):
        raise TypeError(f"Malformed version 2 klej workflow plan: {path}")
    if execution_root is not None and not isinstance(execution_root, str):
        raise TypeError(f"Malformed version 2 klej workflow plan: {path}")
    if not all(
        isinstance(stage, dict)
        and isinstance(stage.get("id"), str)
        and isinstance(stage.get("target"), str)
        and isinstance(stage.get("profiles"), list)
        and all(isinstance(profile, str) for profile in stage["profiles"])
        for stage in stages
    ) or not all(
        isinstance(job, dict)
        and isinstance(job.get("name"), str)
        and job.get("role") in _PHYSICAL_ROLES
        and (
            job.get("logical_id") is None
            or isinstance(job.get("logical_id"), str)
        )
        and (job.get("profile") is None or isinstance(job.get("profile"), str))
        and (
            job.get("slurm_job_id") is None
            or isinstance(job.get("slurm_job_id"), str)
            and re.fullmatch(r"[0-9]+", job["slurm_job_id"])
        )
        for job in physical
    ):
        raise TypeError(f"Malformed version 2 klej workflow plan: {path}")
    logical_ids = {stage["id"] for stage in stages}
    if any(
        job.get("logical_id") is not None
        and job.get("logical_id") not in logical_ids
        for job in physical
    ):
        raise ValueError(
            f"Version 2 plan contains an unknown logical stage: {path}"
        )
    return _Plan(
        path.parent.name,
        path.parent,
        pipeline,
        backend,
        jobs,
        version,
        tuple(stages),
        tuple(physical),
        reservation_id,
        Path(execution_root) if execution_root is not None else None,
    )


def _plans(root: Path) -> list[_Plan]:
    return [_read_plan(path) for path in sorted(root.glob("*/plan.json"))]


def _plan(root: Path, workflow_id: str) -> _Plan:
    if not re.fullmatch(r"[A-Za-z0-9_]+", workflow_id):
        raise ValueError(f"Invalid DAWGZ workflow ID {workflow_id!r}")
    path = root / workflow_id / "plan.json"
    if not path.is_file():
        raise ValueError(f"Unknown klej workflow {workflow_id!r}")
    return _read_plan(path)


def _slurm_accounting(
    job_ids: list[str], fields: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    records = {}
    unique = list(dict.fromkeys(job_ids))
    for offset in range(0, len(unique), _SACCT_BATCH_SIZE):
        batch = unique[offset : offset + _SACCT_BATCH_SIZE]
        command = [
            "sacct",
            "-j",
            ",".join(batch),
            "-o",
            ",".join(("JobID", *fields)),
            "-n",
            "-P",
            "-X",
        ]
        try:
            output = subprocess.run(
                command, capture_output=True, check=True, text=True
            ).stdout
        except FileNotFoundError as error:
            raise RuntimeError(
                "Cannot query Slurm status: sacct was not found"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() if error.stderr else str(error)
            raise RuntimeError(
                f"Cannot query Slurm status: {detail}"
            ) from error
        for line in output.splitlines():
            values = line.split("|", len(fields))
            if len(values) < 2 or not values[0]:
                continue
            values.extend("" for _ in range(len(fields) + 1 - len(values)))
            records[values[0]] = tuple(values[1:])
    return records


def _slurm_states(job_ids: list[str]) -> dict[str, str]:
    return {
        job_id: values[0]
        for job_id, values in _slurm_accounting(job_ids, ("State",)).items()
    }


def _slurm_queue(job_ids: list[str]) -> set[str]:
    unique = list(dict.fromkeys(job_ids))
    if not unique:
        return set()
    command = [
        "squeue",
        "--noheader",
        "--jobs",
        ",".join(unique),
        "--format=%A",
    ]
    try:
        output = subprocess.run(
            command, capture_output=True, check=True, text=True
        ).stdout
    except FileNotFoundError as error:
        raise RuntimeError(
            "Cannot query Slurm queue: squeue was not found"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(f"Cannot query Slurm queue: {detail}") from error
    return {line.strip() for line in output.splitlines() if line.strip()}


def _slurm_details(
    job_ids: list[str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    records = _slurm_accounting(job_ids, ("State", "Start", "ExitCode"))
    states = {job_id: values[0] for job_id, values in records.items()}
    start_times = {
        job_id: values[1] for job_id, values in records.items() if values[1]
    }
    exit_codes = {
        job_id: values[2] for job_id, values in records.items() if values[2]
    }
    return states, start_times, exit_codes


def _normalized_state(state: str | None) -> str:
    return (state or "").partition(" ")[0].removesuffix("+")


def _category(state: str | None) -> str:
    normalized = _normalized_state(state)
    if normalized == "COMPLETED":
        return "succeeded"
    if normalized == "CANCELLED":
        return "cancelled"
    if normalized in _RUNNING:
        return "running"
    if normalized in _PENDING:
        return "pending"
    if normalized in _FAILED:
        return "failed"
    return "unknown"


def _trace_category(trace: str) -> str:
    lowered = trace.lower()
    if "jobneversatisfiederror" in lowered or "unsatisfied depend" in lowered:
        return "cancelled"
    return "failed"


def _trace_detail(trace: str) -> str:
    return next(
        (line.strip() for line in reversed(trace.splitlines()) if line.strip()),
        "submission failed",
    )


def _is_cancellable(state: str | None) -> bool:
    normalized = _normalized_state(state)
    return normalized in _PENDING or (
        normalized in _RUNNING and normalized != "COMPLETING"
    )


def _load_scheduler(plan: _Plan) -> Any | None:
    dump = plan.path / "dump.pkl"
    if not dump.is_file():
        return None
    try:
        return dawgz.Scheduler.load(plan.path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot read DAWGZ workflow record: {plan.path}"
        ) from error


def _load_physical_jobs(
    plan: _Plan,
    states: dict[str, str],
    start_times: dict[str, str],
    exit_codes: dict[str, str],
) -> tuple[tuple[_JobStatus, ...], datetime | None]:
    scheduler = _load_scheduler(plan)
    cleanup_failure = _cleanup_failure_detail(plan)
    by_id = {}
    by_name: dict[str, list[tuple[Any, int]]] = {}
    if scheduler is not None:
        for job, index in scheduler.order.items():
            result = scheduler.results.get(job)
            if result is not None:
                by_id[str(result)] = (job, index)
            by_name.setdefault(job.name, []).append((job, index))
    physical = []
    for fallback_index, item in enumerate(plan.physical_jobs):
        slurm_id = item.get("slurm_job_id")
        matched = by_id.get(slurm_id)
        if matched is None and by_name.get(item["name"]):
            matched = by_name[item["name"]].pop(0)
        job, index = matched if matched is not None else (None, fallback_index)
        trace = (
            scheduler.traces.get(job) if scheduler is not None and job else None
        )
        if plan.backend == "local":
            if trace is not None:
                state, detail = _trace_category(trace), _trace_detail(trace)
            elif scheduler is not None and job in scheduler.results:
                state, detail = "succeeded", "COMPLETED"
            else:
                state, detail = "unknown", "-"
        elif slurm_id is not None:
            raw = states.get(slurm_id)
            state = _category(raw)
            detail = raw or "-"
        elif trace is not None:
            state, detail = _trace_category(trace), _trace_detail(trace)
        else:
            state, detail = "unknown", "-"
        if item["role"] == "cleanup" and cleanup_failure is not None:
            detail = f"{detail}; {cleanup_failure}"
        identity = item["name"]
        if item["role"] in {"stage", "attempt"} and item.get("target"):
            identity = item["target"]
        physical.append(
            _JobStatus(
                index,
                identity,
                item["name"],
                slurm_id,
                start_times.get(slurm_id) if slurm_id else None,
                state,
                detail,
                trace,
                plan.path / f"{scheduler.tag(job)}.log"
                if scheduler is not None and job is not None
                else None,
                item["role"],
                item.get("profile"),
                exit_codes.get(slurm_id) if slurm_id else None,
                item.get("logical_id"),
            )
        )
    submission = getattr(scheduler, "date", None)
    return tuple(physical), submission if isinstance(
        submission, datetime
    ) else None


def _cleanup_failure_detail(plan: _Plan) -> str | None:
    if plan.execution_root is None or plan.reservation_id is None:
        return None
    path = (
        plan.execution_root
        / ".klej"
        / "locks"
        / "runs"
        / plan.reservation_id
        / "owner.json"
    )
    try:
        owner = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    failure = owner.get("cleanup_failure") if isinstance(owner, dict) else None
    if not isinstance(failure, dict):
        return None
    operation = failure.get("operation")
    job_ids = failure.get("job_ids")
    if not isinstance(operation, str) or not isinstance(job_ids, list):
        return None
    if not all(isinstance(job_id, str) for job_id in job_ids):
        return None
    jobs = ", ".join(job_ids) if job_ids else "unknown"
    return f"cleanup {operation} failed for Slurm jobs {jobs}"


def _read_winner_record(path: Path, description: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed {description}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Malformed {description}: {path}")
    return value


def _winner_records(
    plan: _Plan, logical_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    durable = _read_winner_record(
        plan.path / "races" / f"{logical_id}.json",
        "durable race winner metadata",
    )
    live = None
    if plan.execution_root is not None and plan.reservation_id is not None:
        live = _read_winner_record(
            plan.execution_root
            / ".klej"
            / "locks"
            / "runs"
            / plan.reservation_id
            / "races"
            / logical_id
            / "claim"
            / "owner.json",
            "live race claim metadata",
        )
    return durable, live


def _winner(
    plan: _Plan,
    stage: dict[str, Any],
    attempts: list[_JobStatus],
) -> tuple[_JobStatus | None, dict[str, Any] | None]:
    durable, live = _winner_records(plan, stage["id"])

    def claimed(
        record: dict[str, Any] | None, description: str
    ) -> _JobStatus | None:
        if record is None:
            return None
        logical_id = record.get("logical_id")
        if logical_id is not None and logical_id != stage["id"]:
            raise ValueError(
                f"{description} does not match logical stage {stage['target']!r}"
            )
        target = record.get("target")
        if target is not None and target != stage["target"]:
            raise ValueError(
                f"{description} does not match target {stage['target']!r}"
            )
        job_id = record.get("claiming_job_id")
        if job_id is None:
            return None
        match = next(
            (job for job in attempts if job.slurm_id == str(job_id)), None
        )
        if match is None:
            raise ValueError(
                f"{description} identifies unknown attempt {job_id!r} "
                f"for {stage['target']!r}"
            )
        profile = record.get("profile")
        if profile is not None and profile != match.profile:
            raise ValueError(
                f"{description} has profile {profile!r}, expected "
                f"{match.profile!r} for attempt {job_id}"
            )
        return match

    durable_winner = claimed(durable, "Durable race winner metadata")
    live_winner = claimed(live, "Live race claim metadata")
    if (
        durable_winner is not None
        and live_winner is not None
        and durable_winner.slurm_id != live_winner.slurm_id
    ):
        raise ValueError(
            f"Conflicting winner metadata for logical stage {stage['target']!r}"
        )
    selected = durable_winner or live_winner
    if selected is not None:
        return selected, durable if durable_winner is not None else live
    inferred = [
        job
        for job in attempts
        if job.state == "succeeded" and job.exit_code in {None, "0:0"}
    ]
    return (inferred[0], None) if len(inferred) == 1 else (None, None)


def _logical_stage(
    plan: _Plan, stage: dict[str, Any], group: list[_JobStatus]
) -> _JobStatus:
    attempts = [job for job in group if job.role == "attempt"]
    gate = next((job for job in group if job.role == "gate"), None)
    if not attempts or gate is None:
        raise ValueError(
            f"Malformed alternative group {stage['target']!r} in {plan.path}"
        )
    winner, record = _winner(plan, stage, attempts)
    outcome = record.get("outcome") if record else None
    payload_exit = record.get("payload_exit_status") if record else None
    winner_failed = winner is not None and (
        outcome == "failed"
        or isinstance(payload_exit, int)
        and payload_exit != 0
        or winner.state == "failed"
    )
    loser_failures = [
        job
        for job in attempts
        if job is not winner
        and job.state == "failed"
        and job.exit_code != "75:0"
    ]
    if winner_failed:
        state = "failed"
    elif winner is not None and (
        winner.state == "cancelled" or gate.state == "cancelled"
    ):
        state = "cancelled"
    elif (
        winner is not None
        and (outcome == "succeeded" or winner.state == "succeeded")
        and gate.state == "succeeded"
    ):
        state = "succeeded"
    elif gate.state == "failed":
        state = "failed"
    elif any(job.state == "running" for job in [*attempts, gate]):
        state = "running"
    elif any(job.state == "pending" for job in [*attempts, gate]):
        state = "pending"
    elif loser_failures:
        state = "failed"
    elif gate.state == "cancelled" and all(
        job.state in {"cancelled", "failed"} for job in attempts
    ):
        state = "cancelled"
    else:
        state = "unknown"
    profile = winner.profile if winner is not None else None
    detail = (
        f"winner={winner.slurm_id} profile={profile}; gate={gate.detail}"
        if winner is not None
        else f"winner=-; gate={gate.detail}"
    )
    starts = [job.start_time for job in [*attempts, gate] if job.start_time]
    return _JobStatus(
        min(job.index for job in group),
        stage["target"],
        stage.get("name", stage["target"]),
        winner.slurm_id if winner is not None else None,
        winner.start_time
        if winner is not None
        else min(starts)
        if starts
        else None,
        state,
        detail,
        winner.trace if winner is not None else None,
        winner.log if winner is not None else None,
        "logical_stage",
        profile,
        winner.exit_code if winner is not None else None,
        stage["id"],
    )


def _logical_jobs(
    plan: _Plan, physical: tuple[_JobStatus, ...]
) -> tuple[_JobStatus, ...]:
    logical = []
    grouped = {
        stage["id"]: [job for job in physical if job.logical_id == stage["id"]]
        for stage in plan.stages
    }
    consumed: set[int] = set()
    for stage in plan.stages:
        group = grouped[stage["id"]]
        alternatives = [job for job in group if job.role == "attempt"]
        if alternatives:
            logical.append(_logical_stage(plan, stage, group))
            consumed.update(id(job) for job in group)
    logical.extend(job for job in physical if id(job) not in consumed)
    return tuple(sorted(logical, key=lambda job: job.index))


def _load_jobs(
    plan: _Plan,
    states: dict[str, str],
    start_times: dict[str, str] | None = None,
    exit_codes: dict[str, str] | None = None,
) -> tuple[tuple[_JobStatus, ...], datetime | None]:
    start_times = start_times or {}
    exit_codes = exit_codes or {}
    if plan.version == 2:
        physical, submission = _load_physical_jobs(
            plan, states, start_times, exit_codes
        )
        return _logical_jobs(plan, physical), submission
    dump = plan.path / "dump.pkl"
    if not dump.is_file():
        return (
            tuple(
                _JobStatus(
                    index,
                    name,
                    name,
                    job_id,
                    start_times.get(job_id),
                    _category(states.get(job_id)),
                    states.get(job_id, "-"),
                    None,
                    None,
                )
                for index, (job_id, name) in enumerate(plan.jobs.items())
            ),
            None,
        )
    try:
        scheduler = dawgz.Scheduler.load(plan.path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot read DAWGZ workflow record: {plan.path}"
        ) from error
    jobs = []
    for job, index in sorted(scheduler.order.items(), key=lambda item: item[1]):
        result = scheduler.results.get(job)
        trace = scheduler.traces.get(job)
        if plan.backend == "local":
            slurm_id = None
            if trace is not None:
                state = _trace_category(trace)
                detail = _trace_detail(trace)
            elif job in scheduler.results:
                state = "succeeded"
                detail = "COMPLETED"
            else:
                state = "unknown"
                detail = "-"
        elif result is not None:
            slurm_id = str(result)
            raw = states.get(slurm_id)
            state = _category(raw)
            detail = raw if raw is not None else "-"
        elif trace is not None:
            slurm_id = None
            state = _trace_category(trace)
            detail = _trace_detail(trace)
        else:
            slurm_id = None
            state = "unknown"
            detail = "-"
        identity = getattr(job, "klej_target", job.name)
        jobs.append(
            _JobStatus(
                index,
                identity,
                job.name,
                slurm_id,
                start_times.get(slurm_id) if slurm_id is not None else None,
                state,
                detail,
                trace,
                plan.path / f"{scheduler.tag(job)}.log",
            )
        )
    recorded_date = getattr(scheduler, "date", None)
    submission = recorded_date if isinstance(recorded_date, datetime) else None
    return tuple(jobs), submission


def _workflow(
    plan: _Plan,
    states: dict[str, str],
    start_times: dict[str, str] | None = None,
    exit_codes: dict[str, str] | None = None,
) -> _WorkflowStatus:
    jobs, submission = _load_jobs(plan, states, start_times, exit_codes)
    physical = None
    if plan.version == 2:
        physical, _ = _load_physical_jobs(
            plan, states, start_times or {}, exit_codes or {}
        )
    return _WorkflowStatus(plan, jobs, submission, physical)


def _format_table(columns: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    values = [columns, *rows]
    widths = [
        max(len(str(row[index])) for row in values)
        for index in range(len(columns))
    ]
    return "\n".join(
        "  ".join(
            str(value).ljust(width)
            for value, width in zip(row, widths, strict=True)
        )
        for row in values
    ).rstrip()


def _workflows(plans: list[_Plan]) -> list[_WorkflowStatus]:
    job_ids = [
        job_id
        for plan in plans
        if plan.backend == "slurm"
        for job_id in (
            [
                str(job["slurm_job_id"])
                for job in plan.physical_jobs
                if job.get("slurm_job_id")
            ]
            if plan.version == 2
            else list(plan.jobs)
        )
    ]
    records = _slurm_accounting(job_ids, ("State", "Start", "ExitCode"))
    states = {job_id: values[0] for job_id, values in records.items()}
    starts = {
        job_id: values[1] for job_id, values in records.items() if values[1]
    }
    exits = {
        job_id: values[2] for job_id, values in records.items() if values[2]
    }
    return [_workflow(plan, states, starts, exits) for plan in plans]


def _all_workflows(
    root: Path, include_local: bool = False
) -> list[_WorkflowStatus]:
    plans = [
        plan
        for plan in _plans(root)
        if plan.backend == "slurm" or include_local and plan.backend == "local"
    ]
    return _workflows(plans)


def _print_summary(root: Path, failed: bool, finished: bool) -> None:
    rows = []
    workflows = _all_workflows(root, include_local=finished)
    if finished:
        workflows.sort(
            key=lambda workflow: (
                workflow.submission is None,
                workflow.submission.isoformat()
                if workflow.submission is not None
                else "",
                workflow.plan.workflow,
            )
        )
    for workflow in workflows:
        counts = workflow.counts
        include = (
            workflow.terminal
            if finished
            else (
                workflow.terminal
                and bool(counts["failed"] or counts["cancelled"])
                if failed
                else not workflow.terminal
            )
        )
        if include:
            if finished:
                rows.append((
                    workflow.plan.workflow,
                    workflow.plan.pipeline,
                    workflow.submission.isoformat()
                    if workflow.submission is not None
                    else "-",
                    workflow.state,
                ))
            else:
                rows.append((
                    workflow.plan.workflow,
                    workflow.plan.pipeline,
                    *(counts[category] for category in _CATEGORIES),
                ))
    if rows:
        columns = (
            ("WORKFLOW", "PIPELINE", "SUBMISSION", "STATUS")
            if finished
            else _SUMMARY_COLUMNS
        )
        print(_format_table(columns, rows))
    elif finished:
        print("No finished klej workflows.")
    elif failed:
        print("No terminal failed klej workflows.")
    else:
        print("No active klej workflows.")


def _print_detail(root: Path, workflow_id: str, show_physical: bool) -> None:
    plan = _plan(root, workflow_id)
    if plan.backend != "slurm":
        raise ValueError(
            f"Workflow {workflow_id!r} uses the local backend; detailed status is only supported for Slurm workflows"
        )
    job_ids = (
        [
            str(job["slurm_job_id"])
            for job in plan.physical_jobs
            if job.get("slurm_job_id")
        ]
        if plan.version == 2
        else list(plan.jobs)
    )
    states, start_times, exit_codes = _slurm_details(job_ids)
    workflow = _workflow(plan, states, start_times, exit_codes)
    counts = workflow.counts
    print(f"Workflow: {workflow_id}")
    print(f"Pipeline: {plan.pipeline}")
    print(f"State: {workflow.state}")
    print(f"Terminal: {'yes' if workflow.terminal else 'no'}")
    print(
        "Counts: "
        + "  ".join(
            f"{category}={counts[category]}" for category in _CATEGORIES
        )
    )
    rows = [
        (
            job.index,
            job.identity,
            job.slurm_id or "-",
            job.start_time or "-",
            job.state,
            job.detail,
        )
        for job in sorted(
            workflow.jobs,
            key=lambda job: (
                job.start_time is None,
                job.start_time or "",
                job.index,
            ),
        )
    ]
    print()
    print(
        _format_table(
            ("INDEX", "JOB", "SLURM ID", "START TIME", "STATE", "DETAIL"),
            rows,
        )
    )
    if show_physical:
        if workflow.physical_jobs is None:
            print("\nPhysical jobs: unavailable for legacy workflow plans.")
            return
        winners = {}
        for stage in plan.stages:
            attempts = [
                job
                for job in workflow.physical_jobs
                if job.logical_id == stage["id"] and job.role == "attempt"
            ]
            winner, _ = _winner(plan, stage, attempts)
            if winner is not None:
                winners[stage["id"]] = str(winner.slurm_id)
        physical_rows = []
        for job in sorted(workflow.physical_jobs, key=lambda item: item.index):
            display_state = (
                "superseded"
                if job.role == "attempt"
                and job.logical_id in winners
                and job.slurm_id != winners[job.logical_id]
                else job.state
            )
            physical_rows.append((
                job.index,
                job.identity,
                job.role,
                job.profile or "-",
                job.slurm_id or "-",
                job.start_time or "-",
                display_state,
                job.detail,
                job.exit_code or "-",
            ))
        print("\nPhysical jobs:")
        print(
            _format_table(
                (
                    "INDEX",
                    "JOB",
                    "ROLE",
                    "PROFILE",
                    "SLURM ID",
                    "START TIME",
                    "STATE",
                    "DETAIL",
                    "EXIT CODE",
                ),
                physical_rows,
            )
        )


def print_status(
    root: Path,
    workflow_id: str | None = None,
    failed: bool = False,
    finished: bool = False,
    physical: bool = False,
) -> None:
    if physical and workflow_id is None:
        raise ValueError("--physical requires a workflow ID")
    if workflow_id is None:
        _print_summary(root, failed, finished)
    else:
        _print_detail(root, workflow_id, physical)


def _confirm(message: str) -> bool:
    try:
        answer = input(f"{message} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _remove_index_rows(root: Path, workflow_ids: set[str]) -> None:
    record = root / "workflows.csv"
    if not record.exists():
        return
    mode = record.stat().st_mode
    with record.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows = [row for row in rows if len(row) < 2 or row[1] not in workflow_ids]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", dir=root, prefix=".workflows-", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            csv.writer(temporary).writerows(rows)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, record)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _remove_workflows(root: Path, workflows: list[_WorkflowStatus]) -> None:
    workflow_ids = {workflow.plan.workflow for workflow in workflows}
    for workflow in workflows:
        shutil.rmtree(workflow.plan.path)
    _remove_index_rows(root, workflow_ids)


def clear_workflow(root: Path, workflow_id: str, force: bool = False) -> None:
    plan = _plan(root, workflow_id)
    workflow = _workflows([plan])[0]
    if not workflow.terminal:
        raise RuntimeError(
            f"Workflow {workflow_id!r} is not known to be terminal; refusing to remove its records"
        )
    if not force and not _confirm(
        f"Remove workflow {workflow_id!r} and all recorded logs?"
    ):
        print("No workflow records removed.")
        return
    _remove_workflows(root, [workflow])
    print(f"Removed workflow {workflow_id}.")


def clear_finished_workflows(root: Path, force: bool = False) -> None:
    workflows = [
        workflow
        for workflow in _all_workflows(root, include_local=True)
        if workflow.terminal
    ]
    if not workflows:
        print("No finished workflow records to clear.")
        return
    if not force and not _confirm(
        f"Remove {len(workflows)} finished workflow records and all their logs?"
    ):
        print("No workflow records removed.")
        return
    _remove_workflows(root, workflows)
    print(f"Removed {len(workflows)} workflow records.")


def _select_job(workflow: _WorkflowStatus, selector: str) -> _JobStatus:
    matches = {
        job.index: job
        for job in workflow.jobs
        if selector == str(job.index)
        or selector == job.slurm_id
        or selector == job.identity
    }
    if not matches:
        raise ValueError(
            f"Unknown job {selector!r} in workflow {workflow.plan.workflow!r}"
        )
    if len(matches) > 1:
        indices = ", ".join(map(str, sorted(matches)))
        raise ValueError(
            f"Ambiguous job {selector!r}; matching job indices: {indices}"
        )
    return next(iter(matches.values()))


def _log_workflow(plan: _Plan) -> _WorkflowStatus:
    if plan.version != 2:
        return _workflow(plan, _slurm_states(list(plan.jobs)))
    job_ids = [
        str(job["slurm_job_id"])
        for job in plan.physical_jobs
        if job.get("slurm_job_id")
    ]
    states, starts, exits = _slurm_details(job_ids)
    return _workflow(plan, states, starts, exits)


def _select_log_job(
    workflow: _WorkflowStatus, selector: str
) -> tuple[_JobStatus | None, dict[str, Any] | None, list[_JobStatus]]:
    if workflow.plan.version != 2:
        return _select_job(workflow, selector), None, []
    assert workflow.physical_jobs is not None
    physical = [
        job for job in workflow.physical_jobs if job.slurm_id == selector
    ]
    if physical:
        if len(physical) > 1:
            raise ValueError(
                f"Ambiguous Slurm job ID {selector!r} in workflow "
                f"{workflow.plan.workflow!r}"
            )
        return physical[0], None, []
    matches = [
        job
        for job in workflow.jobs
        if selector in {str(job.index), job.identity, job.name}
    ]
    if not matches:
        raise ValueError(
            f"Unknown job {selector!r} in workflow {workflow.plan.workflow!r}"
        )
    if len(matches) > 1:
        indices = ", ".join(str(job.index) for job in matches)
        raise ValueError(
            f"Ambiguous job {selector!r}; matching job indices: {indices}"
        )
    job = matches[0]
    if job.role != "logical_stage" or job.logical_id is None:
        return job, None, []
    stage = next(
        stage for stage in workflow.plan.stages if stage["id"] == job.logical_id
    )
    attempts = [
        physical_job
        for physical_job in workflow.physical_jobs
        if physical_job.logical_id == job.logical_id
        and physical_job.role == "attempt"
    ]
    winner, _ = _winner(workflow.plan, stage, attempts)
    return winner, stage, attempts


def _unresolved_race(
    stage: dict[str, Any], attempts: list[_JobStatus]
) -> RuntimeError:
    successful = [
        job
        for job in attempts
        if job.state == "succeeded" and job.exit_code in {None, "0:0"}
    ]
    condition = "ambiguous" if len(successful) > 1 else "unresolved"
    details = ", ".join(
        f"{job.slurm_id or '-'} ({job.profile or '-'}, {job.state})"
        for job in attempts
    )
    return RuntimeError(
        f"Alternative stage {stage['target']!r} is {condition}; "
        f"attempts: {details}"
    )


def _attempts_can_progress(attempts: list[_JobStatus]) -> bool:
    if any(attempt.state in {"pending", "running"} for attempt in attempts):
        return True
    unknown_ids = [
        attempt.slurm_id
        for attempt in attempts
        if attempt.state == "unknown" and attempt.slurm_id is not None
    ]
    return bool(_slurm_queue(unknown_ids))


def _write(content: bytes) -> None:
    sys.stdout.buffer.write(content)
    sys.stdout.buffer.flush()


def _initial_log(path: Path, tail: int | None) -> int:
    with path.open("rb") as stream:
        if tail is None:
            while content := stream.read(64 * 1024):
                _write(content)
        elif tail:
            lines = deque(stream, maxlen=tail)
            _write(b"".join(lines))
        else:
            stream.seek(0, 2)
        return stream.tell()


def _refresh(job: _JobStatus) -> str:
    if job.slurm_id is None:
        return _trace_category(job.trace) if job.trace else "unknown"
    return _category(_slurm_states([job.slurm_id]).get(job.slurm_id))


def print_logs(
    root: Path,
    workflow_id: str,
    selector: str,
    tail: int | None = None,
    follow: bool = False,
) -> None:
    plan = _plan(root, workflow_id)
    if plan.backend != "slurm":
        raise ValueError(
            f"Workflow {workflow_id!r} uses the local backend; logs are only supported for Slurm workflows"
        )
    workflow = _log_workflow(plan)
    job, stage, attempts = _select_log_job(workflow, selector)
    waited_for_winner = False
    while job is None:
        assert stage is not None
        if not follow or not _attempts_can_progress(attempts):
            raise _unresolved_race(stage, attempts)
        if not waited_for_winner:
            print(
                f"Waiting for winner of alternative stage "
                f"{stage['target']!r}...",
                file=sys.stderr,
            )
            waited_for_winner = True
        time.sleep(_FOLLOW_INTERVAL)
        workflow = _log_workflow(plan)
        job, stage, attempts = _select_log_job(workflow, selector)
    waited = False
    try:
        while job.log is None or not job.log.exists():
            if job.trace:
                trace = job.trace + ("" if job.trace.endswith("\n") else "\n")
                _write(trace.encode(errors="replace"))
                return
            state = _refresh(job)
            if state not in {"pending", "running"}:
                raise RuntimeError(
                    f"Job {job.index} has no Slurm log or stored submission trace"
                )
            if not follow:
                raise RuntimeError(
                    f"Log for job {job.index} has not been created yet"
                )
            if not waited:
                print(
                    f"Waiting for log for job {job.index} to be created...",
                    file=sys.stderr,
                )
                waited = True
            time.sleep(_FOLLOW_INTERVAL)
        position = _initial_log(job.log, tail)
        if not follow:
            return
        with job.log.open("rb") as stream:
            stream.seek(position)
            while True:
                while content := stream.read(64 * 1024):
                    _write(content)
                if _refresh(job) in _TERMINAL:
                    while content := stream.read(64 * 1024):
                        _write(content)
                    return
                time.sleep(_FOLLOW_INTERVAL)
    except KeyboardInterrupt as error:
        raise SystemExit(130) from error
