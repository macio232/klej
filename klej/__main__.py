import argparse
import dawgz
import json
import logging
import os
import re
import subprocess
import sys

from dvc.repo import Repo
from pathlib import Path
from typing import Any

from .config import Config, load_config, merge_config_files
from .runner import (
    cancel_slurm_jobs,
    clear_lock,
    lock_documents,
    register_reservation_jobs,
    release_reservation,
    reserve_targets,
)
from .runner_cache import install_runner
from .status import (
    _is_cancellable,
    _slurm_states,
    clear_finished_workflows,
    clear_workflow,
    print_logs,
    print_status,
)
from .workflow import (
    ReproOptions,
    build_repro_plan,
    create_jobs,
    plan_document,
    print_plan,
    update_plan_jobs,
    write_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DVC pipeline execution on Slurm"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    repro = commands.add_parser("repro")
    repro.add_argument("targets", nargs="*")
    repro.add_argument("--config", type=Path)
    repro.add_argument("--backend", choices=("slurm", "local"), default="slurm")
    repro.add_argument("--dry", "--dry-run", action="store_true")
    repro.add_argument("-f", "--force", action="store_true")
    repro.add_argument("-i", "--interactive", action="store_true")
    repro.add_argument("-s", "--single-item", action="store_true")
    repro.add_argument("-p", "--pipeline", action="store_true")
    repro.add_argument("-P", "--all-pipelines", action="store_true")
    repro.add_argument("-R", "--recursive", action="store_true")
    repro.add_argument("--downstream", action="store_true")
    repro.add_argument("--force-downstream", action="store_true")
    repro.add_argument("--allow-missing", action="store_true")
    repro.add_argument("--glob", action="store_true")
    repro.add_argument("--no-run-cache", action="store_true")
    repro.add_argument("--ignore-errors", action="store_true")
    repro.add_argument("-v", "--verbose", action="store_true")
    repro.add_argument("--local-jobs", type=int, default=1)
    commits = repro.add_mutually_exclusive_group()
    commits.add_argument("--commit-per-stage", action="store_true")
    commits.add_argument("--separate-commit-jobs", action="store_true")
    commits.add_argument("--no-dvc-metadata", action="store_true")
    repro.add_argument("--no-commit", action="store_true")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("workflow_id")
    cancel.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status_selection = status.add_mutually_exclusive_group()
    status_selection.add_argument("workflow_id", nargs="?")
    status_selection.add_argument("--failed", action="store_true")
    status_selection.add_argument("--finished", action="store_true")
    status.add_argument("--physical", action="store_true")
    logs = commands.add_parser("logs")
    logs.add_argument("workflow_id", nargs="?")
    logs.add_argument("job", nargs="?")
    logs.add_argument("--tail", type=_nonnegative_int)
    logs.add_argument("--follow", action="store_true")
    cleanup = logs.add_mutually_exclusive_group()
    cleanup.add_argument("--clear", metavar="WORKFLOW_ID")
    cleanup.add_argument("--clear-all", action="store_true")
    logs.add_argument("--force", action="store_true")
    lock = commands.add_parser("lock")
    lock_commands = lock.add_subparsers(dest="lock_command", required=True)
    lock_commands.add_parser("status")
    clear = lock_commands.add_parser("clear")
    clear.add_argument("lock_id")
    clear.add_argument("--force", action="store_true")
    return parser


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _repo_root(start: Path) -> Path:
    return Path(Repo.find_root(str(start))).resolve()


def _commit_mode(args: argparse.Namespace) -> str:
    if args.commit_per_stage:
        return "stage"
    if args.separate_commit_jobs:
        return "separate"
    if args.no_dvc_metadata:
        return "none"
    return "end"


def _target_path(target: str, invocation_root: Path) -> Path:
    path = target.partition(":")[0]
    candidate = invocation_root / path
    if candidate.is_file():
        return candidate.parent
    if candidate.is_dir():
        return candidate
    return invocation_root


def _normalize_target(
    target: str, invocation_root: Path, repo_root: Path
) -> str:
    path, separator, stage = target.partition(":")
    candidate = invocation_root / path
    path_like = (
        candidate.exists() or "/" in path or path.endswith((".yaml", ".dvc"))
    )
    if separator and path_like:
        return f"{os.path.relpath(candidate.resolve(), repo_root)}:{stage}"
    if path_like:
        return os.path.relpath(candidate.resolve(), repo_root)
    if invocation_root == repo_root:
        return target
    pipeline = os.path.relpath(invocation_root / "dvc.yaml", repo_root)
    return f"{pipeline}:{target}"


def _verbose(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(f"klej: {message}", file=sys.stderr)


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


def _logs(args: argparse.Namespace, root: Path) -> None:
    cleanup = args.clear is not None or args.clear_all
    if cleanup:
        if (
            args.workflow_id is not None
            or args.job is not None
            or args.tail is not None
            or args.follow
        ):
            raise ValueError(
                "Log cleanup cannot be combined with job selection, --tail, or --follow"
            )
        if args.clear is not None:
            clear_workflow(root, args.clear, args.force)
        else:
            clear_finished_workflows(root, args.force)
        return
    if args.force:
        raise ValueError("--force requires --clear or --clear-all")
    if args.workflow_id is None or args.job is None:
        raise ValueError("logs requires WORKFLOW_ID and JOB")
    print_logs(root, args.workflow_id, args.job, args.tail, args.follow)


def _repro(args: argparse.Namespace, invocation_root: Path) -> None:
    if args.verbose:
        _enable_verbose_logging()
    if args.local_jobs < 1:
        raise ValueError("--local-jobs must be positive")
    raw_targets = args.targets or ["dvc.yaml"]
    start = (
        invocation_root
        if args.all_pipelines
        else _target_path(raw_targets[0], invocation_root)
    )
    repo_root = _repo_root(start)
    targets = [
        _normalize_target(target, invocation_root, repo_root)
        for target in raw_targets
    ]
    _verbose(args, f"invocation directory: {invocation_root}")
    _verbose(args, f"DVC repository root: {repo_root}")
    _verbose(args, f"targets: {', '.join(targets)}")

    explicit_path = args.config.resolve() if args.config else None
    explicit = load_config(explicit_path) if explicit_path else None
    root_path = repo_root / "klej.yaml"
    config_cache: dict[str, tuple[Config, str | None]] = {}

    def resolve_config(dvcfile: str) -> tuple[Config, str | None]:
        if explicit is not None:
            return explicit, str(explicit_path)
        if dvcfile in config_cache:
            return config_cache[dvcfile]
        local = (repo_root / dvcfile).parent / "klej.yaml"
        base = root_path if root_path.is_file() else None
        if local.is_file():
            resolved = merge_config_files(base, local)
            path = local
        elif base is not None:
            resolved = load_config(base)
            path = base
        else:
            raise FileNotFoundError(
                f"No klej.yaml found for DVC file {dvcfile!r}"
            )
        config_cache[dvcfile] = resolved, str(path)
        _verbose(args, f"config for {dvcfile}: {path}")
        return config_cache[dvcfile]

    options = ReproOptions(
        force=args.force,
        single_item=args.single_item,
        pipeline=args.pipeline,
        all_pipelines=args.all_pipelines,
        recursive=args.recursive,
        downstream=args.downstream,
        force_downstream=args.force_downstream,
        allow_missing=args.allow_missing,
        glob=args.glob,
    )
    plan = build_repro_plan(
        repo_root,
        targets,
        resolve_config,
        options,
        diagnostic=lambda message: _verbose(args, message),
        configure_logging=_enable_verbose_logging if args.verbose else None,
    )
    for stage in plan.stages:
        condition = "afterany" if args.ignore_errors else "afterok"
        dependencies = ", ".join(stage.dependencies) or "none"
        _verbose(
            args,
            f"stage {stage.target}: profiles={','.join(stage.profiles)}; "
            f"dependencies={dependencies}; condition={condition}",
        )
    configs = [
        stage.config for stage in plan.stages if stage.config is not None
    ]
    fallback_dvcfile = (
        "dvc.yaml" if args.all_pipelines else targets[0].partition(":")[0]
    )
    config = (
        explicit if explicit is not None else configs[0] if configs else None
    )
    klej = {
        (
            tuple(sorted(item.profiles[item.klej].settings.items())),
            item.profiles[item.klej].environment,
            item.environment,
        )
        for item in configs
    }
    if len(klej) > 1 and explicit is None:
        raise ValueError(
            "Selected pipeline configs have different klej settings; "
            "align their klej settings or pass --config"
        )
    mode = _commit_mode(args)
    pipeline = repo_root / (
        plan.stages[0].dvcfile if plan.stages else fallback_dvcfile
    )
    config_path = (
        explicit_path
        if explicit_path is not None
        else Path(plan.stages[0].config_path)
        if plan.stages and plan.stages[0].config_path is not None
        else None
    )
    document = plan_document(
        pipeline, config_path, repo_root, plan, args.backend, mode, config
    )
    document["repro_options"].update({
        name: True
        for name, enabled in {
            "no_commit": args.no_commit,
            "no_run_cache": args.no_run_cache,
            "ignore_errors": args.ignore_errors,
            "verbose": args.verbose,
        }.items()
        if enabled
    })
    if not plan.stages:
        print_plan(document)
        print("\nPipeline is up to date; nothing to schedule.")
        return
    assert config is not None
    with Repo(str(repo_root)) as dvc_repo:
        interactive = args.interactive or bool(
            dvc_repo.config["core"].get("interactive", False)
        )
    if interactive:
        document["repro_options"]["interactive"] = True
    if interactive and plan.stages:
        for stage in plan.stages:
            answer = input(f"Reproduce {stage.target}? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                raise RuntimeError("reproduction aborted by the user")
    if args.dry:
        print_plan(document)
        return
    runner = install_runner(invocation_root)
    document["runner_hash"] = runner.parent.name
    _verbose(args, f"runner hash: {runner.parent.name}")
    reservation = reserve_targets(
        invocation_root, [stage.target for stage in plan.stages]
    )
    _verbose(args, f"reservation: {reservation[0]}")
    backend = "async" if args.backend == "local" else "slurm"
    scheduler_type = (
        dawgz.schedulers.AsyncScheduler
        if backend == "async"
        else dawgz.schedulers.SlurmScheduler
    )
    scheduler_options = (
        {"max_workers": args.local_jobs} if backend == "async" else {}
    )
    try:
        scheduler = scheduler_type(name=repo_root.name, **scheduler_options)
        leaves, _ = create_jobs(
            repo_root,
            invocation_root,
            runner,
            plan,
            config,
            args.backend,
            mode,
            pipeline,
            reservation,
            force=args.force,
            allow_missing=args.allow_missing,
            run_cache=not args.no_run_cache,
            no_commit=args.no_commit,
            ignore_errors=args.ignore_errors,
            verbose=args.verbose,
            record_root=scheduler.path,
        )
        document["reservation_id"] = reservation[0]
        document["execution_root"] = str(invocation_root)
        document["submission_state"] = "planned"
        update_plan_jobs(document, leaves)
        write_plan(scheduler.path / "plan.json", document)
    except BaseException:
        release_reservation(invocation_root, *reservation)
        raise
    submission_started = False
    registration_complete = False
    try:
        submission_started = True
        scheduler(*leaves)
        scheduler.dump()
        document["jobs"] = {
            str(value): job.name for job, value in scheduler.results.items()
        }
        update_plan_jobs(document, leaves, scheduler.results)
        if backend == "slurm":
            document["submission_errors"] = {
                job.name: trace for job, trace in scheduler.traces.items()
            }
            missing = [
                job["name"]
                for job in document["physical_jobs"]
                if job["slurm_job_id"] is None
            ]
            if scheduler.traces or missing:
                raise RuntimeError(
                    "Slurm submission was incomplete: " + ", ".join(missing)
                )
            for job, value in scheduler.results.items():
                _verbose(args, f"submitted {job.name}: Slurm job {value}")
            write_plan(scheduler.path / "plan.json", document)
            register_reservation_jobs(
                invocation_root,
                *reservation,
                document["physical_jobs"],
            )
            registration_complete = True
            document["submission_state"] = "registered"
            write_plan(scheduler.path / "plan.json", document)
            held = [
                job["slurm_job_id"]
                for job in document["physical_jobs"]
                if job["role"] == "attempt"
            ]
            if held:
                subprocess.run(
                    ["scontrol", "release", *held],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            document["submission_state"] = "released"
        else:
            document["submission_state"] = "released"
        write_plan(scheduler.path / "plan.json", document)
    except BaseException as error:
        if submission_started:
            submitted_ids, cleanup_ids = _submitted_slurm_jobs(
                document, scheduler
            )
            try:
                scheduler.dump()
                document["submission_state"] = "failed"
                document["submission_failure"] = str(error)
                document["jobs"] = {
                    str(value): job.name
                    for job, value in scheduler.results.items()
                }
                update_plan_jobs(document, leaves, scheduler.results)
                write_plan(scheduler.path / "plan.json", document)
            finally:
                cancel_slurm_jobs([
                    job_id
                    for job_id in submitted_ids
                    if not registration_complete or job_id not in cleanup_ids
                ])
        else:
            release_reservation(invocation_root, *reservation)
        raise
    print(f"Workflow: {scheduler.uid}")
    print(f"Record: {scheduler.path}")
    if scheduler.traces:
        raise SystemExit(1)


def _submitted_slurm_jobs(
    document: dict[str, Any], scheduler: Any
) -> tuple[list[str], set[str]]:
    job_ids = []
    cleanup_ids = set()

    def add(value: object) -> str | None:
        if value is None:
            return None
        job_id = str(value)
        if job_id not in job_ids:
            job_ids.append(job_id)
        return job_id

    physical = document.get("physical_jobs")
    if isinstance(physical, list):
        for item in physical:
            if not isinstance(item, dict):
                continue
            job_id = add(item.get("slurm_job_id"))
            if item.get("role") == "cleanup" and job_id is not None:
                cleanup_ids.add(job_id)
    for job, value in scheduler.results.items():
        job_id = add(value)
        if job.name == "klej_reservation_cleanup" and job_id is not None:
            cleanup_ids.add(job_id)
    return job_ids, cleanup_ids


def _cancel(workflow_id: str, dry_run: bool) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", workflow_id):
        raise ValueError(f"Invalid Dawgz workflow ID {workflow_id!r}")
    path = dawgz.get_dawgz_dir() / workflow_id
    scheduler = dawgz.Scheduler.load(path)
    if scheduler.backend != "slurm":
        raise ValueError(f"Workflow {workflow_id!r} does not use Slurm")
    try:
        document = json.loads((path / "plan.json").read_text())
    except FileNotFoundError:
        document = {}
    recorded, cleanup = _submitted_slurm_jobs(document, scheduler)
    job_ids = [value for value in recorded if value not in cleanup]
    invalid = [
        job_id for job_id in job_ids if not re.fullmatch(r"[0-9]+", job_id)
    ]
    if invalid:
        raise ValueError(f"Invalid Slurm job IDs in workflow plan: {invalid}")
    states = _slurm_states(job_ids)
    cancellable = [
        job_id for job_id in job_ids if _is_cancellable(states.get(job_id))
    ]
    print(f"Workflow: {workflow_id}")
    if not cancellable:
        print("No cancellable Slurm jobs.")
        return
    verb = "Would cancel" if dry_run else "Cancelling"
    skipped = len(job_ids) - len(cancellable)
    detail = f" ({skipped} not cancellable)" if skipped else ""
    print(f"{verb} {len(cancellable)} of {len(job_ids)} Slurm jobs{detail}.")
    if dry_run:
        return
    failed = cancel_slurm_jobs(cancellable)
    if failed:
        raise RuntimeError("Could not cancel Slurm jobs: " + ", ".join(failed))


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    invocation_root = Path.cwd().resolve()
    try:
        if args.command == "repro":
            _repro(args, invocation_root)
        elif args.command == "cancel":
            _cancel(args.workflow_id, args.dry_run)
        elif args.command == "status":
            print_status(
                dawgz.get_dawgz_dir(),
                args.workflow_id,
                args.failed,
                args.finished,
                args.physical,
            )
        elif args.command == "logs":
            _logs(args, dawgz.get_dawgz_dir())
        else:
            if args.lock_command == "status":
                documents = lock_documents(invocation_root)
                if not documents:
                    print("No klej locks or workflow reservations.")
                for lock_id, owner in documents.items():
                    print(f"{lock_id}: {json.dumps(owner, sort_keys=True)}")
            else:
                clear_lock(invocation_root, args.lock_id, args.force)
                print(f"Cleared lock {args.lock_id}")
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
