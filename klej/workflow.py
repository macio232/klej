import dawgz
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid

from collections.abc import Callable
from contextlib import chdir
from dataclasses import dataclass
from dvc.repo import Repo
from dvc.repo.reproduce import collect_stages, get_active_graph, plan_repro
from pathlib import Path
from typing import Any

from .config import Config, Profile


@dataclass(frozen=True)
class StagePlan:
    name: str
    target: str
    command: str
    profiles: tuple[str, ...]
    dependencies: tuple[str, ...]
    dvcfile: str = "dvc.yaml"
    config: Config | None = None
    config_path: str | None = None
    force: bool = False


@dataclass(frozen=True)
class ReproPlan:
    stages: tuple[StagePlan, ...]
    total_stages: int
    external_dependencies: tuple[str, ...]
    targets: tuple[str, ...] = ("dvc.yaml",)
    options: dict[str, bool] | None = None


@dataclass(frozen=True)
class ReproOptions:
    force: bool = False
    single_item: bool = False
    pipeline: bool = False
    all_pipelines: bool = False
    recursive: bool = False
    downstream: bool = False
    force_downstream: bool = False
    allow_missing: bool = False
    glob: bool = False


def _stage_type(name: str) -> str:
    return name.split("@", 1)[0]


def _target(repo_root: Path, stage: Any) -> str:
    path = os.path.relpath(stage.path, repo_root)
    name = getattr(stage, "name", None)
    return f"{path}:{name}" if name is not None else path


def _stage_key(repo_root: Path, stage: Any) -> str:
    name = getattr(stage, "name", None)
    if name is None:
        return os.path.relpath(stage.path, repo_root)
    return _stage_type(name)


def _dvcfile(repo_root: Path, stage: Any) -> str:
    return os.path.relpath(stage.path, repo_root)


def build_repro_plan(
    repo_root: Path,
    targets: list[str],
    resolve_config: Callable[[str], tuple[Config, str | None]],
    options: ReproOptions | None = None,
    *,
    allow_unmapped: bool = False,
    diagnostic: Callable[[str], None] | None = None,
    configure_logging: Callable[[], None] | None = None,
) -> ReproPlan:
    options = options or ReproOptions()
    with chdir(repo_root), Repo(str(repo_root)) as repo:
        if configure_logging is not None:
            configure_logging()
        requested = [] if options.all_pipelines else targets
        stages = (
            collect_stages(
                repo,
                requested or ["dvc.yaml"],
                recursive=options.recursive,
                glob=options.glob,
            )
            if not options.all_pipelines
            else []
        )
        graph = get_active_graph(repo.index.graph)
        if options.single_item and not (
            options.pipeline or options.all_pipelines
        ):
            ordered = stages
            selected_graph = None
        else:
            ordered = plan_repro(
                graph,
                stages,
                pipeline=options.pipeline,
                downstream=(
                    options.downstream
                    and not (options.pipeline or options.all_pipelines)
                ),
            )
            selected_graph = graph

        selected: set[Any] = set()
        forced = dict.fromkeys(ordered, options.force)
        with repo.lock:
            for stage in ordered:
                upstream = (
                    list(graph.successors(stage))
                    if selected_graph is not None
                    else []
                )
                upstream_selected = any(
                    parent in selected for parent in upstream
                )
                changed = False
                if not forced[stage] and not upstream_selected:
                    changed = stage.changed(
                        allow_missing=options.allow_missing,
                        upstream=upstream,
                    )
                should_run = forced[stage] or upstream_selected or changed
                if diagnostic is not None:
                    reason = (
                        "forced"
                        if forced[stage]
                        else "upstream selected"
                        if upstream_selected
                        else "changed"
                        if changed
                        else "unchanged"
                    )
                    action = "selected" if should_run else "skipped"
                    diagnostic(
                        f"{action} {_target(repo_root, stage)}: {reason}"
                    )
                    if changed and configure_logging is not None:
                        diagnostic(
                            f"DVC status {_target(repo_root, stage)}: "
                            + json.dumps(stage.status(), sort_keys=True)
                        )
                if not should_run:
                    continue
                selected.add(stage)
                if options.force_downstream and selected_graph is not None:
                    forced.update(
                        dict.fromkeys(graph.predecessors(stage), True)
                    )

        dependencies = {_target(repo_root, stage): set() for stage in selected}
        if selected_graph is not None:
            for downstream, upstream in graph.edges:
                if downstream in selected and upstream in selected:
                    dependencies[_target(repo_root, downstream)].add(
                        _target(repo_root, upstream)
                    )

        plan = []
        for stage in selected:
            dvcfile = _dvcfile(repo_root, stage)
            config, config_path = resolve_config(dvcfile)
            key = _stage_key(repo_root, stage)
            if key not in config.stages:
                if allow_unmapped:
                    profiles = (next(iter(config.profiles)),)
                else:
                    raise ValueError(
                        f"No Slurm profile mapping for DVC stage {key!r} "
                        f"from {dvcfile}"
                    )
            else:
                profiles = config.stages[key]
            target = _target(repo_root, stage)
            plan.append(
                StagePlan(
                    getattr(stage, "name", dvcfile),
                    target,
                    stage.cmd or "",
                    profiles,
                    tuple(sorted(dependencies[target])),
                    dvcfile,
                    config,
                    config_path,
                    forced[stage],
                )
            )
    option_values = {
        name: value
        for name, value in vars(options).items()
        if isinstance(value, bool) and value
    }
    return ReproPlan(
        tuple(sorted(plan, key=lambda item: item.target)),
        len(ordered),
        (),
        tuple(targets),
        option_values,
    )


def build_plan(
    repo_root: Path,
    pipeline: Path,
    config: Config,
    force: bool = False,
    assume_external: bool = False,
) -> ReproPlan:
    relative_pipeline = os.path.relpath(pipeline, repo_root)
    plan = build_repro_plan(
        repo_root,
        [relative_pipeline],
        lambda _dvcfile: (config, None),
        ReproOptions(force=force),
        allow_unmapped=True,
    )
    selected = {stage.target for stage in plan.stages}
    external = tuple(
        sorted(
            target
            for target in selected
            if target != relative_pipeline
            and not target.startswith(f"{relative_pipeline}:")
        )
    )
    if external:
        if assume_external:
            kept = tuple(
                StagePlan(
                    stage.name,
                    stage.target,
                    stage.command,
                    stage.profiles,
                    tuple(
                        dependency
                        for dependency in stage.dependencies
                        if dependency not in external
                    ),
                    stage.dvcfile,
                    stage.config,
                    stage.config_path,
                    stage.force,
                )
                for stage in plan.stages
                if stage.target not in external
            )
            return ReproPlan(
                kept,
                plan.total_stages - len(external),
                external,
                plan.targets,
                plan.options,
            )
        joined = ", ".join(external)
        raise ValueError(
            f"Selected pipeline depends on stages outside {relative_pipeline}: {joined}. "
            "Pass --assume-external-deps to omit them."
        )
    for stage in plan.stages:
        key = _stage_type(stage.name)
        if key not in config.stages:
            raise ValueError(f"No Slurm profile mapping for DVC stage {key!r}")
    return plan


def _job_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "dvc_stage"


def _logical_id(target: str) -> str:
    return hashlib.sha256(target.encode()).hexdigest()


def _run_preserving_exit(invocation: list[str], cwd: str) -> None:
    result = subprocess.run(invocation, cwd=cwd, check=False)
    raise SystemExit(result.returncode)


def _invocation(
    repo_root: Path,
    execution_root: Path,
    runner: Path,
    arguments: list[str],
    wrapper: tuple[str, ...],
) -> list[str]:
    relative_runner = os.path.relpath(runner, repo_root)
    payload = ["python", relative_runner, ".", str(execution_root), *arguments]
    return (
        [*wrapper, shlex.join(payload)]
        if wrapper
        else [
            sys.executable,
            relative_runner,
            ".",
            str(execution_root),
            *arguments,
        ]
    )


def _job(
    invocation: list[str],
    name: str,
    profile: Profile,
    environment: tuple[str, ...],
    repo_root: Path,
    *,
    preserve_exit: bool = False,
) -> dawgz.Job:
    factory = dawgz.job(
        _run_preserving_exit if preserve_exit else subprocess.run,
        name=name,
        interpreter=sys.executable,
        env=list(environment + profile.environment),
        settings=profile.settings,
    )
    if preserve_exit:
        return factory(invocation, cwd=str(repo_root))
    return factory(invocation, cwd=str(repo_root), check=True)


def _after(
    job: dawgz.Job,
    dependencies: list[dawgz.Job],
    backend: str,
    *,
    status: str = "success",
) -> dawgz.Job:
    if not dependencies:
        return job
    if backend == "slurm":
        job.settings["kill_on_invalid_dep"] = "yes"
    return job.after(*dependencies, status=status)


def _job_closure(leaves: list[dawgz.Job]) -> list[dawgz.Job]:
    ordered: list[dawgz.Job] = []
    seen: set[dawgz.Job] = set()

    def visit(job: dawgz.Job) -> None:
        if job in seen:
            return
        seen.add(job)
        for dependency in job.dependencies:
            visit(dependency)
        ordered.append(job)

    for leaf in leaves:
        visit(leaf)
    return ordered


def _describe_job(
    job: dawgz.Job,
    role: str,
    *,
    target: str | None = None,
    logical_id: str | None = None,
    profile: str | None = None,
) -> dawgz.Job:
    job.klej_role = role
    job.klej_target = target
    job.klej_logical_id = logical_id
    job.klej_profile = profile
    return job


def create_jobs(
    repo_root: Path,
    execution_root: Path,
    runner: Path,
    plan: ReproPlan,
    config: Config,
    backend: str,
    commit_mode: str,
    pipeline: Path,
    reservation: tuple[str, str] | None,
    *,
    force: bool = False,
    allow_missing: bool = False,
    run_cache: bool = True,
    no_commit: bool = False,
    ignore_errors: bool = False,
    verbose: bool = False,
    record_root: Path | None = None,
) -> tuple[list[dawgz.Job], dict[str, dawgz.Job]]:
    compute: dict[str, dawgz.Job] = {}
    commits: dict[str, dawgz.Job] = {}
    attempts: dict[str, list[dawgz.Job]] = {}
    record_root = record_root or execution_root / ".klej" / "records"
    for stage in plan.stages:
        stage_config = stage.config or config
        wrapper = stage_config.wrapper if backend == "slurm" else ()
        arguments = ["stage", stage.target]
        if commit_mode == "stage":
            arguments.append("--commit")
        elif commit_mode == "none":
            arguments.append("--no-metadata")
        if force or stage.force:
            arguments.append("--force")
        if allow_missing:
            arguments.append("--allow-missing")
        if not run_cache:
            arguments.append("--no-run-cache")
        if no_commit:
            arguments.append("--no-commit")
        if verbose:
            arguments.append("--verbose")
        payload = _invocation(
            repo_root, execution_root, runner, arguments, wrapper
        )
        logical_id = _logical_id(stage.target)
        alternatives = backend == "slurm" and len(stage.profiles) > 1
        if alternatives:
            if reservation is None:
                raise ValueError("Alternative Slurm jobs require a reservation")
            run_id, _token = reservation
            attempts[stage.target] = []
            for index, profile in enumerate(stage.profiles):
                name = _job_name(
                    f"{stage.name}_{logical_id[:8]}_attempt_{index}_{profile}"
                )
                invocation = _invocation(
                    repo_root,
                    execution_root,
                    runner,
                    [
                        "race",
                        run_id,
                        logical_id,
                        stage.target,
                        profile,
                        str(record_root),
                        json.dumps(payload),
                    ],
                    (),
                )
                attempt = _job(
                    invocation,
                    name,
                    stage_config.profiles[profile],
                    stage_config.environment,
                    repo_root,
                    preserve_exit=True,
                )
                attempt.settings["hold"] = True
                attempts[stage.target].append(
                    _describe_job(
                        attempt,
                        "attempt",
                        target=stage.target,
                        logical_id=logical_id,
                        profile=profile,
                    )
                )
            gate = _job(
                _invocation(
                    repo_root, execution_root, runner, ["gate", logical_id], ()
                ),
                _job_name(f"{stage.name}_{logical_id[:8]}_gate"),
                stage_config.profiles[stage_config.klej],
                stage_config.environment,
                repo_root,
            )
            _after(gate, attempts[stage.target], backend).waitfor("any")
            compute[stage.target] = _describe_job(
                gate,
                "gate",
                target=stage.target,
                logical_id=logical_id,
                profile=stage_config.klej,
            )
        else:
            profile = stage.profiles[0]
            job = _job(
                payload,
                _job_name(stage.name),
                stage_config.profiles[profile],
                stage_config.environment,
                repo_root,
            )
            attempts[stage.target] = [job]
            compute[stage.target] = _describe_job(
                job,
                "stage",
                target=stage.target,
                logical_id=logical_id,
                profile=profile,
            )
        if commit_mode == "separate":
            commits[stage.target] = _describe_job(
                _after(
                    _job(
                        _invocation(
                            repo_root,
                            execution_root,
                            runner,
                            [
                                "commit",
                                stage.target,
                                *(["--no-commit"] if no_commit else []),
                                *(["--allow-missing"] if allow_missing else []),
                                *(["--verbose"] if verbose else []),
                            ],
                            wrapper,
                        ),
                        f"commit_{_job_name(stage.name)}",
                        stage_config.profiles[stage_config.commit],
                        stage_config.environment,
                        repo_root,
                    ),
                    [compute[stage.target]],
                    backend,
                ),
                "commit",
                target=stage.target,
                profile=stage_config.commit,
            )
    for stage in plan.stages:
        upstream = commits if commit_mode == "separate" else compute
        dependencies = [upstream[target] for target in stage.dependencies]
        for attempt in attempts[stage.target]:
            _after(
                attempt,
                dependencies,
                backend,
                status="any" if ignore_errors else "success",
            )
    dependency_targets = {
        dependency for stage in plan.stages for dependency in stage.dependencies
    }
    leaves = commits if commit_mode == "separate" else compute
    final = [
        leaves[target] for target in sorted(set(leaves) - dependency_targets)
    ]
    if commit_mode == "end":
        commit_jobs = []
        dvcfiles = sorted({stage.dvcfile for stage in plan.stages})
        for index, dvcfile in enumerate(dvcfiles):
            selected = [
                stage for stage in plan.stages if stage.dvcfile == dvcfile
            ]
            commit_config = selected[0].config or config
            wrapper = commit_config.wrapper if backend == "slurm" else ()
            arguments = ["commit", *(stage.target for stage in selected)]
            if no_commit:
                arguments.append("--no-commit")
            if allow_missing:
                arguments.append("--allow-missing")
            if verbose:
                arguments.append("--verbose")
            name = "dvc_commit" if len(dvcfiles) == 1 else f"dvc_commit_{index}"
            commit_jobs.append(
                _describe_job(
                    _after(
                        _job(
                            _invocation(
                                repo_root,
                                execution_root,
                                runner,
                                arguments,
                                wrapper,
                            ),
                            name,
                            commit_config.profiles[commit_config.commit],
                            commit_config.environment,
                            repo_root,
                        ),
                        final,
                        backend,
                    ),
                    "commit",
                    profile=commit_config.commit,
                )
            )
        final = commit_jobs
    if reservation is not None:
        run_id, token = reservation
        cleanup = _job(
            _invocation(
                repo_root,
                execution_root,
                runner,
                ["cleanup", run_id, token],
                (),
            ),
            "klej_reservation_cleanup",
            config.profiles[config.klej],
            config.environment,
            repo_root,
        )
        cleanup_dependencies = _job_closure(final)
        _after(cleanup, cleanup_dependencies, backend, status="any")
        _describe_job(cleanup, "cleanup", profile=config.klej)
        final = [cleanup]
    return final, compute


def plan_document(
    pipeline: Path,
    config_path: Path | None,
    repo_root: Path,
    plan: ReproPlan,
    backend: str,
    commit_mode: str,
    config: Config | None,
    runner_hash: str | None = None,
) -> dict[str, Any]:
    selection = (
        "all pipelines"
        if (plan.options or {}).get("all_pipelines")
        else ", ".join(plan.targets)
    )
    pipeline_label = os.path.relpath(pipeline, repo_root)
    if len(plan.targets) != 1 or plan.targets[0] != pipeline_label:
        pipeline_label = selection
    result: dict[str, Any] = {
        "plan_version": 2,
        "pipeline": pipeline_label,
        "targets": list(plan.targets),
        "repro_options": plan.options or {},
        "slurm_config": str(config_path) if config_path is not None else None,
        "backend": backend,
        "commit_mode": commit_mode,
        "jobs": {},
        "total_stages": plan.total_stages,
        "scheduled_stages": len(plan.stages),
        "external_dependencies": plan.external_dependencies,
        "stages": [
            {
                "name": stage.name,
                "target": stage.target,
                "command": stage.command,
                "id": _logical_id(stage.target),
                "profiles": list(stage.profiles),
                "settings": [
                    (stage.config or config).profiles[profile].settings
                    for profile in stage.profiles
                ],
                "dependencies": stage.dependencies,
                "dvcfile": stage.dvcfile,
                "config": stage.config_path,
                "force": stage.force,
            }
            for stage in plan.stages
        ],
    }
    if runner_hash:
        result["runner_hash"] = runner_hash
    return result


def physical_jobs_document(
    leaves: list[dawgz.Job], results: dict[dawgz.Job, Any] | None = None
) -> list[dict[str, Any]]:
    results = results or {}
    ordered: list[dawgz.Job] = []
    seen: set[dawgz.Job] = set()

    def visit(job: dawgz.Job) -> None:
        if job in seen:
            return
        seen.add(job)
        for dependency in job.dependencies:
            visit(dependency)
        ordered.append(job)

    for leaf in leaves:
        visit(leaf)
    return [
        {
            "name": job.name,
            "role": getattr(job, "klej_role", "internal"),
            "logical_id": getattr(job, "klej_logical_id", None),
            "target": getattr(job, "klej_target", None),
            "profile": getattr(job, "klej_profile", None),
            "slurm_job_id": (
                str(results[job])
                if job in results and results[job] is not None
                else None
            ),
        }
        for job in ordered
    ]


def update_plan_jobs(
    document: dict[str, Any],
    leaves: list[dawgz.Job],
    results: dict[dawgz.Job, Any] | None = None,
) -> None:
    physical = physical_jobs_document(leaves, results)
    document["physical_jobs"] = physical
    for stage in document["stages"]:
        stage["physical_jobs"] = [
            job["name"] for job in physical if job["logical_id"] == stage["id"]
        ]


def print_plan(document: dict[str, Any]) -> None:
    targets = document.get("targets")
    if isinstance(targets, list):
        print("Targets: " + ", ".join(map(str, targets)))
    print(
        f"Pipeline: {document['pipeline']} ({document['scheduled_stages']}/{document['total_stages']} stages)"
    )
    print(
        f"Backend: {document['backend']}; commit mode: {document['commit_mode']}"
    )
    stages = document["stages"]
    if stages:
        if document["commit_mode"] == "separate":
            commit_jobs = len(stages)
        elif document["commit_mode"] == "end":
            commit_jobs = len({stage["dvcfile"] for stage in stages})
        else:
            commit_jobs = 0
        internal_jobs = commit_jobs + 1  # reservation cleanup
        logical_jobs = len(stages) + internal_jobs
        physical_jobs = internal_jobs + sum(
            len(stage["profiles"]) + 1
            if document["backend"] == "slurm" and len(stage["profiles"]) > 1
            else 1
            for stage in stages
        )
    else:
        logical_jobs = physical_jobs = 0
    print(f"Jobs: {logical_jobs} logical; {physical_jobs} physical")
    if document["external_dependencies"]:
        print(
            "Assumed external: " + ", ".join(document["external_dependencies"])
        )
    for stage in document["stages"]:
        dependencies = ", ".join(stage["dependencies"]) or "-"
        print(f"\n{stage['target']} [{', '.join(stage['profiles'])}]")
        if document["backend"] == "slurm" and len(stage["profiles"]) > 1:
            print(f"  jobs: {len(stage['profiles'])} attempts + 1 gate")
        elif len(stage["profiles"]) > 1:
            print("  jobs: 1 stage (first alternative)")
        else:
            print("  jobs: 1 stage")
        print(f"  after: {dependencies}")
        print(f"  cmd: {stage['command']}")


def write_plan(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)
