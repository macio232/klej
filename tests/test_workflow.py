import ast
import dawgz
import json
import pytest
import subprocess
import sys

from dvc.repo import Repo
from pathlib import Path

from klej import __main__ as cli
from klej.__main__ import _parser, main
from klej.config import Config, Profile
from klej.runner import lock_documents, release_reservation, reserve_targets
from klej.runner_cache import install_runner
from klej.workflow import (
    ReproOptions,
    ReproPlan,
    StagePlan,
    build_plan,
    build_repro_plan,
    create_jobs,
    plan_document,
    print_plan,
    update_plan_jobs,
)


def _config() -> Config:
    return Config(
        profiles={"default": Profile({"cpus": 1})},
        stages={"make": ("default",)},
        commit="default",
        klej="default",
        environment=(),
        wrapper=(),
    )


def _graph_config() -> Config:
    return Config(
        profiles={
            "compute": Profile({"cpus": 1}),
            "commit": Profile({"cpus": 2}),
            "klej": Profile({"cpus": 3}),
        },
        stages={"first": ("compute",), "second": ("compute",)},
        commit="commit",
        klej="klej",
        environment=(),
        wrapper=(),
    )


def _two_stage_plan() -> ReproPlan:
    return ReproPlan(
        stages=(
            StagePlan("first", "dvc.yaml:first", "first", ("compute",), ()),
            StagePlan(
                "second",
                "dvc.yaml:second",
                "second",
                ("compute",),
                ("dvc.yaml:first",),
            ),
        ),
        total_stages=2,
        external_dependencies=(),
    )


def _create_graph_jobs(
    tmp_path: Path,
    commit_mode: str,
    reservation: tuple[str, str] | None = None,
) -> tuple[list[dawgz.Job], dict[str, dawgz.Job]]:
    return create_jobs(
        tmp_path,
        tmp_path,
        tmp_path / "runner.py",
        _two_stage_plan(),
        _graph_config(),
        "local",
        commit_mode,
        tmp_path / "dvc.yaml",
        reservation,
    )


def _arguments(job: dawgz.Job) -> list[str]:
    return ast.literal_eval(job.args_repr[0])


def test_end_commit_job_depends_on_leaf_compute(tmp_path: Path) -> None:
    final, compute = _create_graph_jobs(tmp_path, "end")

    first, second = compute.values()
    assert second.dependencies == {first: "success"}
    assert final[0].name == "dvc_commit"
    assert final[0].dependencies == {second: "success"}


def test_inline_commit_preserves_compute_graph_and_profile(
    tmp_path: Path,
) -> None:
    final, compute = _create_graph_jobs(tmp_path, "stage")

    first, second = compute.values()
    assert second.dependencies == {first: "success"}
    assert final == [second]
    assert all(_arguments(job)[-1] == "--commit" for job in compute.values())
    assert all(job.settings == {"cpus": 1} for job in compute.values())


def test_separate_commits_bridge_compute_stages(tmp_path: Path) -> None:
    final, compute = _create_graph_jobs(tmp_path, "separate")

    first, second = compute.values()
    first_commit = next(iter(second.dependencies))
    second_commit = final[0]
    assert first_commit.name == "commit_first"
    assert first_commit.dependencies == {first: "success"}
    assert second.dependencies == {first_commit: "success"}
    assert second_commit.name == "commit_second"
    assert second_commit.dependencies == {second: "success"}
    assert first_commit.settings == second_commit.settings == {"cpus": 2}


def test_no_commit_jobs_retain_dvc_dependencies(tmp_path: Path) -> None:
    final, compute = _create_graph_jobs(tmp_path, "none")

    first, second = compute.values()
    assert second.dependencies == {first: "success"}
    assert final == [second]
    assert {job.name for job in compute.values()} == {"first", "second"}
    assert [job.klej_target for job in compute.values()] == [
        "dvc.yaml:first",
        "dvc.yaml:second",
    ]


def test_alternative_stage_expands_to_attempts_and_klej_gate(
    tmp_path: Path,
) -> None:
    config = Config(
        profiles={
            "small": Profile({"cpus": 1}),
            "large": Profile({"cpus": 8}),
            "control": Profile({"cpus": 2}),
        },
        stages={"make": ("small", "large")},
        commit="small",
        klej="control",
        environment=(),
        wrapper=("wrapper",),
    )
    plan = ReproPlan(
        (StagePlan("make", "dvc.yaml:make", "cmd", ("small", "large"), ()),),
        1,
        (),
    )

    final, compute = create_jobs(
        tmp_path,
        tmp_path,
        tmp_path / "runner.py",
        plan,
        config,
        "slurm",
        "none",
        tmp_path / "dvc.yaml",
        ("abc123", "token"),
    )

    gate = compute["dvc.yaml:make"]
    attempts = list(gate.dependencies)
    cleanup = final[0]
    assert gate.klej_role == "gate"
    assert gate.settings["cpus"] == 2
    assert gate.wait_mode == "any"
    assert {job.klej_profile for job in attempts} == {"small", "large"}
    assert {job.settings["cpus"] for job in attempts} == {1, 8}
    assert all(job.settings["hold"] is True for job in attempts)
    assert all(status == "success" for status in gate.dependencies.values())
    assert all(
        job.settings["kill_on_invalid_dep"] == "yes" for job in [gate, cleanup]
    )
    assert set(attempts).issubset(cleanup.dependencies)

    document = plan_document(
        tmp_path / "dvc.yaml",
        tmp_path / "klej.yaml",
        tmp_path,
        plan,
        "slurm",
        "none",
        config,
    )
    update_plan_jobs(document, final)
    assert document["plan_version"] == 2
    assert document["jobs"] == {}
    assert document["stages"][0]["profiles"] == ["small", "large"]
    assert {job["role"] for job in document["physical_jobs"]} == {
        "attempt",
        "gate",
        "cleanup",
    }


def test_local_backend_uses_only_first_alternative(tmp_path: Path) -> None:
    config = Config(
        profiles={
            "first": Profile({"cpus": 1}),
            "second": Profile({"cpus": 2}),
        },
        stages={"make": ("first", "second")},
        commit="first",
        klej="first",
        environment=(),
        wrapper=(),
    )
    plan = ReproPlan(
        (StagePlan("make", "dvc.yaml:make", "cmd", ("first", "second"), ()),),
        1,
        (),
    )

    final, compute = create_jobs(
        tmp_path,
        tmp_path,
        tmp_path / "runner.py",
        plan,
        config,
        "local",
        "none",
        tmp_path / "dvc.yaml",
        None,
    )

    assert final == list(compute.values())
    assert compute["dvc.yaml:make"].settings == {"cpus": 1}
    assert compute["dvc.yaml:make"].klej_role == "stage"


@pytest.mark.parametrize(
    ("backend", "commit_mode", "expected"),
    [
        ("slurm", "end", "Jobs: 5 logical; 7 physical"),
        ("slurm", "separate", "Jobs: 5 logical; 7 physical"),
        ("slurm", "stage", "Jobs: 3 logical; 5 physical"),
        ("slurm", "none", "Jobs: 3 logical; 5 physical"),
        ("local", "end", "Jobs: 5 logical; 5 physical"),
    ],
)
def test_plan_reports_whole_workflow_logical_and_physical_job_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    backend: str,
    commit_mode: str,
    expected: str,
) -> None:
    config = Config(
        profiles={
            "small": Profile({"cpus": 1}),
            "large": Profile({"cpus": 8}),
            "control": Profile({"cpus": 1}),
        },
        stages={
            "first": ("small", "large"),
            "second": ("small",),
        },
        commit="small",
        klej="control",
        environment=(),
        wrapper=(),
    )
    plan = ReproPlan(
        (
            StagePlan(
                "first",
                "dvc.yaml:first",
                "first",
                ("small", "large"),
                (),
            ),
            StagePlan(
                "second",
                "nested/dvc.yaml:second",
                "second",
                ("small",),
                ("dvc.yaml:first",),
                dvcfile="nested/dvc.yaml",
            ),
        ),
        2,
        (),
    )
    document = plan_document(
        tmp_path / "dvc.yaml",
        tmp_path / "klej.yaml",
        tmp_path,
        plan,
        backend,
        commit_mode,
        config,
    )

    print_plan(document)

    output = capsys.readouterr().out
    assert expected in output
    if backend == "slurm":
        assert "jobs: 2 attempts + 1 gate" in output
    else:
        assert "jobs: 1 stage (first alternative)" in output
    assert output.splitlines().count("  jobs: 1 stage") == 1


def test_empty_plan_reports_no_projected_jobs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = plan_document(
        tmp_path / "dvc.yaml",
        None,
        tmp_path,
        ReproPlan((), 1, ()),
        "slurm",
        "end",
        None,
    )

    print_plan(document)

    assert "Jobs: 0 logical; 0 physical" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("commit_mode", "dependency_names"),
    [
        ("end", {"first", "second", "dvc_commit"}),
        ("stage", {"first", "second"}),
        (
            "separate",
            {"first", "commit_first", "second", "commit_second"},
        ),
        ("none", {"first", "second"}),
    ],
)
def test_reservation_cleanup_runs_after_every_job_with_any_status(
    tmp_path: Path, commit_mode: str, dependency_names: set[str]
) -> None:
    final, _compute = _create_graph_jobs(
        tmp_path, commit_mode, reservation=("run-id", "token")
    )

    cleanup = final[0]
    assert cleanup.name == "klej_reservation_cleanup"
    assert {job.name for job in cleanup.dependencies} == dependency_names
    assert set(cleanup.dependencies.values()) == {"any"}
    assert cleanup.settings == {"cpus": 3}


def test_build_plan_and_standalone_runner_execute_a_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    pipeline = tmp_path / "dvc.yaml"
    pipeline.write_text(
        "stages:\n"
        "  make:\n"
        f"    cmd: {sys.executable} -c \"open('result.txt', 'w').write('ok')\"\n"
        "    outs: [result.txt]\n"
    )

    plan = build_plan(tmp_path, pipeline, _config(), force=True)

    assert [stage.name for stage in plan.stages] == ["make"]
    runner = install_runner(tmp_path)
    subprocess.run(
        [
            sys.executable,
            runner,
            tmp_path,
            tmp_path,
            "stage",
            plan.stages[0].target,
        ],
        cwd=tmp_path,
        check=True,
    )
    assert (tmp_path / "result.txt").read_text() == "ok"
    assert not (tmp_path / "dvc.lock").exists()


def test_runner_distinguishes_no_commit_from_no_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    runner = install_runner(tmp_path)

    subprocess.run(
        [
            sys.executable,
            runner,
            tmp_path,
            tmp_path,
            "stage",
            "dvc.yaml:make",
            "--commit",
            "--no-commit",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert (tmp_path / "dvc.lock").is_file()
    status = subprocess.run(
        [sys.executable, "-m", "dvc", "status"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "not in cache" in status.stdout

    (tmp_path / "dvc.lock").unlink()
    (tmp_path / "result").unlink()
    subprocess.run(
        [
            sys.executable,
            runner,
            tmp_path,
            tmp_path,
            "stage",
            "dvc.yaml:make",
            "--no-metadata",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert not (tmp_path / "dvc.lock").exists()


def test_no_run_cache_bypasses_restore_but_saves_new_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "source").write_text("data\n")
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: cp source result\n"
        "    deps: [source]\n    outs: [result]\n"
    )
    runner = install_runner(tmp_path)

    subprocess.run(
        [
            sys.executable,
            runner,
            tmp_path,
            tmp_path,
            "stage",
            "dvc.yaml:make",
            "--commit",
            "--no-run-cache",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert any(
        path.is_file()
        for path in (tmp_path / ".dvc" / "cache" / "runs").rglob("*")
    )


def test_build_plan_propagates_dirty_dependencies_without_unrelated_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    pipeline = tmp_path / "dvc.yaml"
    (tmp_path / "a-source.txt").write_text("first\n")
    (tmp_path / "c-source.txt").write_text("unrelated\n")
    pipeline.write_text(
        "stages:\n"
        "  a:\n"
        "    cmd: cp a-source.txt a.txt\n"
        "    deps: [a-source.txt]\n"
        "    outs: [a.txt]\n"
        "  b:\n"
        "    cmd: cp a.txt b.txt\n"
        "    deps: [a.txt]\n"
        "    outs: [b.txt]\n"
        "  c:\n"
        "    cmd: cp c-source.txt c.txt\n"
        "    deps: [c-source.txt]\n"
        "    outs: [c.txt]\n"
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro"], cwd=tmp_path, check=True
    )
    (tmp_path / "a-source.txt").write_text("changed\n")
    config = Config(
        profiles={"default": Profile({"cpus": 1})},
        stages={"a": ("default",), "b": ("default",), "c": ("default",)},
        commit="default",
        klej="default",
        environment=(),
        wrapper=(),
    )

    plan = build_plan(tmp_path, pipeline, config)

    assert {stage.target for stage in plan.stages} == {
        "dvc.yaml:a",
        "dvc.yaml:b",
    }


def test_reservations_are_visible_and_released(tmp_path: Path) -> None:
    reservation = reserve_targets(tmp_path, ["make"])

    documents = lock_documents(tmp_path)
    assert f"run:{reservation[0]}" in documents

    release_reservation(tmp_path, *reservation)
    assert lock_documents(tmp_path) == {}


def test_checkpoint_marked_output_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    pipeline = tmp_path / "dvc.yaml"
    pipeline.write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n"
        "    outs:\n      - result:\n          checkpoint: true\n"
    )

    plan = build_plan(tmp_path, pipeline, _config(), force=True)

    assert [stage.target for stage in plan.stages] == ["dvc.yaml:make"]


def test_repository_plan_includes_cross_file_dependencies_and_target_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  source:\n    cmd: echo data > data.txt\n"
        "    outs: [data.txt]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: cp ../data.txt result.txt\n"
        "    deps: [../data.txt]\n    outs: [result.txt]\n"
    )
    config = Config(
        profiles={"default": Profile({"cpus": 1})},
        stages={"source": ("default",), "make": ("default",)},
        commit="default",
        klej="default",
        environment=(),
        wrapper=(),
    )

    def resolve(_path: str) -> tuple[Config, str]:
        return config, "klej.yaml"

    plan = build_repro_plan(
        tmp_path,
        ["nested/dvc.yaml:make"],
        resolve,
        ReproOptions(force=True),
    )
    single = build_repro_plan(
        tmp_path,
        ["nested/dvc.yaml:make"],
        resolve,
        ReproOptions(force=True, single_item=True),
    )

    assert [stage.target for stage in plan.stages] == [
        "dvc.yaml:source",
        "nested/dvc.yaml:make",
    ]
    assert plan.stages[1].dependencies == ("dvc.yaml:source",)
    assert [stage.target for stage in single.stages] == ["nested/dvc.yaml:make"]


def test_repository_plan_supports_glob_recursive_downstream_and_dvc_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "source").write_text("data\n")
    subprocess.run(
        [sys.executable, "-m", "dvc", "add", "source"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n"
        "  train-one:\n    cmd: cp source one\n"
        "    deps: [source]\n    outs: [one]\n"
        "  train-two:\n    cmd: cp one two\n"
        "    deps: [one]\n    outs: [two]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dvc.yaml").write_text(
        "stages:\n  evaluate:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    config = Config(
        profiles={"default": Profile({"cpus": 1})},
        stages={
            "source.dvc": ("default",),
            "train-one": ("default",),
            "train-two": ("default",),
            "evaluate": ("default",),
        },
        commit="default",
        klej="default",
        environment=(),
        wrapper=(),
    )

    def resolve(_path: str) -> tuple[Config, None]:
        return config, None

    globbed = build_repro_plan(
        tmp_path,
        ["train-*"],
        resolve,
        ReproOptions(force=True, glob=True, single_item=True),
    )
    recursive = build_repro_plan(
        tmp_path,
        ["nested"],
        resolve,
        ReproOptions(force=True, recursive=True),
    )
    downstream = build_repro_plan(
        tmp_path,
        ["train-one"],
        resolve,
        ReproOptions(force=True, downstream=True),
    )
    data = build_repro_plan(
        tmp_path,
        ["source.dvc"],
        resolve,
        ReproOptions(force=True, single_item=True),
    )
    all_pipelines = build_repro_plan(
        tmp_path,
        ["ignored"],
        resolve,
        ReproOptions(force=True, all_pipelines=True),
    )

    assert {stage.name for stage in globbed.stages} == {
        "train-one",
        "train-two",
    }
    assert [stage.target for stage in recursive.stages] == [
        "nested/dvc.yaml:evaluate"
    ]
    assert {stage.name for stage in downstream.stages} == {
        "train-one",
        "train-two",
    }
    assert [stage.target for stage in data.stages] == ["source.dvc"]
    assert {stage.target for stage in all_pipelines.stages} == {
        "source.dvc",
        "dvc.yaml:train-one",
        "dvc.yaml:train-two",
        "nested/dvc.yaml:evaluate",
    }


def test_allow_missing_skips_an_otherwise_unchanged_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "source").write_text("data\n")
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: cp source result\n"
        "    deps: [source]\n    outs: [result]\n"
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "result").unlink()
    config = _config()

    changed = build_repro_plan(
        tmp_path,
        ["make"],
        lambda _path: (config, None),
    )
    allowed = build_repro_plan(
        tmp_path,
        ["make"],
        lambda _path: (config, None),
        ReproOptions(allow_missing=True),
    )

    assert [stage.target for stage in changed.stages] == ["dvc.yaml:make"]
    assert allowed.stages == ()


def test_force_downstream_marks_only_descendants_for_forced_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "source").write_text("first\n")
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n"
        "  first:\n    cmd: echo stable > first\n"
        "    deps: [source]\n    outs: [first]\n"
        "  second:\n    cmd: cp first second\n"
        "    deps: [first]\n    outs: [second]\n"
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "source").write_text("changed\n")
    config = Config(
        profiles={"default": Profile({"cpus": 1})},
        stages={"first": ("default",), "second": ("default",)},
        commit="default",
        klej="default",
        environment=(),
        wrapper=(),
    )

    normal = build_repro_plan(
        tmp_path,
        ["dvc.yaml"],
        lambda _path: (config, None),
    )
    forced = build_repro_plan(
        tmp_path,
        ["dvc.yaml"],
        lambda _path: (config, None),
        ReproOptions(force_downstream=True),
    )

    assert {stage.name: stage.force for stage in normal.stages} == {
        "first": False,
        "second": False,
    }
    assert {stage.name: stage.force for stage in forced.stages} == {
        "first": False,
        "second": True,
    }


def test_ignore_errors_uses_afterany_for_stage_dependencies(
    tmp_path: Path,
) -> None:
    final, compute = create_jobs(
        tmp_path,
        tmp_path,
        tmp_path / "runner.py",
        _two_stage_plan(),
        _graph_config(),
        "local",
        "none",
        tmp_path / "dvc.yaml",
        None,
        ignore_errors=True,
    )

    first, second = compute.values()
    assert second.dependencies == {first: "any"}
    assert final == [second]


def test_plan_document_records_multi_target_selection(tmp_path: Path) -> None:
    plan = ReproPlan(
        _two_stage_plan().stages,
        2,
        (),
        ("first", "second"),
        {"force": True},
    )

    document = plan_document(
        tmp_path / "dvc.yaml",
        tmp_path / "klej.yaml",
        tmp_path,
        plan,
        "slurm",
        "end",
        _graph_config(),
    )

    assert document["pipeline"] == "first, second"
    assert document["targets"] == ["first", "second"]
    assert document["repro_options"] == {"force": True}


def test_repro_cli_exposes_selected_dvc_options_only() -> None:
    args = _parser().parse_args([
        "repro",
        "prepare",
        "train",
        "-f",
        "-s",
        "--allow-missing",
        "--no-run-cache",
        "--ignore-errors",
        "-v",
    ])

    assert args.targets == ["prepare", "train"]
    assert args.force and args.single_item and args.allow_missing
    assert args.no_run_cache and args.ignore_errors and args.verbose
    for unsupported in (["repro", "--pull"], ["repro", "-q"], ["repro", "-k"]):
        with pytest.raises(SystemExit):
            _parser().parse_args(unsupported)


def test_cli_dry_run_resolves_stage_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\n"
        "stages: {make: default}\n"
        "commit: default\nklej: default\n"
    )

    main(["repro", "make", "--dry", "-v"])

    captured = capsys.readouterr()
    assert "dvc.yaml:make" in captured.out
    assert "Jobs: 3 logical; 3 physical" in captured.out
    assert "selected dvc.yaml:make: changed" in captured.err
    assert "DVC status dvc.yaml:make:" in captured.err
    assert not (tmp_path / ".klej").exists()


def test_mocked_slurm_submission_registers_alternative_race_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    records = tmp_path / "records"
    dawgz.set_dawgz_dir(records)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  train:\n    cmd: echo trained > model\n    outs: [model]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles:\n"
        "  small: {cpus: 1}\n"
        "  large: {cpus: 8}\n"
        "  control: {cpus: 1}\n"
        "stages: {train: [small, large]}\n"
        "commit: small\n"
        "klej: control\n"
    )
    identifiers = iter(map(str, range(100, 200)))
    scripts: list[str] = []
    release_snapshot: dict[str, object] = {}

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            scripts.append(Path(command[-1]).read_text())
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{next(identifiers)}\n", stderr=""
            )
        if command[:2] == ["scontrol", "release"]:
            plan_path = next(records.glob("*/plan.json"))
            document = json.loads(plan_path.read_text())
            owner_path = next(
                (tmp_path / ".klej/locks/runs").glob("*/owner.json")
            )
            owner = json.loads(owner_path.read_text())
            race = next(iter(owner["races"].values()))
            attempt_ids = [
                job["slurm_job_id"]
                for job in document["physical_jobs"]
                if job["role"] == "attempt"
            ]
            assert document["submission_state"] == "registered"
            assert all(job["slurm_job_id"] for job in document["physical_jobs"])
            assert {item["job_id"] for item in race["attempts"]} == set(
                attempt_ids
            )
            assert race["gate_job_id"] is not None
            assert command[2:] == attempt_ids
            release_snapshot.update(document=document, owner=owner)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", run)

    main(["repro", "train", "--force"])

    assert release_snapshot
    assert len(scripts) == 5
    assert not any("#SBATCH --array" in script for script in scripts)
    attempt_scripts = [script for script in scripts if "_attempt_" in script]
    assert len(attempt_scripts) == 2
    assert all("#SBATCH --hold" in script for script in attempt_scripts)
    gate = next(script for script in scripts if "_gate" in script)
    assert "#SBATCH --dependency=afterok:" in gate
    assert "?afterok:" in gate
    assert "#SBATCH --kill-on-invalid-dep=yes" in gate
    plan = release_snapshot["document"]
    assert isinstance(plan, dict)
    assert [job["role"] for job in plan["physical_jobs"]].count("attempt") == 2
    assert {job["role"] for job in plan["physical_jobs"]} == {
        "attempt",
        "gate",
        "commit",
        "cleanup",
    }


def _alternative_submission_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    records = tmp_path / "records"
    dawgz.set_dawgz_dir(records)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  train:\n    cmd: echo trained > model\n    outs: [model]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles:\n"
        "  small: {cpus: 1}\n"
        "  large: {cpus: 8}\n"
        "  control: {cpus: 1}\n"
        "stages: {train: [small, large]}\n"
        "commit: small\n"
        "klej: control\n"
    )
    return records


def test_registration_failure_cancels_cleanup_and_retains_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _alternative_submission_project(tmp_path, monkeypatch)
    identifiers = iter(map(str, range(100, 200)))
    cancelled = []

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{next(identifiers)}\n", stderr=""
            )
        if command[0] == "scancel":
            cancelled.append(command[1])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", run)

    def fail_registration(*args: object) -> None:
        physical_jobs = args[3]
        assert isinstance(physical_jobs, list)
        physical_jobs.append({
            "name": "plan_only",
            "role": "stage",
            "slurm_job_id": "199",
        })
        raise RuntimeError("registration failed")

    monkeypatch.setattr(
        "klej.__main__.register_reservation_jobs",
        fail_registration,
    )

    with pytest.raises(SystemExit):
        main(["repro", "train", "--force"])

    document = json.loads(next(records.glob("*/plan.json")).read_text())
    physical = document["physical_jobs"]
    assert document["submission_state"] == "failed"
    assert document["submission_failure"] == "registration failed"
    assert set(cancelled) == {
        job["slurm_job_id"] for job in physical if job["slurm_job_id"]
    } | {"199"}
    cleanup_id = next(
        job["slurm_job_id"] for job in physical if job["role"] == "cleanup"
    )
    assert cleanup_id in cancelled
    assert "199" in cancelled
    assert list(lock_documents(tmp_path)) == [
        f"run:{document['reservation_id']}"
    ]


def test_partial_attempt_release_failure_preserves_registered_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _alternative_submission_project(tmp_path, monkeypatch)
    identifiers = iter(map(str, range(200, 300)))
    cancelled = []
    write_plan = cli.write_plan

    def add_plan_only_job(path: Path, document: dict[str, object]) -> None:
        write_plan(path, document)
        if document.get("submission_state") == "registered":
            physical = document["physical_jobs"]
            assert isinstance(physical, list)
            physical.append({
                "name": "plan_only",
                "role": "stage",
                "slurm_job_id": "299",
            })

    monkeypatch.setattr(cli, "write_plan", add_plan_only_job)

    def run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{next(identifiers)}\n", stderr=""
            )
        if command[:2] == ["scontrol", "release"]:
            raise subprocess.CalledProcessError(1, command)
        if command[0] == "scancel":
            cancelled.append(command[1])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        main(["repro", "train", "--force"])

    document = json.loads(next(records.glob("*/plan.json")).read_text())
    physical = document["physical_jobs"]
    cleanup_id = next(
        job["slurm_job_id"] for job in physical if job["role"] == "cleanup"
    )
    non_cleanup_ids = {
        job["slurm_job_id"]
        for job in physical
        if job["role"] != "cleanup" and job["slurm_job_id"]
    }
    owner = lock_documents(tmp_path)[f"run:{document['reservation_id']}"]
    assert document["submission_state"] == "failed"
    assert "scontrol" in document["submission_failure"]
    assert set(cancelled) == non_cleanup_ids | {"299"}
    assert cleanup_id not in cancelled
    assert set(owner["slurm_job_ids"]) == non_cleanup_ids | {cleanup_id}
    assert owner["cleanup_job_id"] == cleanup_id


def test_interactive_confirmation_happens_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\n"
        "stages: {make: default}\n"
        "commit: default\nklej: default\n"
    )
    prompts = []

    def accept(prompt: str) -> str:
        prompts.append(prompt)
        return "yes"

    monkeypatch.setattr("builtins.input", accept)
    main(["repro", "make", "--dry", "-i", "-f"])

    assert prompts == ["Reproduce dvc.yaml:make? [y/N] "]
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(SystemExit):
        main(["repro", "make", "--dry", "-i", "-f"])
    assert not (tmp_path / ".klej").exists()


def test_partial_root_config_is_validated_after_pipeline_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\ncommit: default\nklej: default\n"
    )
    (pipeline / "klej.yaml").write_text("stages: {make: default}\n")

    main(["repro", "pipeline/dvc.yaml", "--dry", "-f"])

    assert "pipeline/dvc.yaml:make" in capsys.readouterr().out


def test_up_to_date_repro_does_not_validate_automatic_config_fragments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  root:\n    cmd: echo root > root.out\n    outs: [root.out]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dvc.yaml").write_text(
        "stages:\n  nested:\n    cmd: echo nested > nested.out\n"
        "    outs: [nested.out]\n"
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "-P"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\ncommit: default\nklej: default\n"
    )
    (nested / "klej.yaml").write_text("stages: {nested: default}\n")

    main(["repro", "nested/dvc.yaml", "--dry"])
    main(["repro", "--all-pipelines", "--dry"])

    output = capsys.readouterr().out
    assert output.count("Pipeline is up to date; nothing to schedule.") == 2
    assert not (tmp_path / ".klej").exists()


def test_up_to_date_repro_still_validates_explicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "invalid.yaml").write_text("stages: {make: missing}\n")

    with pytest.raises(SystemExit):
        main(["repro", "--config", "invalid.yaml", "--dry"])


def test_multi_pipeline_klej_settings_must_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  root:\n    cmd: echo root > root.out\n    outs: [root.out]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dvc.yaml").write_text(
        "stages:\n  nested:\n    cmd: echo nested > nested.out\n"
        "    outs: [nested.out]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles:\n  default: {cpus: 1}\n  cleanup: {cpus: 1}\n"
        "stages: {root: default}\ncommit: default\nklej: cleanup\n"
    )
    (nested / "klej.yaml").write_text(
        "profiles:\n  cleanup: {cpus: 2}\nstages: {nested: default}\n"
    )

    with pytest.raises(SystemExit):
        main(["repro", "--all-pipelines", "--force", "--dry"])


def test_explicit_config_bypasses_discovery_for_same_named_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo root > root.out\n    outs: [root.out]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo nested > nested.out\n"
        "    outs: [nested.out]\n"
    )
    (tmp_path / "klej.yaml").write_text("not: a valid config\n")
    (nested / "klej.yaml").write_text("also: invalid\n")
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        "profiles: {shared: {cpus: 2}}\n"
        "stages: {make: shared}\ncommit: shared\nklej: shared\n"
    )

    main([
        "repro",
        "--all-pipelines",
        "--force",
        "--dry",
        "--config",
        str(explicit),
    ])

    output = capsys.readouterr().out
    assert "dvc.yaml:make [shared]" in output
    assert "nested/dvc.yaml:make [shared]" in output


def test_cli_local_backend_runs_without_installing_runner_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    dawgz.set_dawgz_dir(tmp_path / "dawgz-records")
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n"
        "  make:\n"
        f"    cmd: {sys.executable} -c \"open('result.txt', 'w').write('ok')\"\n"
        "    outs: [result.txt]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\n"
        "stages: {make: default}\n"
        "commit: default\n"
        "klej: default\n"
    )

    main(["repro", "--backend", "local", "--no-commit"])

    assert (tmp_path / "result.txt").read_text() == "ok"
    assert lock_documents(tmp_path) == {}


def test_cli_stores_klej_state_in_invocation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    dawgz.set_dawgz_dir(tmp_path / "dawgz-records")
    project = tmp_path / "project"
    project.mkdir()
    Repo.init(project, no_scm=True)
    (project / "dvc.yaml").write_text(
        "stages:\n"
        "  make:\n"
        f"    cmd: {sys.executable} -c \"open('result.txt', 'w').write('ok')\"\n"
        "    outs: [result.txt]\n"
    )
    (project / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\n"
        "stages: {make: default}\n"
        "commit: default\n"
        "klej: default\n"
    )

    main([
        "repro",
        "project/dvc.yaml",
        "--backend",
        "local",
        "--no-commit",
    ])

    assert (project / "result.txt").read_text() == "ok"
    assert (tmp_path / ".klej" / "runners").is_dir()
    assert not (project / ".klej").exists()


def test_cross_file_dependencies_require_explicit_assumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    Repo.init(tmp_path, no_scm=True)
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  source:\n    cmd: echo data > data.txt\n"
        "    outs: [data.txt]\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    pipeline = nested / "dvc.yaml"
    pipeline.write_text(
        "stages:\n  make:\n    cmd: cp ../data.txt result.txt\n"
        "    deps: [../data.txt]\n    outs: [result.txt]\n"
    )

    with pytest.raises(ValueError, match="--assume-external-deps"):
        build_plan(tmp_path, pipeline, _config(), force=True)

    plan = build_plan(
        tmp_path,
        pipeline,
        _config(),
        force=True,
        assume_external=True,
    )
    assert plan.external_dependencies == ("dvc.yaml:source",)
    assert [stage.target for stage in plan.stages] == ["nested/dvc.yaml:make"]
