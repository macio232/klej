import dawgz
import json
import os
import pytest
import shlex
import shutil
import subprocess
import sys

from pathlib import Path

from klej.__main__ import main
from klej.config import Config, Profile, load_config
from klej.runner import lock_documents
from klej.workflow import build_plan, plan_document


def test_monte_carlo_pi_example_runs_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parents[1] / "examples" / "monte-carlo-pi"
    example = tmp_path / "monte-carlo-pi"
    shutil.copytree(
        source,
        example,
        ignore=shutil.ignore_patterns("dvc.lock", "outputs"),
    )
    (example / ".gitignore").write_text("/outputs/\n/.dawgz/\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.setenv("CPU_PARTITIONS", "test")
    monkeypatch.chdir(example)
    dawgz.set_dawgz_dir(tmp_path / "dawgz-records")

    config = load_config(example / "klej.yaml")
    plan = build_plan(example, example / "dvc.yaml", config, force=True)
    simulations = {f"dvc.yaml:simulate@{seed}" for seed in range(4)}
    stages = {stage.target: stage for stage in plan.stages}

    assert set(stages) == {*simulations, "dvc.yaml:estimate"}
    assert stages["dvc.yaml:estimate"].dependencies == tuple(
        sorted(simulations)
    )
    assert all(
        stages[target].profiles == ("simulation",) for target in simulations
    )
    assert stages["dvc.yaml:estimate"].profiles == ("aggregation",)

    main(["repro", "--backend", "local", "--local-jobs", "4"])

    result = json.loads((example / "outputs" / "pi.json").read_text())
    assert result["samples"] == 400_000
    assert result["shards"] == 4
    assert 3.0 < result["pi"] < 3.3
    assert lock_documents(example) == {}
    assert build_plan(example, example / "dvc.yaml", config).stages == ()


def test_foreach_expansions_have_independent_alternative_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(__file__).parents[1] / "examples" / "monte-carlo-pi"
    example = tmp_path / "monte-carlo-pi"
    shutil.copytree(
        source,
        example,
        ignore=shutil.ignore_patterns("dvc.lock", "outputs"),
    )
    (example / ".gitignore").write_text("/outputs/\n")
    subprocess.run(["git", "init", "-q"], cwd=example, check=True)
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.setenv("CPU_PARTITIONS", "test")
    original = load_config(example / "klej.yaml")
    config = Config(
        profiles={
            **original.profiles,
            "simulation_large": Profile({"cpus": 2}),
        },
        stages={
            **original.stages,
            "simulate": ("simulation", "simulation_large"),
        },
        commit=original.commit,
        klej=original.klej,
        environment=original.environment,
        wrapper=original.wrapper,
    )
    plan = build_plan(example, example / "dvc.yaml", config, force=True)
    document = plan_document(
        example / "dvc.yaml", None, example, plan, "slurm", "none", config
    )
    simulations = [
        stage for stage in document["stages"] if "simulate@" in stage["target"]
    ]

    assert len(simulations) == 4
    assert len({stage["id"] for stage in simulations}) == 4
    assert all(
        stage["profiles"] == ["simulation", "simulation_large"]
        for stage in simulations
    )


def test_alternative_executors_smoke_example_is_self_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        Path(__file__).parents[1] / "examples" / "alternative-executors-smoke"
    )
    example = tmp_path / "alternative-executors-smoke"
    shutil.copytree(source, example)
    subprocess.run(["git", "init", "-q"], cwd=example, check=True)
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.setenv("CPU_PARTITIONS", "test")
    config = load_config(example / "klej.yaml")
    plan = build_plan(example, example / "dvc.yaml", config, force=True)

    assert {stage.name for stage in plan.stages} == {"success", "failure"}
    assert all(
        stage.profiles == ("candidate_a", "candidate_b")
        for stage in plan.stages
    )
    success = subprocess.run(
        [sys.executable, "workload.py", "success"], cwd=example, check=False
    )
    failure = subprocess.run(
        [sys.executable, "workload.py", "failure"], cwd=example, check=False
    )
    python = example / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    python.chmod(0o755)
    wrapped = subprocess.run(
        [
            "sh",
            "wrapper.sh",
            "python -c \"open('wrapped', 'w').write('ok')\"",
        ],
        cwd=example,
        check=False,
        env={**os.environ, "PATH": "/usr/bin:/bin"},
    )

    assert success.returncode == 0
    assert failure.returncode == 23
    assert wrapped.returncode == 0
    assert (example / "outputs/success.json").is_file()
    assert (example / "wrapped").read_text() == "ok"
    assert len((example / "markers/success.log").read_text().splitlines()) == 1
    assert len((example / "markers/failure.log").read_text().splitlines()) == 1
    assert len((example / "markers/wrapper.log").read_text().splitlines()) == 1
