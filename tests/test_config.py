import pytest

from pathlib import Path

from klej.config import find_config, load_config, merge_config_files


def test_load_config_supports_wrapper_and_profile_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCOUNT", "research")
    path = tmp_path / "klej.yaml"
    path.write_text(
        """
profiles:
  cpu:
    account: $ACCOUNT
    cpus: 2
  gpu:
    gres: gpu:1
    environment:
      APPTAINER_NV: 1
stages:
  prepare: cpu
  train: gpu
commit: cpu
klej: cpu
environment:
  APPTAINER_NV: 0
wrapper: [./apptainer/run.sh]
"""
    )

    config = load_config(path)

    assert config.profiles["cpu"].settings == {
        "account": "research",
        "cpus": 2,
    }
    assert config.profiles["gpu"].environment == ("export APPTAINER_NV=1",)
    assert config.stages == {"prepare": ("cpu",), "train": ("gpu",)}
    assert config.environment == ("export APPTAINER_NV=0",)
    assert config.wrapper == ("./apptainer/run.sh",)


def test_find_config_prefers_pipeline_then_invocation_directory(
    tmp_path: Path,
) -> None:
    invocation = tmp_path / "work"
    pipeline_dir = tmp_path / "pipelines"
    invocation.mkdir()
    pipeline_dir.mkdir()
    fallback = invocation / "klej.yaml"
    adjacent = pipeline_dir / "klej.yaml"
    fallback.touch()

    assert find_config(pipeline_dir / "dvc.yaml", invocation) == fallback

    adjacent.touch()
    assert find_config(pipeline_dir / "dvc.yaml", invocation) == adjacent


def test_pipeline_config_partially_overlays_repository_config(
    tmp_path: Path,
) -> None:
    base = tmp_path / "klej.yaml"
    local = tmp_path / "pipeline" / "klej.yaml"
    local.parent.mkdir()
    base.write_text(
        "profiles:\n"
        "  cpu: {cpus: 1, mem: 1G}\n"
        "  commit: {cpus: 1}\n"
        "stages: {prepare: cpu}\n"
        "commit: commit\n"
        "klej: commit\n"
    )
    local.write_text("profiles:\n  cpu: {mem: 2G}\nstages: {train: cpu}\n")

    config = merge_config_files(base, local)

    assert config.profiles["cpu"].settings == {"cpus": 1, "mem": "2G"}
    assert config.stages == {"prepare": ("cpu",), "train": ("cpu",)}


def test_profile_environment_overlay_is_recursive(tmp_path: Path) -> None:
    base = tmp_path / "klej.yaml"
    local = tmp_path / "pipeline.yaml"
    base.write_text(
        "profiles:\n"
        "  cpu:\n"
        "    environment: {A: one, B: two}\n"
        "stages: {prepare: cpu}\n"
        "commit: cpu\nklej: cpu\n"
    )
    local.write_text(
        "profiles:\n  cpu:\n    environment: {B: override}\n"
        "stages: {train: cpu}\n"
    )

    config = merge_config_files(base, local)

    assert config.profiles["cpu"].environment == (
        "export A=one",
        "export B=override",
    )


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {}}\nstages: {build: default}\n"
        "commit: default\nklej: default\ntyop: true\n"
    )

    with pytest.raises(ValueError, match="unknown keys: tyop"):
        load_config(path)


def test_klej_names_internal_operation_profile(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {stage: {}, control: {cpus: 1}}\n"
        "stages: {build: stage}\n"
        "commit: stage\nklej: control\n"
    )

    config = load_config(path)

    assert config.klej == "control"


def test_finalize_reports_focused_migration_error(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {}}\nstages: {build: default}\n"
        "commit: default\nfinalize: default\n"
    )

    with pytest.raises(
        ValueError, match=r"^finalize has been renamed to klej$"
    ):
        load_config(path)


def test_klej_profile_must_be_a_single_name(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {}}\nstages: {build: default}\n"
        "commit: default\nklej: [default]\n"
    )

    with pytest.raises(TypeError, match="'klej' must name a profile"):
        load_config(path)


def test_stage_profiles_accept_ordered_alternatives(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {small: {}, large: {}}\n"
        "stages: {build: [small, large]}\n"
        "commit: small\nklej: small\n"
    )

    assert load_config(path).stages["build"] == ("small", "large")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("[]", "must name at least one profile"),
        ("[small, small]", "contains duplicate profile 'small'"),
        ("[small, 1]", "must be a profile name or a list of profile names"),
        ("1", "must be a profile name or a list of profile names"),
        ("[small, missing]", "undefined profiles: missing"),
    ],
)
def test_invalid_stage_profile_alternatives_are_rejected(
    tmp_path: Path, value: str, message: str
) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {small: {}}\n"
        f"stages: {{build: {value}}}\n"
        "commit: small\nklej: small\n"
    )

    with pytest.raises((TypeError, ValueError), match=message):
        load_config(path)


def test_kill_on_invalid_dependency_is_managed(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {kill_on_invalid_dep: no}}\n"
        "stages: {build: default}\ncommit: default\nklej: default\n"
    )

    with pytest.raises(ValueError, match="kill-on-invalid-dep"):
        load_config(path)


def test_hold_is_managed_for_alternative_submission(tmp_path: Path) -> None:
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {hold: true}}\n"
        "stages: {build: default}\ncommit: default\nklej: default\n"
    )

    with pytest.raises(ValueError, match="managed option 'hold'"):
        load_config(path)


def test_environment_reference_is_not_resolved_into_saved_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOKEN", "secret-value")
    path = tmp_path / "klej.yaml"
    path.write_text(
        "profiles: {default: {}}\nstages: {build: default}\n"
        "commit: default\nklej: default\n"
        "environment: {TOKEN_COPY: $TOKEN}\n"
    )

    config = load_config(path)

    assert config.environment == ("export TOKEN_COPY=${TOKEN}",)
    assert "secret-value" not in config.environment[0]
