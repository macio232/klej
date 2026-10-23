import json
import os
import pytest

from dvc.config import Config as DvcConfig
from dvc.repo import Repo
from pathlib import Path

from klej.__main__ import _parser, main
from klej.runner import release_reservation, reserve_targets


def _repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "site-cache"))
    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True)


def _config(tmp_path: Path) -> DvcConfig:
    return DvcConfig(str(tmp_path / ".dvc"))


def _legacy_guards(tmp_path: Path) -> list[Path]:
    paths = [
        tmp_path / ".dvc/tmp/lock",
        tmp_path / ".dvc/tmp/rwlock.lock",
        tmp_path / ".dvc/tmp/updater.lock",
    ]
    for path in paths:
        path.write_text("legacy\n")
    return paths


def _pipeline(tmp_path: Path) -> None:
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  make:\n    cmd: echo ok > result\n    outs: [result]\n"
    )
    (tmp_path / "klej.yaml").write_text(
        "profiles: {default: {cpus: 1}}\n"
        "stages: {make: default}\n"
        "commit: default\n"
        "klej: default\n"
    )


def test_setup_locking_parser_options() -> None:
    args = _parser().parse_args(["setup-locking", "--local", "--mute"])

    assert args.command == "setup-locking"
    assert args.local and args.mute


def test_setup_locking_mute_rejects_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)

    with pytest.raises(SystemExit):
        main(["setup-locking", "--mute", "--force"])

    assert "cannot be combined" in capsys.readouterr().err
    assert not (tmp_path / ".klej").exists()


def test_setup_locking_mute_changes_only_project_klej_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    guards = _legacy_guards(tmp_path)
    rwlock = tmp_path / ".dvc/tmp/rwlock"
    rwlock.write_text("")
    _pipeline(tmp_path)

    main(["setup-locking", "--mute"])

    assert _config(tmp_path)["core"]["hardlink_lock"] is False
    assert all(path.exists() for path in guards)
    assert rwlock.read_text() == ""
    project = tmp_path / ".klej/config"
    assert "mute_warning = true" in project.read_text()
    assert not (tmp_path / ".klej/config.local").exists()
    assert (
        "Muted DVC locking warning in .klej/config." in capsys.readouterr().out
    )

    rwlock.write_text(json.dumps({"read": {}, "write": {}}))
    main(["repro", "--dry"])

    assert "core.hardlink_lock is disabled" not in capsys.readouterr().err


def test_setup_locking_mute_local_updates_only_local_klej_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path, monkeypatch)

    main(["setup-locking", "--mute", "--local"])

    assert not (tmp_path / ".klej/config").exists()
    assert (
        "mute_warning = true" in (tmp_path / ".klej/config.local").read_text()
    )
    assert _config(tmp_path)["core"]["hardlink_lock"] is False


def test_local_klej_config_can_restore_locking_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    _pipeline(tmp_path)
    klej_root = tmp_path / ".klej"
    klej_root.mkdir()
    (klej_root / "config").write_text("[locking]\nmute_warning = true\n")
    (klej_root / "config.local").write_text("[locking]\nmute_warning = false\n")

    main(["repro", "--dry"])

    assert "core.hardlink_lock is disabled" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        "[locking]\nmute_warning = maybe\n",
        "[unsupported]\nvalue = true\n",
        "[locking]\nunsupported = true\n",
    ],
)
def test_invalid_klej_config_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    _repository(tmp_path, monkeypatch)
    _pipeline(tmp_path)
    klej_root = tmp_path / ".klej"
    klej_root.mkdir()
    (klej_root / "config").write_text(content)

    with pytest.raises(SystemExit):
        main(["repro", "--dry"])

    assert ".klej/config" in capsys.readouterr().err


def test_setup_locking_updates_project_config_and_cleans_legacy_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    guards = _legacy_guards(tmp_path)
    rwlock = tmp_path / ".dvc/tmp/rwlock"
    content = json.dumps({"read": {}, "write": {}})
    rwlock.write_text(content)

    main(["setup-locking"])

    config = _config(tmp_path)
    assert config["core"]["hardlink_lock"] is True
    assert config["core"]["no_scm"] is True
    assert config.load_one("repo")["core"]["hardlink_lock"] == "true"
    assert all(not path.exists() for path in guards)
    assert rwlock.read_text() == content
    assert not (tmp_path / ".klej").exists()
    assert (
        "Configured DVC hardlink locking in .dvc/config."
        in capsys.readouterr().out
    )


def test_setup_locking_local_updates_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path, monkeypatch)

    main(["setup-locking", "--local"])

    config = _config(tmp_path)
    assert config.load_one("local")["core"]["hardlink_lock"] == "true"
    assert "hardlink_lock" not in config.load_one("repo").get("core", {})


def test_setup_locking_refuses_local_false_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    config = _config(tmp_path)
    with config.edit("local") as local:
        local.setdefault("core", {})["hardlink_lock"] = "false"

    with pytest.raises(SystemExit):
        main(["setup-locking"])

    assert "--local" in capsys.readouterr().err
    assert "hardlink_lock" not in _config(tmp_path).load_one("repo").get(
        "core", {}
    )


def test_setup_locking_refuses_active_klej_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    reservation = reserve_targets(tmp_path, ["dvc.yaml:make"])

    with pytest.raises(SystemExit):
        main(["setup-locking"])

    assert "active klej locks" in capsys.readouterr().err
    assert _config(tmp_path)["core"]["hardlink_lock"] is False
    release_reservation(tmp_path, *reservation)


def test_setup_locking_force_preserves_malformed_rwlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    guards = _legacy_guards(tmp_path)
    rwlock = tmp_path / ".dvc/tmp/rwlock"
    rwlock.write_text("")
    reservation = reserve_targets(tmp_path, ["dvc.yaml:make"])

    main(["setup-locking", "--force"])

    assert _config(tmp_path)["core"]["hardlink_lock"] is True
    assert all(not path.exists() for path in guards)
    assert rwlock.read_text() == ""
    assert "may disrupt active DVC or Slurm work" in capsys.readouterr().err
    release_reservation(tmp_path, *reservation)


@pytest.mark.parametrize(
    "content",
    ["", json.dumps({"read": {"input": [{"pid": 1, "cmd": "dvc"}]}})],
)
def test_setup_locking_refuses_non_idle_rwlock_without_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
) -> None:
    _repository(tmp_path, monkeypatch)
    guard = tmp_path / ".dvc/tmp/rwlock.lock"
    guard.write_text("legacy\n")
    rwlock = tmp_path / ".dvc/tmp/rwlock"
    rwlock.write_text(content)

    with pytest.raises(SystemExit):
        main(["setup-locking"])

    assert "rwlock" in capsys.readouterr().err
    assert _config(tmp_path)["core"]["hardlink_lock"] is False
    assert guard.exists()
    assert rwlock.read_text() == content


def test_setup_locking_refuses_hardlinked_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    guard = tmp_path / ".dvc/tmp/lock"
    guard.write_text("claim\n")
    os.link(guard, tmp_path / ".dvc/tmp/claim.lock")

    with pytest.raises(SystemExit):
        main(["setup-locking"])

    assert "active or ambiguous" in capsys.readouterr().err
    assert _config(tmp_path)["core"]["hardlink_lock"] is False


def test_setup_locking_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path, monkeypatch)

    main(["setup-locking"])
    main(["setup-locking"])

    assert _config(tmp_path)["core"]["hardlink_lock"] is True
    assert not (tmp_path / ".dvc/tmp/lock").exists()


@pytest.mark.parametrize(
    ("arguments", "warning"),
    [
        (["repro", "--dry"], True),
        (["repro", "--dry", "--backend", "local"], False),
    ],
)
def test_repro_warns_about_locking_only_for_slurm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    warning: bool,
) -> None:
    _repository(tmp_path, monkeypatch)
    _pipeline(tmp_path)

    main(arguments)

    stderr = capsys.readouterr().err
    assert ("setup-locking" in stderr) is warning
    assert stderr.count("core.hardlink_lock is disabled") == int(warning)


def test_slurm_repro_does_not_warn_when_hardlink_locking_is_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repository(tmp_path, monkeypatch)
    _pipeline(tmp_path)
    config = _config(tmp_path)
    with config.edit("repo") as project:
        project.setdefault("core", {})["hardlink_lock"] = "true"

    main(["repro", "--dry"])

    assert "setup-locking" not in capsys.readouterr().err
