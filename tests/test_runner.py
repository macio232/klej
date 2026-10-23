import pytest
import sys
import types

from contextlib import nullcontext
from pathlib import Path
from typing import Self

from klej.runner import commit_targets, run_stage


@pytest.mark.parametrize("metadata", ["deferred", "inline", "none"])
def test_run_stage_waits_to_reacquire_dvc_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str
) -> None:
    wait_for_lock = None

    class StageFailure(Exception):
        pass

    class Stage:
        def reproduce(self, **kwargs: object) -> None:
            raise StageFailure("stage failed")

    class Stages:
        def collect(self, target: str) -> list[Stage]:
            return [Stage()]

    class Repo:
        def __init__(self, root: str, *, _wait_for_lock: bool = False) -> None:
            nonlocal wait_for_lock
            wait_for_lock = _wait_for_lock
            self.stage = Stages()
            self.lock = nullcontext()

        def close(self) -> None:
            pass

    dvc = types.ModuleType("dvc")
    dvc_repo = types.ModuleType("dvc.repo")
    dvc_repo.Repo = Repo
    monkeypatch.setitem(sys.modules, "dvc", dvc)
    monkeypatch.setitem(sys.modules, "dvc.repo", dvc_repo)

    with pytest.raises(StageFailure, match="stage failed"):
        run_stage(tmp_path, tmp_path, "dvc.yaml:stage", metadata=metadata)

    assert wait_for_lock is True


def test_no_commit_metadata_waits_for_dvc_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wait_for_lock = None

    class Stage:
        def save(self, **kwargs: object) -> None:
            pass

        def dump(self, **kwargs: object) -> None:
            pass

    class Stages:
        def collect(self, target: str) -> list[Stage]:
            return [Stage()]

    class Repo:
        def __init__(self, root: str, *, _wait_for_lock: bool = False) -> None:
            nonlocal wait_for_lock
            wait_for_lock = _wait_for_lock
            self.stage = Stages()
            self.lock = nullcontext()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    dvc = types.ModuleType("dvc")
    dvc_repo = types.ModuleType("dvc.repo")
    dvc_repo.Repo = Repo
    monkeypatch.setitem(sys.modules, "dvc", dvc)
    monkeypatch.setitem(sys.modules, "dvc.repo", dvc_repo)

    commit_targets(tmp_path, tmp_path, ["dvc.yaml:stage"], no_commit=True)

    assert wait_for_lock is True
