import hashlib
import os
import uuid

from pathlib import Path

from . import runner


def install_runner(execution_root: Path) -> Path:
    source = Path(runner.__file__).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    directory = execution_root / ".klej" / "runners" / digest
    path = directory / "runner.py"
    if path.exists():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".runner.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(source)
    os.replace(temporary, path)
    return path
