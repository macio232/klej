import json
import os
import socket
import sys

from datetime import UTC, datetime
from pathlib import Path


def observation(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    mode = sys.argv[1]
    if mode not in {"success", "failure"}:
        raise ValueError(f"unknown mode {mode!r}")
    record = observation(mode)
    markers = Path("markers")
    markers.mkdir(exist_ok=True)
    with (markers / f"{mode}.log").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    if mode == "failure":
        return 23
    output = Path("outputs/success.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
