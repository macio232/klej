import argparse
import json

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    shards = [json.loads(path.read_text()) for path in args.inputs]
    inside = sum(shard["inside"] for shard in shards)
    samples = sum(shard["samples"] for shard in shards)
    result = {
        "inside": inside,
        "pi": 4 * inside / samples,
        "samples": samples,
        "shards": len(shards),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
