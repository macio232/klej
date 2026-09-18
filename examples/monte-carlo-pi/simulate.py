import argparse
import json
import random

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()

    generator = random.Random(args.seed)
    inside = sum(
        generator.random() ** 2 + generator.random() ** 2 <= 1
        for _ in range(args.samples)
    )
    result = {"inside": inside, "samples": args.samples, "seed": args.seed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
