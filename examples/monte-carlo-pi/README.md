# Monte Carlo pi

This example estimates pi by sampling points in a unit square and counting
how many fall inside its quarter-circle. Four deterministic simulation stages
run independently, then one aggregation stage combines their JSON outputs.

```text
simulate@0 --\
simulate@1 ---\
simulate@2 ----> estimate
simulate@3 ---/
```

The example is an independent DVC repository and uses only the Python standard
library.

## Run locally

From this directory, inspect the plan and then execute it with four local
workers:

```console
klej repro --dry-run
klej repro --backend local --local-jobs 4
python -m json.tool outputs/pi.json
```

A completed pipeline is skipped on the next invocation. Pass `--force` to run
all stages again:

```console
klej repro --backend local --local-jobs 4 --force
```

## Run on Slurm

Review `klej.yaml` and add any account, partition, or other settings required
by your cluster. Then inspect and submit the workflow:

```console
klej repro --dry-run --force
klej repro --force
```

Submission creates five compute jobs followed by klej's commit and reservation
cleanup jobs. The dry run therefore reports seven logical and seven physical
jobs. Monitor them with `klej status`.

## Attribution

This example was inspired by
[DAWGZ's `pi.py` example](https://github.com/francois-rozet/dawgz/blob/master/examples/pi.py),
which demonstrates parallel Monte Carlo sampling followed by aggregation. The
code and DVC configuration here are a new implementation for klej.
