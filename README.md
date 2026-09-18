# klej

`klej` is the missing glue between
[DVC](https://dvc.org/) and [Slurm](https://slurm.schedmd.com/slurm.html). It
turns selected DVC stages into a dependency-aware [DAWGZ](https://github.com/francois-rozet/dawgz) workflow and
executes each stage as a separate Slurm job.

The name means “glue” in Polish.

## Installation

`klej` requires Python 3.11 or newer.

Install the pre-alpha release from PyPI as an isolated command with
[uv](https://docs.astral.sh/uv/):

```console
uv tool install klej==0.0.1
```

Alternatively, install it into the active Python environment with pip:

```console
python -m pip install klej==0.0.1
```

To install manually from a source checkout, create a virtual environment and
install the local project:

```console
git clone https://github.com/macio232/klej.git
cd klej
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

This is an early release intended for evaluation and research workflows, not
production use.

## Project status and safety

`klej` is an early-stage research tool. It may contain bugs, and its commands,
configuration, and Python interfaces may change without backward
compatibility. Do not rely on it for production-critical workloads without
independent testing.

Submitting a pipeline can create many jobs that run in parallel and may
consume substantial cluster resources or incur significant cost. Review the
plan with `--dry-run`, understand your cluster's quotas and billing, and
monitor submitted jobs. You are responsible for the workloads and costs
created by your use of `klej`; use it at your own risk.

## Usage

Run from a DVC repository:

```console
klej repro path/to/dvc.yaml --dry-run
klej repro path/to/dvc.yaml
klej repro [targets ...]
klej repro models/dvc.yaml:train --downstream
klej repro pipelines/ -R
klej status
klej status WORKFLOW_ID
klej status --failed
klej status --finished
klej logs WORKFLOW_ID JOB --tail 100 --follow
klej logs --clear WORKFLOW_ID
klej logs --clear-all --force
klej cancel WORKFLOW_ID --dry-run
klej lock status
```

Targets default to `./dvc.yaml` and use the same forms as `dvc repro`: stage
names, `dvc.yaml` and `.dvc` paths, `path/dvc.yaml:stage`, foreach or matrix
expansions, glob patterns with `--glob`, and directories with `-R`. DVC
dependencies in other pipeline files are included automatically.

Regular, foreach, matrix-expanded, frozen, import, `.dvc`, and
checkpoint-marked stages are supported according to pinned DVC 3.66
`dvc repro` behavior. Experiment checkpoint snapshots and resume behavior are
not provided.

The supported DVC selection and execution flags are `-f/--force`,
`-i/--interactive`, `-s/--single-item`, `-p/--pipeline`,
`-P/--all-pipelines`, `-R/--recursive`, `--downstream`,
`--force-downstream`, `--allow-missing`, `--glob`, `--no-run-cache`,
`--ignore-errors`, `--no-commit`, `--dry`, and `-v/--verbose`. `--dry-run` is
an alias for `--dry`. `--pull` and `-q/--quiet` are not supported.
`-k/--keep-going` is unnecessary: klej always continues independent Slurm
branches while failed dependencies prevent their downstream jobs from
running. `--ignore-errors` changes those dependencies from Slurm `afterok` to
`afterany`; failures remain visible in workflow status.

`-v/--verbose` reports repository and target resolution, selected and skipped
stages, effective configuration files, profiles, dependencies, runner and
reservation identifiers, and submitted Slurm IDs. It also enables DVC debug
logging and klej cache/metadata/lock diagnostics in job logs. Resolved
environment values are not printed.

An explicit `--config` is one complete configuration used for every selected
pipeline. Otherwise, repository-root `klej.yaml` is the base and a
`klej.yaml` beside each pipeline may partially override it. Named `profiles`
and `stages` are merged; local `commit`, `klej`, `environment`, and
`wrapper` values replace the base values. With one explicit configuration,
identically named stages in different pipeline files necessarily use the same
profile mapping. Multi-pipeline runs create one cleanup job, so all effective
pipeline configurations must resolve to the same klej-operation profile and
environment. Align those settings or use `--config` when they differ.

`klej cancel` queries Slurm and cancels only jobs that are still cancellable.
Its output reports counts rather than printing one line per job.

Dry-run output reports projected logical and physical job totals for the whole
workflow, including commit and reservation-cleanup jobs. A normal stage is one
logical and physical job. A Slurm stage with multiple profiles remains one
logical job but produces one physical attempt per profile plus a success gate;
its stage detail shows that expansion explicitly. No Slurm arrays are used.

```yaml
profiles:
  cpu:
    account: my-account
    cpus: 4
    mem: 16G
  gpu:
    account: my-account
    gres: gpu:1
    environment:
      APPTAINER_NV: 1

stages:
  prepare: cpu
  # Competing Slurm requests for one logical execution.
  train: [gpu, cpu]

commit: cpu
klej: cpu

environment:
  APPTAINER_NV: 0

# The wrapper receives the runner command as one final argument.
wrapper: [./apptainer/run.sh]
```

A stage may name one profile or an ordered, non-empty list of distinct
profiles. With two or more profiles on Slurm, klej submits one attempt per
profile; the first allocation to atomically claim the stage runs the wrapper
and payload. Other attempts exit with status 75 without entering either. The
claim is permanent, so a failing winner fails the logical stage and does not
fall back to another profile. At-most-once applies to entering the configured
wrapper and user payload, not to allocation: multiple Slurm allocations may
briefly start klej's coordinator. Correctness therefore requires a filesystem
shared by the submission host and every compute node, with atomic directory
creation and atomic file replacement. The local backend runs the stage once
with the first profile; scalar mappings and one-item lists retain the
single-job behavior.

Alternative attempts are submitted held. klej records every physical job ID
in the workflow plan and target reservation before releasing them. The
winning attempt best-effort cancels its siblings; reservation cleanup retries
active registered physical jobs other than itself, waits for Slurm's
configured `KillWait` plus five seconds after any cancellation, and confirms
they have left `squeue` before releasing the reservation. If the configuration
cannot be read, cleanup warns and uses Slurm's 30-second `KillWait` default. A
query or cancellation failure, or a job still present after the wait, fails
cleanup and retains the reservation for inspection and explicit lock
recovery. If submission or registration fails after `sbatch` may have run,
klej cancels all known jobs, including cleanup, and retains the target
reservation for explicit lock recovery. Once all physical IDs are registered,
a later submission failure cancels every other job but preserves cleanup so it
can release the reservation safely.

Version-2 workflow records are readable from their initial planned state, even
before any Slurm ID exists. Cancellation and submission recovery combine IDs
from the physical plan and the DAWGZ scheduler record so a partially persisted
submission does not leave a known job behind.

The opt-in
[`alternative-executors-smoke`](examples/alternative-executors-smoke)
example provides a low-cost real-cluster verification pipeline. It records
wrapper and payload entry for a successful race and an intentionally failing
winner, with commands for inspecting logical and physical status and logs.
Cluster policies and accounting differ, so review its profiles, quotas, and
costs before submitting it.

## Commit modes

By default, `klej repro` creates one commit job after all leaf compute jobs
finish. That job commits the selected `dvc.yaml` using the `commit` profile
from `klej.yaml`.

Three mutually exclusive flags provide alternative metadata placement:

* `--commit-per-stage` runs `dvc commit` for each stage target inside its
  compute job. The commit therefore uses the stage's profile.
* `--separate-commit-jobs` creates a dedicated commit job after each compute
  job. These jobs use the `commit` profile, and downstream computation waits
  for the upstream commit job to finish.
* `--no-dvc-metadata` executes stages without updating DVC metadata or caches.

Independently, DVC-compatible `--no-commit` still updates `dvc.lock` but does
not cache outputs or create run-cache entries.

```console
# One final commit job (default)
klej repro path/to/dvc.yaml

# Commit each target in its compute job
klej repro path/to/dvc.yaml --commit-per-stage

# Commit each target in a separate job
klej repro path/to/dvc.yaml --separate-commit-jobs

# Update dvc.lock without caching outputs
klej repro path/to/dvc.yaml --no-commit

# Do not update DVC metadata
klej repro path/to/dvc.yaml --no-dvc-metadata
```

All commits are serialized through klej's repository-wide commit lock and
invoke `dvc commit --force`.

## Execution environment

klej installs an immutable, standalone compute runner under `.klej/runners/`
in the directory where the command is run and keeps shared coordination data
under `.klej/locks/` there. Add `.klej/` to the consumer repository's ignore
rules. The repository must be visible at the same path to submission and
compute nodes. Containers need Python and the pinned DVC version, but do not
need klej installed.

## Wrapper examples

Environment values can refer to submission-side variables as `$NAME` or
`${NAME}`. A profile's environment is applied after the global environment.
The wrapper runs from the DVC repository root and is used only with Slurm.

Set `wrapper` to an executable that prepares the compute-node environment and
runs its final argument as a shell command. For example, put one of the scripts
below in the repository and reference it from `klej.yaml`:

```yaml
wrapper: [./wrappers/run.sh]
```

Make the script executable with `chmod +x wrappers/run.sh`. The environment
must provide Python and the pinned DVC version.

An Apptainer wrapper can bind the repository at the same path inside the
container:

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="$(pwd -P)"
readonly IMAGE="${PROJECT_DIR}/containers/project.sif"

[[ -f "${IMAGE}" ]] || {
  printf 'container image not found: %s\n' "${IMAGE}" >&2
  exit 1
}

exec apptainer exec \
  --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
  --pwd "${PROJECT_DIR}" \
  "${IMAGE}" bash -c "$1"
```

For a Conda environment:

```bash
#!/usr/bin/env bash
set -euo pipefail

exec conda run --no-capture-output -n my-environment bash -c "$1"
```

For a virtual environment stored in the repository:

```bash
#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
exec bash -c "$1"
```

The equivalent wrapper using uv is:

```bash
#!/usr/bin/env bash
set -euo pipefail

exec uv run -- bash -c "$1"
```

To load environment modules provided by the cluster:

```bash
#!/usr/bin/env bash
set -euo pipefail

module load python/3.12 cuda/12.4
exec bash -c "$1"
```

Replace the example environment names, image path, and module versions with
those available on the cluster. Add `--nv` to `apptainer exec` when the
container needs NVIDIA GPU access.

## Local backend

`--backend local` is intended for development and CI validation of klej's
planning and execution path. It is not a full alternative to `dvc repro` and
does not promise Slurm backend parity.

## Examples

Runnable pipelines are available in [`examples`](examples/README.md). Each
example is an independent DVC repository with its own `dvc.yaml` and
`klej.yaml`.

## Workflow status

`klej status` lists active Slurm workflows recorded in the current `.dawgz`
directory. A workflow is active while any job is pending or running. The
summary shows the workflow ID, DVC pipeline, and counts of pending, running,
succeeded, failed, cancelled, and unknown jobs. Counts include compute,
commit, and reservation-cleanup jobs. Local-backend workflows are omitted.

`klej status WORKFLOW_ID` prints the aggregate state and every DAWGZ job in
stable zero-based order. It deliberately does not truncate or page large
foreach or matrix workflows. New compute jobs are identified by their exact
expanded DVC target, such as `dvc.yaml:simulate@0`; internal jobs retain their
descriptive DAWGZ name. Older records without exact-target metadata remain
readable using their recorded names. The `START TIME` column contains Slurm's
unmodified accounting start date-time, or `-` when it is unavailable. Rows
with known start times are ordered from oldest to newest; missing start times
appear last in stable DAWGZ order.

For alternative resource profiles, status shows one logical row and summary
count per stage. The winning attempt and success gate determine that row;
losing attempts do not count as failed workloads. Cancelling a claimed stage
is reported as cancelled unless its payload failure was already recorded or
the claimant failed in Slurm. Use
`klej status WORKFLOW_ID --physical` to append every attempt, gate, commit, and
cleanup job with its raw Slurm state and exit code. Losing siblings appear as
`superseded`. Workflow terminality and safe record clearing still account for
all physical jobs, including rows hidden from the default logical view.

`klej status --failed` lists terminal Slurm workflows containing at least one
failed or cancelled job. A workflow containing an unknown job is not known to
be terminal. Slurm accounting has a cluster-defined retention window: once a
job disappears from `sacct`, klej reports it as unknown, so sufficiently old
failed workflows may no longer appear in this listing.

`klej status --finished` lists every recorded workflow known to be terminal,
including local-backend records, with an aggregate status of `succeeded`,
`failed`, or `cancelled`. Its `SUBMISSION` column contains the DAWGZ workflow
submission date-time, and rows are ordered from oldest to newest. Legacy
records without that timestamp show `-` after timestamped workflows. Pending,
running, and unknown workflows are omitted.

Slurm states are normalized to pending, running, succeeded, failed, cancelled,
or unknown while the complete scheduler value remains visible in detailed
status. For example, `COMPLETED+` is succeeded, `CANCELLED by 12345` is
cancelled, and timeout, out-of-memory, and node failures are failed.

## Job logs

`klej logs WORKFLOW_ID JOB` writes the currently available DAWGZ log directly
to standard output without a heading. `JOB` may be the zero-based index from
detailed status, an exact Slurm ID, or the exact displayed target/name when it
is unique. Internal jobs, including separate commit jobs, use their job names
as identities. For an alternative-profile stage, its target, name, or logical
index selects the attempt that claimed the race, including a claimant whose
payload failed. An exact Slurm ID instead selects that physical job directly,
so superseded attempts and internal gate, commit, or cleanup jobs remain
inspectable. Exact Slurm IDs take precedence over logical indexes. Ambiguous
selectors are rejected with the matching indices.

Use `--tail N` to start with the last `N` lines and `--follow` to stream new
content. Combining them follows from the tailed content; without `--tail`,
following starts with the complete existing log. If an active job has not
created its log yet, follow mode waits and reports that once on standard error.
Following stops after the job becomes terminal and all available bytes have
been emitted. Submission tracebacks recorded by DAWGZ are printed when no
Slurm log was created. If an alternative race is unresolved, a normal request
lists its attempt IDs, profiles, and states. Follow mode waits while an attempt
can still claim the race and then follows the claimant. When accounting has no
state for an attempt, follow mode checks whether it remains in the Slurm queue
before waiting. It does not choose an attempt merely because its log appeared
first. Detailed status and logs are not supported for local-backend workflow
records.

`klej logs --clear WORKFLOW_ID` removes a known-terminal workflow's complete
DAWGZ record, including its logs, and removes its rows from `workflows.csv`.
`klej logs --clear-all` does the same for every known-terminal Slurm and local
workflow while retaining pending, running, and unknown records. Both commands
ask for confirmation; `--force` skips that prompt but never permits deletion
of an active or unknown workflow. For versioned plans, this safety check
includes every physical attempt and internal gate, commit, and cleanup job,
including jobs hidden by the logical status view. Cleanup does not cancel
jobs, release klej locks, or delete DVC outputs.

Do not run record cleanup concurrently with workflow submission. DAWGZ does
not coordinate updates to its `workflows.csv` index with external cleanup.
