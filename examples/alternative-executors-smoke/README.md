# Alternative executors: real-Slurm smoke test

This independent DVC repository exercises klej's multi-profile execution on a
real Slurm cluster. It uses two intentionally equivalent, low-cost contender
profiles. Edit `klej.yaml` first if your cluster requires an account,
partition, QoS, or different minimum resources.

`wrapper.sh` searches the current directory and its ancestors for
`.venv/bin/python`, adds the first match to `PATH`, and verifies that it can
import DVC. This supports running the example from this repository's standard
`.venv` on a shared filesystem. If compute nodes use modules, Conda, or a
container instead, replace that environment setup in `wrapper.sh` before
submitting.

The example records entry into both the configured wrapper and the user
payload. Run one case at a time from this directory, using
`--no-dvc-metadata` so no commit job enters the wrapper and affects its marker
count.

## Successful winner

Start clean and inspect the expansion:

```console
rm -rf markers outputs
klej repro success --force --no-dvc-metadata --dry-run
```

The plan should report two logical jobs and four physical jobs: two attempts,
one success gate, and one cleanup job. Submit it and retain the printed
workflow ID:

```console
klej repro success --force --no-dvc-metadata
klej status WORKFLOW_ID --physical
wc -l markers/wrapper.log markers/success.log
```

After completion, both marker files must contain exactly one line. Status must
show one winning attempt, one superseded sibling, and a successful logical
stage. View the claimant by logical target, then copy the sibling's Slurm ID
from physical status and inspect it directly:

```console
klej logs WORKFLOW_ID dvc.yaml:success
klej logs WORKFLOW_ID LOSER_SLURM_ID
```

## Failed winner without fallback

The second stage intentionally exits with status 23 after recording payload
entry:

```console
rm -rf markers outputs
klej repro failure --force --no-dvc-metadata
klej status WORKFLOW_ID --physical
wc -l markers/wrapper.log markers/failure.log
```

Both marker files must still contain exactly one line. The claimant and the
logical stage must fail, and the sibling must not run the wrapper or payload.
Its exact Slurm ID remains selectable with `klej logs`.

Multiple allocations may briefly start the coordinator. The at-most-once
guarantee applies to wrapper and payload entry and depends on atomic directory
creation on the filesystem shared by the submission host and compute nodes.
Review cluster quotas and costs before submitting this example.
