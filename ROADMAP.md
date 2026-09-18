# Roadmap

This roadmap records the current direction for `klej`. Items under “Later”
are directions rather than commitments; they should be implemented only when
a concrete use case justifies them.

## Next — v0.1

### Operational confidence

* Harden actionable error reporting and compatibility tests around the pinned
  private DVC APIs.
* Complete release installation documentation and versioning. The current
  `0.0.1` package metadata remains pre-v0.1.

## Later

* Make an installed-package compute runner the default, retaining the
  repository shim as a development or fallback mechanism.
* Reduce reliance on pinned private DVC APIs and establish an explicit DVC
  version compatibility policy.
* Add safe garbage collection for content-addressed runner shims if their
  accumulation becomes material.
* Add supported Python configuration or monitoring APIs when a concrete
  caller requires them.

## Non-goals for now

* Local execution is a development and CI aid, not a promise of parity with
  Slurm or `dvc repro`.
* Container-specific integration such as first-class Apptainer configuration
  remains outside the package; the generic wrapper is the intended boundary.
* The inner runner command remains owned by `klej` rather than becoming public
  configuration.
