# Examples

Each example is an independent DVC repository. Change into its directory
before running the documented commands.

## Monte Carlo pi

[`monte-carlo-pi`](monte-carlo-pi) runs four simulations concurrently and
combines their results into an estimate of pi. It is suitable for quick local
and Slurm smoke tests.

The idea is inspired by
[DAWGZ's Monte Carlo pi example](https://github.com/francois-rozet/dawgz/blob/master/examples/pi.py).
The implementation here is newly written as a DVC pipeline executed by klej.

## Alternative executors smoke test

[`alternative-executors-smoke`](alternative-executors-smoke) is an opt-in,
low-cost real-Slurm verification pipeline for competing profiles. It includes
successful and intentionally failing stages plus wrapper and payload markers
that demonstrate at-most-once entry and no fallback after winner failure.
