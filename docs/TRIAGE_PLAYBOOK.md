# First-pass triage playbook

The goal is that within an hour of a red release run, every failure sits in a bucket with an owner and a one-line repro, or is explicitly marked as unknown.

## 1. Is the run trustworthy?
Check the signal-quality gate first. If infra + flaky exceeds 2%, the run is INCONCLUSIVE. Fix the fleet or re-run before reading any product result.

## 2. Read the buckets, not the failures
`python -m rit triage results/*.jsonl`. N red cases should read as a few buckets. If you get one bucket per case, localization failed: look at an `unknown` bucket's representative by hand.

## 3. Interpret the localization

| Diagnosis | Meaning | Usual owner |
|---|---|---|
| `non-finite output` / `output numerically wrong with correct inputs` | The kernel got the right inputs and produced the wrong output | kernels |
| `stale prefetched read` | The value was correct in SRAM, but the runtime read it before its producer wrote it | runtime |
| `input clobbered in SRAM after write` | The producer wrote the right value, and something overwrote the slot before the consumer read it | compiler (memory planner) |
| `compile error` | The program didn't build | compiler |
| `infra` | A device timeout persisted through retries | core-infra |

## 4. Bisect, then compare layers
`python -m rit bisect --case <repro> --good <last release> --bad <rc>`.
If the first bad commit's component **differs** from the localized layer, the commit **exposed** a latent defect. Route the ticket to the localized layer's owner, and copy the exposing commit's author because they'll need to know about the revert-or-wait decision.

## 5. File it
Paste the bucket row and the representative repro. Add the minimal trigger, which goes straight into the owning team's unit suite as a regression test. Then add a RIT end-to-end test asserting that the harness catches this class of bug.

## Anti-patterns
- Retrying a product failure "to see if it's flaky". The runner retries only infra, so a product failure that passes on retry is a real nondeterminism bug.
- Bucketing by error message. Two bugs can share `max relative error > tolerance`, and one bug can show up under three different messages.
- Routing by bisect alone. That's how B3 ends up with the compiler team.
