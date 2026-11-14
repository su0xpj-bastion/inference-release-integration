# Release qualification strategy: inference core

## Ownership boundary
| Owner | Owns | Does not own |
|---|---|---|
| Feature team | design, feature-level qualification, feature regression | interactions with other teams' features |
| **RIT** | integration readiness, cross-stack validation, regression stability on main and release branches, first-pass rollout triage | fixing the defect (it routes the bug to the owning team with a repro) |

## Signal tiers
| Tier | Matrix | Budget | Blocks |
|---|---|---|---|
| Presubmit | pairwise covering array (15 cases) | < 2 min | merge to main |
| Nightly | 3-way covering array (24 cases) | < 10 min | promotion to release branch |
| Release | full matrix (192 cases), sharded | < 30 min | RC sign-off |

Coverage grows with distance from the developer. Each tier's blind spots are documented and pinned by tests (`test_detection_by_mode`), so a matrix change that drops a class of bug fails CI instead of failing silently.

## Risk scoring
`risk = layers crossed by the enabled features + fp16 + large shape`, which gives P0 (≥4), P1 (3) or P2.
Layers crossed counts because single-layer features already have an owning team's suite. fp16 counts for numerical headroom, and the large shape counts for memory pressure. A release is INCONCLUSIVE if any planned P0 case didn't run.

## Gates (all evaluated; decision = worst)
1. **Coverage:** every planned P0 case executed. Missing → INCONCLUSIVE.
2. **Signal quality:** infra + flaky rate ≤ 2%. Above that → INCONCLUSIVE. A noisy green isn't evidence.
3. **Correctness:** zero unwaived failures. Waivers live in `rit/known_issues.json`, each with an id and an owner.
4. **Regression:** zero new failures against the last shipped release.
5. **Performance:** no case more than +5% device cycles against the baseline.

## Adding a new inference-core feature to RIT
1. Map the feature to the layers it touches in `FEATURE_LAYERS`. That alone raises the risk of every case that enables it.
2. Add the flag to `FLAGS`. The covering arrays absorb it automatically, and the full matrix doubles.
3. If it has a numerical contract, decide its tolerance per dtype before writing tests, not after the first failure.
4. Add an end-to-end test that the harness *catches* a seeded version of its most likely integration bug. A harness that can't see a class of bug fails silently.

## Exit criteria for an RC
- The decision is GO from a `release`-mode run of the candidate's exact sha.
- Every bucket from earlier RCs has a fix commit, and that fix's own repro passes in this RC.
- The report artifact is attached to the release ticket.
