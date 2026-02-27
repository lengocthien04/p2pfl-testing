# Run 10 Analysis (2026-02-27)

## Setup
- Config: cifar10_dclique_cliqueavg.yaml
- 24 nodes, dclique_4 topology, neighbor-only aggregation, DSGDCliqueAvg
- Seed: 666, log: /workspace/p2pfl-testing/logs/run-10.log
- Started: 14:09 UTC

## Round Progress (as of ~14:34 UTC)
- Round 1: All 24/24 nodes completed aggregation
- Round 2: All 24/24 nodes completed aggregation
- Round 3: ~10 nodes completed aggregation, rest still in gossip/collection
- Aggregation consistently shows: "Clique averaging: 4 clique models, 5 total neighbor models"
- All nodes collect 24/24 models before aggregating — no partial collections observed

## Late Round Errors (harmless)
- ~199 "Vote received in a late round" errors
- ~213 "Models ready from X in a late round" errors
- These are stale gossip messages from slower nodes arriving at faster nodes that have already advanced rounds
- Messages are correctly ignored; they do NOT affect model collection or aggregation

## Gossip Timeout Overhead
- Most nodes hit the hard timeout (~300-345s) per gossip phase
- This is consistent with previous runs (expected for 24-node topology)

## Concerning: Evaluation Results

### Latest eval per node (sorted by test_metric)

| Node | Round~ | test_loss | test_metric (acc) |
|------|--------|-----------|-------------------|
| 42047 | ~2 | 1.868 | 0.805 |
| 55879 | ~2 | 2.009 | 0.744 |
| 36143 | ~2 | 1.801 | 0.689 |
| 43151 | ~2 | 1.508 | 0.658 |
| 60429 | ~3 | 2.936 | 0.287 |
| 41665 | ~2 | 5.624 | 0.076 |
| 44149 | ~2 | 2.976 | 0.035 |
| 47845 | ~3 | 2.351 | 0.025 |
| 53173 | ~2 | 8.887 | 0.023 |
| 41663 | ~3 | 2.621 | 0.019 |
| 43193 | ~3 | 2.471 | 0.015 |
| 50219 | ~2 | 1.897 | 0.003 |
| 39099 | ~3 | 3.013 | 0.002 |
| 43257 | ~3 | 2.151 | 0.0 |
| 39739 | ~3 | 1.971 | 0.0 |
| 52203 | ~3 | 3.361 | 0.0 |
| 59299 | ~3 | 3.012 | 0.0 |
| 45609 | ~3 | 3.184 | 0.0 |
| 35677 | ~2 | 2.968 | 0.0 |
| 54227 | ~2 | 2.831 | 0.0 |
| 60461 | ~2 | 3.352 | 0.0 |
| 43315 | ~2 | 3.150 | 0.0 |
| 36941 | ~2 | 5.845 | 0.0 |
| 50271 | ~2 | 9.824 | 0.0 |

### Issues Observed

1. **13/24 nodes (54%) have 0.0% accuracy** after 2-3 rounds of training.

2. **Metrics appear frozen across rounds** — same value from round 0 through round 3:
   - 60429: stuck at 0.287 (rounds 1, 2, 3)
   - 47845: stuck at 0.025 (rounds 0, 1, 2, 3)
   - 41665: stuck at 0.076 (rounds 0, 1, 2)
   - 39099: stuck at 0.002 (rounds 1, 2, 3)
   This suggests the metric may not be properly reflecting updated model weights.

3. **Losses diverging for some nodes** instead of decreasing:
   - 50271: 2.32 → 3.75 → 9.82
   - 53173: 2.31 → 2.34 → 8.89
   - 47845: 2.31 → 21.15 → 2.29 → 2.35
   - 45609: 2.32 → 13.18 → 3.18

4. **Only 4 nodes show meaningful accuracy** (>50%): 42047, 55879, 36143, 43151.

### Possible Root Causes to Investigate

- **Metric reset issue**: The "Resetting metric state" happens before eval, but frozen metrics suggest the evaluator may not be using the newly aggregated model.
- **Aggregation producing degenerate weights**: DSGDCliqueAvg may be averaging in a way that destroys learned features for most nodes, while a few nodes (perhaps in the same clique) retain useful weights.
- **Non-IID data distribution**: With 24 nodes and CIFAR-10, each node may have very skewed class distributions, causing divergent optimization directions that the clique averaging cannot reconcile.
- **Learning rate / training hyperparams**: Early rounds with high loss divergence could indicate learning rate is too high for the aggregation scheme.
- **Too early to judge**: Only 2-3 rounds completed. Some federated learning algorithms need more rounds to converge. However, the frozen metrics and loss divergence are still red flags worth monitoring.

## Deep-Dive Code Analysis (What is actually happening)

### 1) This does **not** look like a metric-reset bug

I traced the eval path in code:

- `TrainStage.execute()` evaluates at the start of each round (`TrainStage.__evaluate`) and then trains.
- At the end of previous round, each node does `learner.set_model(agg_model)` after aggregation.
- `Learner.set_model()` explicitly resets metric state (`pt_model.metric.reset()`).
- `LightningLearner.evaluate()` runs `trainer.test(...)` on the current model object.

So evaluation is reading the current model state, and metric reset is already wired in.

### 2) Why many metrics can still look "frozen" or near-zero

The bigger issue is the data/eval setup under strong non-IID:

- Config uses `DirichletPartitionStrategy` with `alpha: 0.1` across **24 nodes**.
- In `DirichletPartitionStrategy.generate_partitions()`, train and test are partitioned **independently** (two separate `_partition_data(...)` calls).
- That means a node's local train distribution and local test distribution can be very mismatched.

With alpha=0.1, this mismatch can be extreme. A node can train on one label-skewed shard and be evaluated on a different skewed shard, producing persistent very low/zero local accuracy even if training is functioning.

### 3) Clique averaging is very aggressive in this topology

For your observed case (`4 clique models, 5 total neighbor models`):

- Stage 1: average 4 clique models.
- Stage 2: average `[clique_avg] + neighbor_models_no_self` (4 terms) => 5 terms total.

Effective weight per round (clique size 4, degree 4):

- self model: **0.05**
- each clique neighbor: **0.25**
- non-clique neighbor: **0.20**

So self influence is tiny each round, and mixing pressure is very high. Combined with non-IID alpha=0.1 and `lr_rate=0.1`, instability/divergence is expected for some nodes.

### 4) There is also a design inconsistency in `DSGDCliqueAvg`

The class docstring says stage 2 should mix clique result with **non-clique** neighbors, but implementation currently mixes with all neighbors except self (which still includes clique neighbors individually). This is not necessarily a crash bug, but it changes weight dynamics a lot.

## What is wrong (ranked)

1. **Evaluation protocol is misleading for this experiment**
   - Independent train/test Dirichlet partitioning at alpha=0.1 causes per-node metric distortion.

2. **Optimization is too aggressive for this heterogeneity**
   - `lr=0.1` with one-hop neighbor mixing and heavy clique weighting is unstable.

3. **CliqueAvg stage-2 weighting likely over-amplifies clique neighbors**
   - Self signal is heavily diluted each round.

## Solution to apply (recommended order)

### Immediate (must do first)

1. **Fix evaluation meaning**:
   - Either evaluate all nodes on one shared global test set, or
   - make each node's test split derived from its own train partition (same client distribution).

Without this, node-level accuracy comparisons are not trustworthy.

### Stabilize training next

2. **Lower LR** in `cifar10_dclique_cliqueavg.yaml`:
   - from `0.1` -> `0.01` (or `0.02` as intermediate test).

3. **Reduce heterogeneity for debugging**:
   - temporarily raise Dirichlet `alpha` from `0.1` -> `0.5` (or `1.0`) to verify algorithmic stability.

### Algorithm adjustment (if still unstable)

4. **Adjust `DSGDCliqueAvg` stage-2 mixing** to avoid over-diluting self signal.
   - Option A: mix `clique_avg` with only non-clique neighbors.
   - Option B: keep current structure but introduce explicit weights (do not uniform-average stage 2).

## Bottom line

Infrastructure/gossip is progressing now (that part is fixed). The current bad metrics are most likely from **(a) misleading local evaluation under extreme non-IID partition mismatch** plus **(b) unstable optimization settings (high LR + aggressive mixing)**, not from a broken metric reset path.
