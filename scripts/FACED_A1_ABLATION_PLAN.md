# FACED A1 Ablation Plan

This plan starts from the current best stable setting:

- Base config: `conservative` router profile
- PSD: off
- Backbone path: `AttnRes pre_attn`
- MoE: top-1 typed MoE on the last encoder layer

The immediate goal is not just to beat the current score. It is to separate three failure modes that can all look like a plateau in the logs:

1. The raw spatial router is collapsing, while capacity correction makes the assigned histogram look healthy.
2. The MoE modules are learning too slowly relative to the pretrained backbone and classifier.
3. `AttnRes` is feeding the router a signal that is too strong too early, so specialization never stabilizes.

## A1 Sweep

Use `ABLATION_TAG=A1` in [`submit_train.slurm`](/gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore/scripts/submit_train.slurm). The script will run these cases sequentially:

1. `A1_base`
   - Exact conservative no-PSD baseline.
   - Purpose: anchor the sweep to the known stable regime.

2. `A1_nomultilr`
   - Same as baseline, but with `--no-multi_lr`.
   - Hypothesis: routers, experts, and classifier are currently under-trained because new modules are below the backbone LR.

3. `A1_warmup4`
   - Same as baseline, but longer soft-routing warmup and slower jitter anneal.
   - Hypothesis: raw spatial routing is locking too early and never recovers.

4. `A1_gated`
   - Same as baseline, but enable `--attnres_gated --attnres_gate_init -1.0`.
   - Hypothesis: the raw baseline stream needs to remain more visible early so the router can learn a cleaner specialization signal.

## What To Read In The Outputs

For each run, compare the default routing summary files against the `_raw_top1_*` analysis files in the routing export directory:

- If `assigned` usage is balanced but `raw_top1` still collapses, capacity correction is rescuing the MoE more than the router itself.
- If `raw_top1` becomes less collapsed and validation/test improve, the spatial router was the main bottleneck.
- If `raw_top1` improves but accuracy stays flat, routing is healthier but the experts are not yet adding useful specialization.
- If train/val improve but test drops, the bottleneck is shifting toward generalization rather than routing.

The most important fields in the epoch logs are:

- `pre_hist`
- `pre_H`
- `pre_margin`
- `reroute_rate`
- validation/test `kappa`

## Decision Rule After A1

1. If `A1_nomultilr` wins:
   - Keep the higher effective LR for new modules and then retest mild router regularization changes.

2. If `A1_warmup4` wins:
   - Keep the longer warmup and test one small increase in balance/entropy regularization next.

3. If `A1_gated` wins:
   - Keep gated `AttnRes` and test whether the same benefit holds with a small LR change.

4. If none of the three beat `A1_base`, but raw routing still collapses:
   - The next target should be the spatial router inputs or the MoE placement itself, not broader regularization.
