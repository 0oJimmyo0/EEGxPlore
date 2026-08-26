# ICASSP 2027 focused revision package

This directory is the active planning and provenance area for a focused revision of the rejected CBraMod selective-adaptation paper. The current scope is defined by [REVISION_CONTRACT.md](REVISION_CONTRACT.md) and executed in the order specified by [REVISION_PLAN.md](REVISION_PLAN.md).

## Active identity

The paper studies selective upper-layer adaptation with cross-depth residual reuse in CBraMod. It is an empirical revision of the rejected paper, not a new backbone or a second foundation-model study.

The active scope is deliberately limited to:

- CBraMod only;
- SEED-V, FACED, ISRUC, and PhysioNet-MI;
- the existing upper-layer selective adaptation, AttnRes, and specialist components;
- one matched benchmark protocol and a small, predeclared method ladder.

There is no new architecture in the active plan. Learned depth-conditioned routing, compact-context routing, PSD/context features, new expert decompositions, LaBraM, and broad hyperparameter sweeps are excluded from the primary result.

## Script status

Superseded DepthAgg and Static/Routed launchers, contract tests, and exploratory sweep scripts have been removed from the active tree. Their outputs and Git history remain available for provenance checks and recovery, but they are not valid commands for the current paper.

Retained utilities are limited to data/provenance checks and existing CBraMod training paths:

- `scripts/SEED-V/submit_seedv_train.slurm`
- `scripts/FACED/submit_train.slurm`
- `scripts/ISRUC/submit_train.slurm`
- `scripts/PHYSIO-MI/train_physio_compact_shared.slurm`
- `scripts/SEED-V/audit_seedv_lmdb_split.py`
- `scripts/SEED-V/build_seedv_subject_disjoint_manifest.py`
- `scripts/FACED/write_faced_channel_manifest.py`
- `experiments/icassp2027/scripts/audit_phase0.py`
- `experiments/icassp2027/scripts/extract_metadata.py`
- `experiments/icassp2027/scripts/generate_manifests.py`
- `experiments/icassp2027/scripts/frozen_probe.py`

The retained legacy launchers still contain historical defaults. Do not use them as the active experiment interface until the focused revision launcher described in `REVISION_PLAN.md` is added. In particular, do not aggregate results from `output/icassp2027_depth` into the main table.

## Separation rule

Do not import TMLR numerical results, figures, prose, checkpoints, or registries into this package. The TMLR study uses a different interaction-alignment question and estimand. Any reused ICASSP result must pass the artifact audit and have an independently recorded dataset, split, seed, code commit, checkpoint, selection rule, and metric provenance.
