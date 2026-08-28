# ICASSP 2027 focused revision package

This directory is the active planning and provenance area for a focused revision of the rejected CBraMod selective-adaptation paper. The current scope is defined by [REVISION_CONTRACT.md](REVISION_CONTRACT.md) and executed in the order specified by [REVISION_PLAN.md](REVISION_PLAN.md).

## Active identity

The paper studies robust selective adaptation of CBraMod with AttnRes and
typed specialists. It is an empirical revision of the rejected paper, not a
new backbone or a second foundation-model study.

The active scope is deliberately limited to:

- CBraMod only;
- SEED-V, FACED, and ISRUC;
- the existing upper-layer selective adaptation, AttnRes, and specialist components;
- one matched benchmark protocol and a small, predeclared method ladder.

PhysioNet-MI, TUEV, and the subject-disjoint SEED-V protocol are archived
evidence only. They are not part of the active ICASSP compute plan. The new
paper-facing rows are SEED-V Upper-1 and matched Dense/Selective rows on FACED
and ISRUC, using seeds `42`, `3407`, and `2024`.

There is no new architecture in the active plan. Learned depth-conditioned routing, compact-context routing, PSD/context features, new expert decompositions, LaBraM, and broad hyperparameter sweeps are excluded from the primary result.

The existing `selective_fresh` condition is a separately locked independent
recipe and remains unchanged. New paper-derived selective runs use the
explicit `selective_paper` executable condition, which shares the same
AttnRes-plus-typed-specialist implementation but resolves dataset-specific
paper-derived training protocols. `historical_selective` is an archival label
for the unrecoverable rejected-paper family and is permanently locked in the
active profile. The old `MoE-only` logs
already used `pre_attn` AttnRes and must not be relabeled as `specialist_only`.

Legacy SEED-V values reported in the rejected manuscript are eligible only as
clearly labeled `legacy_reported_paper` evidence. They are not current
reproductions and must not be combined into pooled statistics with new rows.

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
- `experiments/icassp2027/revision/run_revision.sh`
- `experiments/icassp2027/revision/submit_revision.slurm`
- `experiments/icassp2027/revision/verify_paper_protocol.py`
- `experiments/icassp2027/revision/paper_protocol_*.json`
- `experiments/icassp2027/revision/paper_table_manifest.csv`
- `experiments/icassp2027/revision/audit_paper_scope.py`
- `experiments/icassp2027/revision/audit_revision_config.py`
- `experiments/icassp2027/revision/verify_data_contract.py`
- `experiments/icassp2027/revision/verify_fresh_selective_recipe.py`
- `experiments/icassp2027/revision/fresh_selective_recipe.json`
- `experiments/icassp2027/revision/build_evidence_registry.py`
- `experiments/icassp2027/revision/HISTORICAL_RECIPE_AUDIT.md`
- `experiments/icassp2027/revision/historical_recipe_1785556.json`
- `experiments/icassp2027/revision/verify_historical_recipe.py`
- `experiments/icassp2027/revision/audit_historical_bundle.py`
- `experiments/icassp2027/revision/historical_candidates.csv`
- `experiments/icassp2027/revision/test_*.py`

The retained legacy launchers still contain historical defaults. Use only the
focused revision launcher for paper-facing runs. In particular, do not
aggregate results from `output/icassp2027_depth` into the main table. Smoke
runs use `output/icassp2027_smoke` and are permanently ineligible for the
paper registry.

## Separation rule

Do not import TMLR numerical results, figures, prose, checkpoints, or registries into this package. The TMLR study uses a different interaction-alignment question and estimand. Any reused ICASSP result must pass the artifact audit and have an independently recorded dataset, split, seed, code commit, checkpoint, selection rule, and metric provenance.
