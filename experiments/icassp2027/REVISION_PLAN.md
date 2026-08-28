# Focused ICASSP implementation plan

This plan freezes a small evidence package around the rejected-paper identity.
It does not reopen the broader routing or four-dataset study.

## 1. Freeze provenance artifacts

Maintain the row-level registry under `output/icassp2027_revision/`. New rows
are `new_multiseed`; manuscript rows are either `legacy_reported_paper` or
`legacy_context_only`. The registry must distinguish manuscript-reported
values from independently reconstructed runs and must retain TMLR-overlap
review fields.

The manuscript-reported SEED-V Dense, AttnRes-only, and MoE/selective rows use
seeds `42`, `3407`, and `2024`. Their weighted-F1 values remain explicitly
weighted-F1, not macro-F1. The uniform comparison therefore uses balanced
accuracy, weighted-F1, and kappa; new-run macro-F1 remains recorded as an
additional metric. Missing checkpoints, commits, and raw logs remain unknown.

## 2. Use locked paper-derived protocols

The existing `fresh_selective_recipe.json` is preserved unchanged as an
independent archived recipe. New paper-facing selective runs use
`selective_paper` plus the dataset protocol file for SEED-V, FACED, or ISRUC.
The protocol verifier records a hash and validates the supported execution
fields. It does not claim that unrecovered historical fields are known.

## 3. Paper launcher modes

`run_revision.sh` defaults to `RUN_MODE=paper` and accepts only the active
datasets, primary benchmark protocol, declared seeds, and the frozen new-run
matrix:

- SEED-V / `upper1`;
- FACED / `full` and `selective_paper`;
- ISRUC / `full` and `selective_paper`.

`RUN_MODE=smoke` uses one epoch, the same architecture/data path, and the
separate `output/icassp2027_smoke/` root. Smoke artifacts are explicitly
ineligible for paper aggregation. Historical conditions remain blocked.

## 4. GPU budget

Minimum: 9 new jobs — SEED-V Upper-1 ×3 and ISRUC Dense/Selective ×3.

Preferred: 15 new jobs — add FACED Dense/Selective ×3. Do not launch a broad
condition or seed grid before these rows are reviewed.

## 5. Reporting

The main table reports dataset-level values only. It does not pool legacy
SEED-V values with new FACED/ISRUC values. The SEED-V component table labels
legacy rows and the new Upper-1 control separately. Per-seed metrics, protocol
hashes, efficiency logs, and smoke/failure notes belong in the artifact or
supplement.

## Stop conditions

Stop after the planned new blocks are complete and the efficiency summary is
available. Do not add TUEV, PhysioNet-MI, LaBraM, a second backbone, subject-
disjoint primary experiments, new routing designs, or broad sweeps merely to
rescue a weak result.
