# Focused ICASSP implementation plan

This plan freezes a small evidence package around a simplified revision of the
rejected-paper identity. It does not reopen the broader routing or
four-dataset study.

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

## 2. Freeze the final method

The paper-facing method is `specialist_augmented_full`, defined by
`paper_method_specialist_augmented_full_v1.json`:

- full CBraMod trainability;
- `pre_attn` AttnRes from layer 0;
- one top-layer typed spatial/spectral specialist block;
- four experts and the ordinary typed router;
- no PSD, compact-feature, or depth-aware router features;
- no component-wise learning-rate scaling.

The method recipe is separate from the dataset-specific optimization
protocols. The development-only historical candidates remain unchanged and
are never relabeled as final paper evidence.

## 3. Use locked paper-derived protocols

The existing `fresh_selective_recipe.json` is preserved unchanged as an
independent archived recipe. New paper-facing selective runs use
`selective_paper` plus the dataset protocol file for SEED-V, FACED, or ISRUC.
The protocol verifier records a hash and validates the supported execution
fields. It does not claim that unrecovered historical fields are known.

## 4. Paper launcher modes

`run_revision.sh` defaults to `RUN_MODE=paper` and accepts only the active
datasets, primary benchmark protocol, declared seeds, and the frozen new-run
matrix:

- FACED / `full` and `specialist_augmented_full`;
- ISRUC / `full` and `specialist_augmented_full`.

The final manifest uses seeds `3407`, `2024`, and `2027`. Seed `42` is
development-only for the final method. Historical SEED-V rows remain in the
legacy manifest and are not pooled with the final 12 rows.

`RUN_MODE=smoke` uses one epoch, the same architecture/data path, and the
separate `output/icassp2027_smoke/` root. Smoke artifacts are explicitly
ineligible for paper aggregation. Historical conditions remain blocked.

## 5. GPU budget

Final confirmatory block: 12 jobs — FACED and ISRUC, Full and
AttnRes + Typed Specialists, over seeds `3407`, `2024`, and `2027`.

Run one one-epoch smoke for the new alias first. Do not launch depth-routing
multiseeds, broad sweeps, replacement seeds, or additional architectures.

## 6. Reporting

The main table reports FACED/ISRUC Full versus AttnRes + Typed Specialists as
mean ± standard deviation over the three unseen seeds. The seed-42 opt-only
and depth-routing comparisons are development/component evidence. The SEED-V
component table labels legacy rows separately. Per-seed metrics, recipe
hashes, protocol hashes, efficiency logs, and smoke/failure notes belong in
the artifact or supplement.

## 7. Stop conditions

Stop after the planned new blocks are complete and the efficiency summary is
available. Do not add TUEV, PhysioNet-MI, LaBraM, a second backbone, subject-
disjoint primary experiments, new routing designs, or broad sweeps merely to
rescue a weak result.
