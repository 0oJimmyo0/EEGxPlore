# Minimal-effort implementation plan

This plan turns the revision contract into a small number of auditable artifacts. The order matters: do not launch a broad new sweep before the old artifacts and baseline protocol are understood.

## 1. Build the artifact-reuse registry

Create one registry under `output/icassp2027_revision/` with one row per candidate run. Use
`revision/build_evidence_registry.py` to regenerate it from immutable run artifacts. Record:

`dataset, condition, seed, split, preprocessing, epoch budget, selection rule, code commit, checkpoint path, checkpoint hash, metric files, trainable parameters, runtime, GPU, TMLR-overlap status, reuse decision, notes`.

Use the following decisions:

- `verified`: complete and protocol-matched; eligible for the main table;
- `candidate`: potentially useful but missing a provenance field; do not aggregate;
- `rerun`: useful code path or result direction, but protocol mismatch prevents reuse;
- `supplement`: valid diagnostic with a different protocol;
- `forbidden`: TMLR-derived or otherwise out of scope.

The generated registry defaults every new row to `unreviewed_pending_row_audit`
and `candidate_pending_audit`; a complete run is not automatically promoted to
the paper table. Historical rows must be entered only after the separate
`revision/HISTORICAL_RECIPE_AUDIT.md` checklist is complete. The tracked
`revision/historical_candidates.csv` index is intentionally seed-level and
candidate-only; run `revision/audit_historical_bundle.py --strict` before
claiming that the three historical SEED-V families are matched.

The current `output/icassp2027_depth` pilots are supplemental candidates only. The old ISRUC non-frozen jobs that ended in OOM are invalid evidence. Existing logs can save compute, but only after this row-level audit.

## 2. Add one focused launcher

Use the single launcher under `experiments/icassp2027/revision/` that calls the existing CBraMod training entry point. It takes dataset, condition, seed, split/protocol, GPU count, and output directory through explicit arguments/environment variables and rejects forbidden datasets/backbones and historical output roots. The `historical_selective` label is guarded until the `1785556` recipe audit is complete.

The launcher should create a self-contained run manifest before training and a result manifest after training. It must not implement a new model. It should expose the named conditions in `REVISION_CONTRACT.md`, with the depth-independent specialist condition as the default candidate and any depth-enabled condition behind an explicit supplemental flag.

For `historical_selective`, `audit_revision_config.py` also verifies the
resolved arguments against `revision/historical_recipe_1785556.json` and records
the recipe hash. The recipe remains pending until the historical bundle audit
recovers all required values.

Before submitting a batch, run one smoke job per dataset with the smallest valid budget and verify: data loading, label counts, validation-only selection, checkpoint creation, metric extraction, and output isolation.

## 3. Re-establish the matched baseline first

Priority order for compute:

1. SEED-V under the original rejected-paper benchmark protocol;
2. FACED;
3. PhysioNet-MI;
4. ISRUC after resolving the known memory/resource issue.

For each dataset, run Frozen + head, Full FT, and AttnRes-only first. These are the minimum baselines needed to determine whether the observed gains are real and whether a dataset is wired correctly. Then run Specialist-only and the combined condition using exactly the same split, selection rule, budget, and seeds. Use `historical_selective` only for the audited historical-recipe reproduction; use `combined` for the implementation control before that audit.

Promote a dataset to the main table only when the core conditions have a complete matched three-seed block. A weak result is still useful; an unmatched result is not.

## 4. Add only high-value supplemental evidence

After the main block is healthy, run the smallest controls that directly address reviewer concerns:

- SEED-V subject-disjoint: Frozen, Full FT, and the combined method, reported separately;
- one compact generic adaptation control if LoRA/generic/axis-blind artifacts are not already valid and cheap to reproduce;
- one no-depth versus existing depth-enabled comparison, clearly labeled as supplemental.

Do not add TUEV only to satisfy a reviewer request if doing so creates a large per-class analysis burden. Do not add LaBraM, a new operator, a new router, or a broad expert/depth sweep.

## 5. Assemble the four-page evidence package

The main paper needs only:

- one compact table with the matched core conditions across the four datasets;
- mean ± standard deviation for balanced accuracy and macro-F1;
- one small parameter/runtime or trainable-capacity summary;
- a concise ablation or supplement showing what AttnRes and specialist capacity each contribute.

Move per-seed values, split details, subject-disjoint results, failure/OOM notes, and extended diagnostics to supplementary material or the artifact repository. The paper should claim robustness only over the completed matched blocks, not over every historical run.

## Stop conditions

Stop adding experiments when the four-dataset core table is complete, the main uncertainty is reported, and the supplemental subject-disjoint diagnostic is available. If the combined method is not consistently better than both relevant controls, narrow the claim to an empirical study rather than rescue the result with architecture changes.
