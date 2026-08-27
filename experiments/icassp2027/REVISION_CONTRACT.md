# ICASSP 2027 revision contract

Status: active planning contract for the focused revision of the rejected CBraMod paper.

## Paper identity

The central question is whether selective upper-layer adaptation with cross-depth residual reuse improves EEG downstream adaptation in CBraMod. The contribution is a compact, controlled empirical revision of the rejected paper, with stronger protocol control and uncertainty reporting.

This identity is separate from the TMLR study. The ICASSP paper must not reuse TMLR numerical results, figures, prose, checkpoints, or registries. The TMLR native interaction-alignment/axis-blind comparison is not an ICASSP method family.

## Scope

- Backbone: CBraMod only.
- Primary datasets: SEED-V, FACED, ISRUC, and PhysioNet-MI.
- Primary protocol: the rejected-paper CBraMod benchmark protocol, made identical across methods and seeds.
- Supplemental protocol: subject-disjoint SEED-V, reported separately from the primary table.
- Output root for new runs: `output/icassp2027_revision/`.
- Development seeds: `42`, `1024`, and `3407`, unless the artifact audit proves that an existing, larger matched block is valid.
- Model changes: none. Use the existing CBraMod implementation and existing adaptation components.

The exact split, preprocessing, optimizer, epoch budget, checkpoint-selection rule, and data-root mapping must be recorded before launching the first matched block.

## Main method ladder

The smallest publishable comparison is:

1. Frozen backbone + trainable classifier head;
2. Upper-1 selective fine-tuning;
3. Full fine-tuning;
4. AttnRes-only adaptation;
5. Specialist-only adaptation with depth-independent selection;
6. AttnRes + specialist adaptation, if the matched artifact audit supports it.

For `specialist_only`, use the existing typed spatial/spectral specialist bank with the normalized layer representation as its baseline-only router input; the original CBraMod parameters and dense shared FFN remain frozen. The historical “MoE-only, no depth router” runs already include `pre_attn` AttnRes, so they are reuse candidates for the combined condition, not evidence for the specialist-only row.

The combined method is the candidate contribution, not an assumed winner. If it does not beat the relevant controls, the paper must be framed as a controlled empirical study and the claim narrowed accordingly.

Learned depth-conditioned routing, compact-context routing, PSD/context inputs, new expert layouts, new low-rank operators, and broad depth/expert sweeps are not part of the primary result. At most one already-implemented depth-enabled condition may appear as a clearly labeled supplemental diagnostic if it is cheap and its provenance is complete.

## Evidence rules

- Report balanced accuracy and macro-F1 as mean ± standard deviation over matched seeds.
- Select checkpoints using validation Cohen's kappa only; test labels remain inaccessible during development.
- Do not mix protocols, split definitions, epoch budgets, or selection rules within a main comparison.
- Historical outputs are candidates, not results. Promote one only after recording dataset, split, seed, code commit, checkpoint, selector, trainable parameter count, metric files, and overlap status.
- Every new run records the repository commit, data-manifest/checksum information, full command/configuration, seed, GPU allocation, checkpoint path, and final metrics.
- A failed, OOM, incomplete, or provenance-incomplete run is not a result row.

## Explicit exclusions

LaBraM, TUEV, Mumtaz2016, a second backbone, TMLR-native-axis experiments, the deleted DepthAgg/Static/Routed primary paths, and broad hyperparameter sweeps are outside the minimal ICASSP revision.

## Decision gate

The revision is ready for paper assembly when it has a complete matched core table, uncertainty over the declared seeds, one efficiency/parameter summary, and the separate SEED-V subject-disjoint diagnostic. Do not add another architecture merely because one dataset is weak; first verify the baseline, split, and checkpoint-selection contract.
