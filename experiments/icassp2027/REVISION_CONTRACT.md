# ICASSP 2027 revision contract

Status: active planning contract for the focused revision of the rejected CBraMod paper.

## Paper identity

The central question is whether selective upper-layer adaptation with AttnRes
and typed specialists improves EEG downstream adaptation in CBraMod. The
contribution is a compact, controlled empirical revision of the rejected paper,
with stronger protocol control, uncertainty reporting, and efficiency
accounting. Cross-depth reuse is an implementation mechanism, not the paper's
primary identity.

This identity is separate from the TMLR study. The ICASSP paper must not reuse TMLR numerical results, figures, prose, checkpoints, or registries. The TMLR native interaction-alignment/axis-blind comparison is not an ICASSP method family.

## Scope

- Backbone: CBraMod only.
- Primary datasets: SEED-V, FACED, ISRUC, and PhysioNet-MI.
- Primary protocol: the rejected-paper CBraMod benchmark protocol, made identical across methods and seeds.
- Supplemental protocol: subject-disjoint SEED-V, reported separately from the primary table.
- Output root for new runs: `output/icassp2027_revision/`.
- Development seeds: `42`, `1024`, and `3407`, unless the artifact audit proves that an existing, larger matched block is valid.
- Model changes: none. Use the existing CBraMod implementation and existing adaptation components.

The primary runs reuse the rejected-paper serialized artifacts already mounted
under `/data/neurogroup/mingyangjiang/data`: `SEED-V_processed_lmdb`, `FACED`,
`ISRUC`, and `PHYSIO_MI`. The active revision launcher verifies these roots
before training. It also verifies the legacy split source and representative
tensor schema, then checks that the EEGxPlore loader applies the historical
single `/100` scaling exactly once. Do not regenerate or substitute a dataset
root for a paper-facing run without recording a new provenance audit.

The exact split, preprocessing, optimizer, epoch budget, checkpoint-selection rule, and data-root mapping must be recorded before launching the first matched block.

The stored-data fidelity contract is:

| Dataset | Stored representation | Rejected-paper split and loader contract |
|---|---|---|
| SEED-V | LMDB, `(62, 1, 200)` | `__keys__`; trials `0–4/5–9/10–14`; drop `M1/M2/VEO/HEO`, 200 Hz, 0.3–75 Hz, no average reference; loader `/100` |
| FACED | LMDB, `(32, 10, 200)` | `__keys__`; serialized artifact split; loader `/100` |
| ISRUC | `seq/` and `labels/`, `(20, 6, 6000)` plus `(20,)` labels | subjects `1–80/81–90/91–100`; loader `/100` |
| PhysioNet-MI | LMDB, `(64, 4, 200)` | `__keys__`; serialized artifact split; loader `/100` |

These are the CBraMod-compatible EEGxPlore loaders. The separate LaBraM
implementation intentionally returns raw tensors and owns `/100` scaling in
its training engine; it is not part of this paper.

## Main method ladder

The smallest publishable comparison is:

1. Frozen backbone + trainable classifier head;
2. Upper-1 selective fine-tuning;
3. Full fine-tuning;
4. AttnRes-only adaptation;
5. Fresh selective adaptation: AttnRes + typed specialists, using the locked
   `revision/fresh_selective_recipe.json`;
6. Specialist-only adaptation with depth-independent selection, as an optional
   component control.

For `specialist_only`, use the existing typed spatial/spectral specialist bank with the normalized layer representation as its baseline-only router input; the original CBraMod parameters and dense shared FFN remain frozen. The historical “MoE-only, no depth router” runs already include `pre_attn` AttnRes, so they are reuse candidates for the combined condition, not evidence for the specialist-only row.

`selective_fresh` is the paper-facing label for a new, independently
provenanced run family. It resolves to the existing AttnRes-plus-specialist
parameter path, but every run must carry the hash of the locked
`fresh_selective_recipe.json`; the recipe cannot be changed in response to test
results. `combined` remains an implementation/debug control.

`historical_selective` is archival context only. The original family `1785556`
has no recoverable checkpoint and lacks required raw recipe/provenance fields,
so it is permanently non-launchable and its reported values must not appear in
the ICASSP main table. The same rule applies to the historical dense and
AttnRes candidate families until their provenance is complete.

The selective method is the candidate contribution, not an assumed winner. If
it does not beat the relevant controls, the paper must be framed as a
controlled empirical study and the claim narrowed accordingly.

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
