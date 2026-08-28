# ICASSP 2027 revision contract

Status: active focused revision of the rejected CBraMod selective-adaptation paper.

## Paper identity

The paper studies robust selective adaptation of CBraMod with AttnRes and
typed spatial/spectral specialists. Its central comparison is selective
upper-layer adaptation versus dense fine-tuning under fixed downstream
protocols. Depth-conditioned routing is a secondary diagnostic, not the main
claim.

This is separate from the TMLR study. The ICASSP package must not reuse TMLR
numerical rows, figures, prose, checkpoints, or registries. TMLR-native
interaction-alignment and axis-comparison experiments are outside this paper.

## Scope

- Backbone: CBraMod only.
- Active datasets: SEED-V, FACED, and ISRUC.
- Archived datasets: PhysioNet-MI and TUEV; no new paper-facing runs.
- Active protocol: `cbramod_benchmark` only.
- Archived protocol: `seedv_subject_disjoint`; it is not a readiness gate.
- Declared seeds: `42`, `3407`, and `2024`.
- New-run output root: `output/icassp2027_revision/`.
- Smoke output root: `output/icassp2027_smoke/`; smoke rows are ineligible.
- Model changes: none. Reuse the existing CBraMod adaptation components.

The active launcher verifies the serialized CBraMod-compatible data artifacts,
split source, representative schema, and exactly-once `/100` loader scaling.
The data roots and preprocessing contract are recorded in the run manifest.

## Evidence classes

| Class | Meaning | Paper use |
|---|---|---|
| `legacy_reported_paper` | Complete seed-level values printed in the rejected manuscript; checkpoint/config unavailable in the current checkout | Main table only with an explicit legacy label |
| `legacy_context_only` | Single-run, exploratory, incomplete, or ambiguous historical evidence | Context/appendix only |
| `new_multiseed` | New run under a locked ICASSP protocol with independent artifacts | Primary evidence |

Legacy evidence is never silently promoted to a current reproduction. It must
carry its source table, verification level, and paper eligibility in the
registry. Do not pool legacy and new values into a single statistical summary.

## Main evidence plan

The main cross-dataset comparison is Dense versus Selective:

| Dataset | Dense | Selective |
|---|---|---|
| SEED-V | legacy reported, three seeds | legacy reported, three seeds |
| FACED | new, three seeds | new, three seeds |
| ISRUC | new, three seeds | new, three seeds |

The SEED-V component table contains Dense, AttnRes-only, Selective, and a new
Upper-1 control. The old rejected-paper “MoE-only” rows are interpreted as
AttnRes plus typed specialists without depth routing; they are not
`specialist_only` evidence.

Executable condition names are deliberately separate from paper labels:

| Paper label | Executable condition |
|---|---|
| Dense | `full` |
| AttnRes-only | `attnres_only` |
| Selective, new paper-derived | `selective_paper` |
| Selective, existing independent recipe | `selective_fresh` |
| Upper-1 | `upper1` |

`specialist_only`, `combined`, and `selective_fresh` remain available for
implementation tests or separately archived studies, but are not required
paper rows. `historical_selective` is permanently non-launchable.

## Dataset-derived protocols

The current `fresh_selective_recipe.json` remains untouched. New paper-derived
runs use locked dataset protocol files. The manuscript-supported settings are:

| Dataset | Epochs | Batch | LR | Weight decay |
|---|---:|---:|---:|---:|
| SEED-V | 25 | 64 | `3e-5` | `3e-2` |
| FACED | 40 | 32 | `2e-4` | `2e-2` |
| ISRUC | 30 | 16 | `3e-5` | `2e-2` |

These are paper-derived ICASSP execution protocols, not claims that every
historical implementation field has been recovered. Each protocol records
which fields came from the manuscript and which are explicit current-run
defaults.

## Provenance and reporting rules

- Select checkpoints using validation Cohen's kappa only.
- Report balanced accuracy and macro-F1 for new runs as mean ± standard deviation.
- Historical manuscript weighted-F1 values remain weighted-F1; never relabel them as macro-F1.
- Record repository commit, data-contract hash, protocol hash, method condition, seed, checkpoint, metrics, runtime, GPU, and trainable parameters for every new run.
- Failed, OOM, incomplete, smoke, or provenance-incomplete runs are not result rows.
- Do not calculate pooled statistics across legacy and new provenance classes.

## Explicit exclusions

LaBraM, TUEV, PhysioNet-MI as a primary dataset, Mumtaz2016, a second
backbone, subject-disjoint SEED-V as a readiness requirement, TMLR-native-axis
experiments, new operators, new routing designs, and broad hyperparameter
sweeps are outside the active revision.

## Readiness gate

The paper is ready for assembly when the declared legacy rows are visibly
labeled, the new Upper-1 and FACED/ISRUC Dense/Selective blocks are complete
over the three declared seeds, and one efficiency summary is available. If
Selective is not consistently better, narrow the claim to a controlled
empirical study rather than adding architecture variants.
