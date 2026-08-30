# ICASSP 2027 revision contract

Status: active focused revision of the rejected CBraMod selective-adaptation paper.

## Paper identity

The paper studies robust CBraMod adaptation with AttnRes and typed
spatial/spectral specialists. The confirmed method intentionally fine-tunes
the full backbone and adds one top-layer specialist block. Depth-conditioned
routing is a development diagnostic, not the main claim.

This is separate from the TMLR study. The ICASSP package must not reuse TMLR
numerical rows, figures, prose, checkpoints, or registries. TMLR-native
interaction-alignment and axis-comparison experiments are outside this paper.

## Scope

- Backbone: CBraMod only.
- Active datasets: SEED-V, FACED, ISRUC, and PhysioNet-MI.
- Archived dataset: TUEV; no new paper-facing runs.
- Active protocol: `cbramod_benchmark` only.
- Archived protocol: `seedv_subject_disjoint`; it is not a readiness gate.
- Development seed: `42`.
- Confirmatory seeds: `3407`, `2024`, and newly prespecified `2027`.
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

The main confirmatory cross-dataset comparison is Full fine-tuning versus
AttnRes + Typed Specialists:

| Dataset | Dense | Selective |
|---|---|---|
| FACED | new, three unseen seeds | new, three unseen seeds |
| ISRUC | new, three unseen seeds | new, three unseen seeds |
| SEED-V | new, three unseen seeds | new, three unseen seeds |
| PhysioNet-MI | new, three unseen seeds | new, three unseen seeds |

The rejected-paper SEED-V Dense, AttnRes-only, and Selective rows remain a
separately labeled legacy component/context table. They are not pooled with
the new confirmatory rows. Seed-42 FACED/ISRUC comparisons, including the
historical-route candidate and opt-only candidate, remain development
evidence and are excluded from confirmatory mean ± SD.

Executable condition names are deliberately separate from paper labels:

| Paper label | Executable condition |
|---|---|
| Dense | `full` |
| AttnRes-only | `attnres_only` |
| AttnRes + Typed Specialists | `specialist_augmented_full` |

`selective_paper`, `specialist_only`, `combined`, and `selective_fresh` remain
available for implementation tests or archived diagnostics, but are not part
of the final confirmatory table. `historical_selective` is permanently
non-launchable.

## Dataset-derived protocols

The current `fresh_selective_recipe.json` remains untouched. New paper-derived
runs use locked dataset protocol files. The manuscript-supported settings are:

| Dataset | Epochs | Batch | LR | Weight decay |
|---|---:|---:|---:|---:|
| SEED-V | 25 | 64 | `3e-5` | `3e-2` |
| FACED | 40 | 32 | `2e-4` | `2e-2` |
| ISRUC | 30 | 16 | `3e-5` | `2e-2` |
| PhysioNet-MI | 30 | 64 | `3e-5` | `2e-2` |

These are paper-derived ICASSP execution protocols, not claims that every
historical implementation field has been recovered. Each protocol records
which fields came from the manuscript and which are explicit current-run
defaults. Component-wise learning-rate scaling is explicitly disabled in all
four paper-facing protocols; it remains enabled only in the archived
independent `fresh_selective` recipe.

## Provenance and reporting rules

- Select checkpoints using validation Cohen's kappa only.
- Report balanced accuracy, weighted-F1, and kappa for the uniform cross-source comparison as mean ± standard deviation.
- Record macro-F1 for every new run as an additional metric; it is not the primary cross-source metric because the legacy SEED-V rows do not contain it.
- Historical manuscript weighted-F1 values remain weighted-F1; never relabel them as macro-F1.
- Record repository commit, data-contract hash, protocol hash, locked method
  recipe hash, method condition, seed, checkpoint, metrics, runtime, GPU, and
  trainable parameters for every new run.
- Failed, OOM, incomplete, smoke, or provenance-incomplete runs are not result rows.
- Do not calculate pooled statistics across legacy and new provenance classes.

## Explicit exclusions

LaBraM, TUEV, Mumtaz2016, a second
backbone, subject-disjoint SEED-V as a readiness requirement, TMLR-native-axis
experiments, new operators, new routing designs, and broad hyperparameter
sweeps are outside the active revision. The explicit depth-aware routing
candidate is retained only as a small development control.

## Readiness gate

The paper is ready for assembly when the declared legacy rows are visibly
labeled, the 24 rows in `paper_table_manifest_v4.csv` are complete over the
four datasets and three unseen seeds, and one accuracy/overhead summary is available. If the
specialist method is not consistently better than full fine-tuning, narrow the
claim to a controlled simplification study rather than adding architecture
variants.
