# CBraMod Execution Memory

Last updated: 2026-08-02

## Current paper identity

The revised project is **Interaction-Aligned Adaptation for EEG Foundation
Models**. The canonical plan is
`/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/docs/cross_backbone_execution_plan.md`.

This repository contains the historical EEGxPlore/ICASSP CBraMod path. The
active TMLR CBraMod study is isolated in the clean clone
`/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod`; LaBraM runs belong in the
sibling LaBraM repository. These paths must never be mixed for paper results.

## Repository state

- Repository: `/data/neurogroup/mingyangjiang/EEGxPlore/EEGxPlore`
- Branch: `SEED-V`
- HEAD at audit: `861c222`
- Working tree at audit: clean
- Historical LaBraM substitution flags remain in `finetune_main.py`, but are
  not part of the revised paper protocol.

## Confirmed implementation facts

- `finetune_main.py:120-127` exposes CBraMod and a historical LaBraM choice.
- `models/cbramod.py:358-382` realizes a `[B,C,S,D]` patch grid.
- CBraMod has native spatial/channel and temporal/spectral processing through
  `models/criss_cross_transformer.py`.
- `models/attn_res.py` provides `FullAttnRes`.
- `models/moe.py` provides typed capacity/domain specialists, router input modes,
  and `hard_capacity`/`soft` dispatch.
- Existing compact EEG, PSD, subject, and depth router contexts are optional
  and remain diagnostic until the revised audit is complete.

## Scientific decisions

- The primary CBraMod method must use the common low-rank residual interaction
  primitive, not the legacy typed-MoE path.
- AttnRes, typed MoE, routing, and depth are retained as historical or
  diagnostic ablations, not assumed central contributions.
- The first CBraMod work is a trust/provenance audit and SEED-V mechanism
  decomposition. Do not run all five datasets while the method is moving.

## TMLR CBraMod study status

The active TMLR CBraMod repository is `/data/neurogroup/mingyangjiang/EEGxPlore/CBraMod`.
Its audited FACED contract is: strict original CBraMod checkpoint loading,
32-channel manifest, runtime geometry `[B,32,10,200]`, `/100` scaling, source
classifier head, batch size 64, 50 epochs, source loader behavior,
per-iteration cosine schedule, validation-kappa checkpoint selection, and
exactly three final seeds `{42,1024,3407}`.

The CBraMod native adapter has two distinct regimes:

- frozen native: frozen backbone + native channel/patch adapter + classifier;
- native full: trainable backbone + the same native adapter + classifier.

The native adapter is never combined with LoRA or generic controls. The native
full condition tests whether the adapter complements dense adaptation; the
frozen condition tests low-drift adaptation.

FACED status:

- dense full fine-tuning: verified, three seeds complete;
- seed-42 screens: verified for frozen classifier, frozen native channel,
  frozen native patch, frozen native channel+patch, generic bottleneck, LoRA
  QKV-r8, upper-2, and parameter-matched axis-blind control;
- generic bottleneck, frozen classifier, LoRA, upper-2, and axis-blind:
  eligible for three-seed promotion after artifact checks;
- native frozen multiseeds: pending residual-scale and trajectory stabilization;
- native full-backbone-plus-adapter: implemented as
  `native_full_finetune`, pending construction and seed-42 validation gates.

## Per-backbone/dataset TMLR checklist

Every backbone–dataset cell must complete this ladder before moving to the next
dataset:

1. protocol/provenance audit and dense smoke;
2. three-seed dense full-finetune baseline;
3. three-seed frozen classifier;
4. three-seed generic bottleneck, LoRA QKV-r8, upper-2, and parameter-matched
   axis-blind controls;
5. native frozen channel, patch, and channel+patch screens, then three seeds
   only after residual and trajectory checks;
6. native full-backbone-plus-adapter channel, patch, and channel+patch screens,
   then three seeds only after strict wiring and trajectory checks;
7. complete epoch trajectories, validation-selected test metrics, parameter
   counts, optimizer groups, residual/alpha/gradient/update diagnostics, and
   failure audit.

Completion requires seeds `{42,1024,3407}` for each required condition or a
documented scientific negative-result decision. Smoke and failed runs do not
count, and test metrics never select a method or epoch.

## Dataset status

- FACED: historical CBraMod pipeline and result artifacts exist; paper-grade
  protocol/seed completeness must be audited under the revised contract.
- SEED-V: existing LMDB and split-audit tooling exist; current protocol is not
  subject-independent and must be reported as such. A grouped/subject-disjoint
  evaluation is required for the final claim.
- ISRUC: loader exists; exact preprocessing, channel order, subject split, and
  artifact hash must be frozen before final runs.
- TUEV: loader and class support checks exist; per-class metrics and split
  provenance are required.
- PhysioNet-MI: loader exists, but inclusion in the primary matrix is pending a
  complete data/provenance audit.

## Baseline and implementation gaps

Still required in this repository: paper-grade frozen/full/upper-k baselines,
LoRA, generic bottleneck, parameter-matched axis-blind adapter, common aligned
primitive, parameter-count checker, unified result registry, and paired
multi-seed statistical aggregation. No LaBraM implementation should be copied
here to fill these gaps.

## Status vocabulary

Use only: `verified`, `pilot`, `exploratory`, `historical`, `pending`, or
`blocked`. Existing CBraMod/LaBraM substitution results are historical unless
they satisfy the new repository boundary and contract.

## Immediate handoff

1. Keep CBraMod in this repository and LaBraM in LaBraM.
2. Audit the official CBraMod dispatch/default/result provenance.
3. Freeze SEED-V data/split/metric contracts.
4. Implement matched generic and aligned controls without LaBraM imports.
5. Complete CBraMod baselines on SEED-V before expanding to FACED, ISRUC, TUEV,
   and PhysioNet-MI.
