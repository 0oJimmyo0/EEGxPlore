# ICASSP 2027 Routing Study Contract

Status: Phase 0 frozen for implementation; SEED-V manifest revised to explicit 10/3/3 allocation
Branch: `icassp2027-routing`

## Scientific identity

Working title: **Sample-Conditioned Specialist Routing for Subject-Disjoint CBraMod Transfer**

Central question:

> Across four subject-disjoint CBraMod transfer tasks, does giving a parameter-identical specialist router access to each sample improve adaptation over input-independent expert allocation?

The primary estimand is the paired difference between sample-routed and static typed specialists:

`delta_route(D, seed) = metric(routed, D, seed) - metric(static, D, seed)`.

ICASSP is explicitly CBraMod-specific. It does not claim cross-backbone generality.

## Frozen scope

- Backbone: CBraMod only.
- Adaptation site: the upper four CBraMod transformer blocks.
- Specialist banks: spatial and spectral-temporal typed banks.
- Dispatch: soft dispatch.
- Route implementation: the new `typed_conditional` mode; legacy
  `typed_capacity_domain` is excluded from ICASSP runs.
- Static router: the same router network fed by a learned constant input.
- Routed router: the same router network fed by the current sample representation plus the same learned constant.
- Router dropout, jitter, depth/context features, and router-specific regularizers: disabled in the primary Static/Routed comparison.
- Every original pretrained CBraMod parameter is frozen for Static/Routed;
  only the upper-four conditional specialist banks, routers, learned
  constants, and task head are trainable. The shared FFN remains as the frozen
  dense foundation.
- Primary datasets: SEED-V, FACED, ISRUC, PhysioNet-MI.
- Split regime: fresh subject-disjoint manifests generated for this study.
- Primary test metrics: balanced accuracy and macro-F1.
- Secondary metrics: Cohen's kappa and weighted-F1.
- Checkpoint selection: validation Cohen's kappa only.
- Development seeds: `42, 1024, 3407`.
- Additional Static/Routed seeds: `2027, 2718`.
- Mechanism analysis: SEED-V subject-by-expert routing profiles only.

The accepted implementation decisions and the ICASSP configuration firewall
are recorded in `IMPLEMENTATION_DECISIONS.md`.

## Fixed method matrix

| ID | Condition | Role |
|---|---|---|
| B0 | Frozen probe | Representation reference |
| B1 | Full fine-tuning | Dense reference |
| B2 | Upper-4 fine-tuning | Selective-tuning baseline |
| B3 | Upper-layer LoRA | Generic PEFT baseline |
| B4 | Generic bottleneck | Generic residual PEFT baseline |
| S | Static typed specialists | Causal capacity control |
| R | Sample-routed typed specialists | Proposed condition |

Static and routed specialist conditions must share the same model state schema,
initialization, trainable parameter counts, expert banks, shared FFN,
optimizer/scheduler, epoch budget, and checkpoint rule. The only intended
intervention is whether the router receives sample-dependent information.

## Dataset protocol requirements

Every dataset must provide a common sample metadata table before splitting:

```text
sample_key, subject_id, session_id, recording_id, label
```

Each frozen manifest must include:

- deterministic group-stratified subject assignment;
- zero subject overlap across train/validation/test;
- required class support in every split;
- sample and subject counts;
- manifest SHA-256;
- key-existence audit;
- split-generation metadata.

No model result may be used to generate or revise a manifest.

## Evidence and artifact firewall

- No TMLR numerical result rows, checkpoints, figures, or registries may enter the ICASSP evidence package.
- No TMLR native-axis or axis-agnostic adapter code may be invoked by an ICASSP run.
- No LaBraM, TUEV, AttnRes, depth-routing, EEG/PSD-context, or native-geometry claims belong to this paper.
- The ICASSP manuscript must be authored from a blank ICASSP source file.
- All ICASSP results must be generated under ICASSP manifests and recorded in the ICASSP registry.

## Run policy

Phase 0 has no training matrix. It consists of metadata extraction, manifest
generation, split audits, and frozen-probe sanity checks. Full training begins
only after all four datasets pass the Phase 0 gates.

The planned evidence package is:

- 28 short smoke jobs: 7 methods x 4 datasets;
- 84 three-seed runs: 7 methods x 4 datasets x 3 seeds;
- 16 additional Static/Routed runs: 2 methods x 4 datasets x 2 seeds.

Test performance is never used to modify the contract, manifests, or model
configuration.
