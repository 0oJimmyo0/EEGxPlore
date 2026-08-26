# ICASSP 2027 experiment package

This directory contains the isolated contract, manifests, audits, exports,
configs, and result registries for the CBraMod subject-disjoint ICASSP studies.

The active paper profile is the selective cross-depth aggregation study defined
in `DEPTH_AGGREGATION_CONTRACT.md`. The earlier Static/Routed study is retained
as historical development work; see `ROUTING_ARCHIVE.md`.

Phase 0 order:

1. Extract common per-sample metadata for SEED-V, FACED, ISRUC, and PhysioNet-MI.
2. Verify subject recovery and class support.
3. Generate deterministic group-stratified manifests.
4. Audit overlap, key existence, counts, and manifest hashes.
5. Run frozen-probe sanity checks under the frozen manifests.

Do not launch the ICASSP training matrix before all five gates pass for all four
datasets. Historical outputs and TMLR artifacts are not inputs to this package.

Phase 0 status (2026-08-24): all gates pass. The frozen manifest hashes are:

| Dataset | Subjects (train/val/test) | Loader containers (train/val/test) | Manifest SHA-256 |
|---|---:|---:|---|
| SEED-V | 10 / 3 / 3 | 73,590 / 22,077 / 22,077 | `9785d78e458915ac8dc4fce2264ae0a4e5f0e2f0378a5bc0a28bbfe77507725e` |
| FACED | 86 / 18 / 19 | 7,224 / 1,512 / 1,596 | `7f564428b51131993e341cb4da881e8b8e167056c90499190245052ea1ed131a` |
| ISRUC | 70 / 15 / 15 | 3,017 / 738 / 707 | `253008f042c7698b17e5e9110c62c776d97238ba09ad6ffe82d75ad7b5839c79` |
| PhysioNet-MI | 76 / 16 / 17 | 6,843 / 1,464 / 1,530 | `71344f5bf12edfafedee53da7247ad10f7f8f7b678abbe084c86b8f531133601` |

Re-run the checks from the repository root with:

```bash
python experiments/icassp2027/scripts/audit_phase0.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python experiments/icassp2027/scripts/frozen_probe.py \
  --checkpoint /data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth
```

The original split audit is retained in `audits/phase0_summary.json`; it shows
subject overlap for all three SEED-V original splits, while the other three
datasets' original splits were already subject-disjoint.

## Pre-training gates

Run these repository-level gates before scheduling the short integration pilots:

```bash
python experiments/icassp2027/scripts/test_typed_conditional.py
python experiments/icassp2027/scripts/test_full_static_routed_contract.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python experiments/icassp2027/scripts/test_two_step_optimizer_contract.py
python experiments/icassp2027/scripts/test_icassp_wiring_contract.py
python experiments/icassp2027/scripts/test_depth_aggregation_contract.py
python experiments/icassp2027/scripts/test_foundation_loading_contract.py \
  --checkpoint /data/neurogroup/mingyangjiang/data/weights/pretrained_weights.pth
python experiments/icassp2027/scripts/test_icassp_matrix_contract.py
```

The existing routing tests remain regression tests for the archived routing
implementation. The depth contract covers strict foundation loading, the
DepthAgg trainability mask, uniform initialization, gradient connectivity,
frozen foundation tensors, layer scope, and optimizer groups.

After these gates pass, run the matched SEED-V depth health diagnostic first.
It compares Frozen, DepthAgg, Upper-4, and Full under the same predefined
20-epoch training policy. Do not launch the four-dataset matrix until train
and validation behavior is finite and non-degenerate.

The canonical pilot launcher is parameterized but keeps all causal settings
shared:

```bash
EPOCHS=20 MODEL_ROOT=output/icassp2027_depth/health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V frozen
EPOCHS=20 MODEL_ROOT=output/icassp2027_depth/health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V depth_aggregation
EPOCHS=20 MODEL_ROOT=output/icassp2027_depth/health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V upper4
EPOCHS=20 MODEL_ROOT=output/icassp2027_depth/health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V full
```

The launcher and Slurm wrapper default to this same 20-epoch health root, but
the explicit values above should remain in any submitted job record. Audit all
16 active cells before scheduling with:

```bash
python experiments/icassp2027/scripts/test_icassp_matrix_contract.py
```

The default health policy uses AdamW, `lr=1e-4`, component multipliers
backbone/depth/classifier = `0.5/1.0/3.5`, no warmup, label smoothing `0.1`,
no class weighting, and zero data-loader workers. Each run writes
`selected_checkpoint_diagnostics.json` containing train/validation/test
metrics, classwise recall, prediction histograms, and SEED-V subject metrics.

Once the health gate passes, run the same four-method launcher policy on
SEED-V, FACED, ISRUC, and PhysioNet-MI under a fresh
`output/icassp2027_depth/` root, followed by the planned multi-seed
confirmation runs. The archived Static/Routed runs are not part of this
matrix.

For the ACCRE GPU queue, use the same launcher through the checked-in Slurm
wrapper after committing the experiment:

```bash
sbatch --export=ALL,EXPECTED_COMMIT="$(git rev-parse HEAD)",EPOCHS=20,MODEL_ROOT="$PWD/output/icassp2027_depth/health20",DATASET=SEED-V,METHOD=depth_aggregation experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,EXPECTED_COMMIT="$(git rev-parse HEAD)",EPOCHS=20,MODEL_ROOT="$PWD/output/icassp2027_depth/health20",DATASET=ISRUC,METHOD=upper4 experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,EXPECTED_COMMIT="$(git rev-parse HEAD)",EPOCHS=20,MODEL_ROOT="$PWD/output/icassp2027_depth/health20",DATASET=PhysioNet-MI,METHOD=frozen experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,EXPECTED_COMMIT="$(git rev-parse HEAD)",EPOCHS=20,MODEL_ROOT="$PWD/output/icassp2027_depth/health20",DATASET=FACED,METHOD=frozen experiments/icassp2027/configs/submit_pilot.slurm
```

Override dataset roots, `MODEL_ROOT`, `CUDA_ID`, or resource settings through
the environment. The old Static/Routed summary validator remains available
only for the archived routing study.
