# ICASSP 2027 routing experiment package

This directory contains the isolated contract, manifests, audits, exports,
configs, and result registry for the CBraMod subject-disjoint routing study.

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
```

The full wrapper test covers the SEED-V classifier and the production
trainability mask. The optimizer audit requires specialist output updates on
step one, finite router gradients on step two, unchanged pretrained tensors,
Static batch invariance, and Routed sample dependence.

After these gates pass, run the matched SEED-V health diagnostic first. It
compares Frozen, Upper-4, Full, Static, and Routed under the same 20-epoch
training policy. Do not launch the four-dataset matrix until train and
validation behavior shows that the adaptation is learning rather than
collapsing to a majority-class predictor.

The canonical pilot launcher is parameterized but keeps all causal settings
shared:

```bash
EPOCHS=20 MODEL_ROOT=output/icassp2027_health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V frozen
EPOCHS=20 MODEL_ROOT=output/icassp2027_health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V upper4
EPOCHS=20 MODEL_ROOT=output/icassp2027_health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V full
EPOCHS=20 MODEL_ROOT=output/icassp2027_health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V static
EPOCHS=20 MODEL_ROOT=output/icassp2027_health20 \
  bash experiments/icassp2027/configs/run_pilot.sh SEED-V routed
```

The default health policy uses AdamW, `lr=1e-4`, component multipliers
backbone/router/expert/classifier = `0.5/2.0/1.5/3.5`, no warmup, label
smoothing `0.1`, no class weighting, and zero data-loader workers. Each run
writes `selected_checkpoint_diagnostics.json` containing train/validation/test
metrics, classwise recall, prediction histograms, and SEED-V subject metrics.

Once the health gate passes, reuse the same launcher policy on SEED-V, FACED,
ISRUC, and PhysioNet-MI, followed by the planned multi-seed confirmation
runs. The `full` baseline is supported by the launcher but is not part of the
final Static-versus-Routed paper comparison unless needed as a diagnostic.

For the ACCRE GPU queue, use the same launcher through the checked-in Slurm
wrapper after committing the experiment:

```bash
sbatch --export=ALL,DATASET=SEED-V,METHOD=static experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,DATASET=SEED-V,METHOD=routed experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,DATASET=ISRUC,METHOD=upper4 experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,DATASET=PhysioNet-MI,METHOD=frozen experiments/icassp2027/configs/submit_pilot.slurm
sbatch --export=ALL,DATASET=FACED,METHOD=frozen experiments/icassp2027/configs/submit_pilot.slurm
```

Override dataset roots, `MODEL_ROOT`, `CUDA_ID`, or resource settings through
the environment. After the two SEED-V runs, validate their summary pair with:

```bash
python experiments/icassp2027/scripts/validate_static_routed_pair.py \
  --static_summary <STATIC_RUN_SUMMARY.json> \
  --routed_summary <ROUTED_RUN_SUMMARY.json>
```
