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
