# A1 Ablation Plan (Current Meaning)

`A1` now refers to the SEED-V block-summary depth-context ablation, not the older FACED no-multi-lr/warmup/gated sweep.

## One-Change Contract

Approach 1 must keep exactly one depth-context change active:

1. `moe_attnres_depth_context_mode=block_shared_typed_proj`
2. fixed `moe_attnres_depth_block_count=4`
3. fixed `moe_attnres_depth_router_dim=15`
4. no compact-summary-only toggles mixed in (summary-mode/probe-MLP knobs)
5. CBraMod SEED-V cohort path and split source: `/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb` + LMDB `__keys__`

The launcher `scripts/SEED-V/submit_seedv_train.slurm` enforces this when:

- `A1_STRICT_BLOCK_ABLATION=1` (default)

## Submit

```bash
cd scripts/SEED-V
sbatch --export=ALL,SEEDV_PROTOCOL=cbramod_benchmark,DATASET_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb,RUN_NAME=seedv_a1_block_strict submit_seedv_train.slurm
```

Leave `SEEDV_SPLIT_MANIFEST` unset for CBraMod-cohort parity.

Optional compact-baseline comparator (explicitly opt out of strict A1):

```bash
cd scripts/SEED-V
sbatch --export=ALL,SEEDV_PROTOCOL=cbramod_benchmark,DATASET_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb,A1_STRICT_BLOCK_ABLATION=0,DEPTH_CONTEXT_MODE=compact_shared,DEPTH_SUMMARY_MODE=attn_delta4,DEPTH_PROBE_MLP_FOR_ROUTER=on,RUN_NAME=seedv_compact_baseline submit_seedv_train.slurm
```

## Required Diagnostics Checks

After a run, inspect these files under `model_dir`:

1. `block_summary_stats.json`
2. `router_context_stats.json`
3. `routing_diagnostics.json`
4. `epoch_diagnostics.jsonl`

Quick verifier:

```bash
python scripts/SEED-V/verify_block_context_diagnostics.py \
  --run_dir <MODEL_DIR_RUN1> \
  --run_dir <MODEL_DIR_RUN2>
```

Pass criteria:

1. block rows exist with `depth_context_mode=block_shared_typed_proj`
2. block metadata fields are populated (`block_count`, `block_layer_counts`, norms)
3. projected depth context norms are non-empty
4. values vary within run and/or across runs
