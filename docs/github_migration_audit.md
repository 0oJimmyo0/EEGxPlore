# GitHub Migration Audit

This audit prepares the root `EEGxPlore` repo for GitHub migration from the current cluster checkout.

## Goal

Push the code, scripts, lightweight docs, and reproducible metadata to the root GitHub repo while keeping:

- runtime logs,
- checkpoints and pretrained binaries,
- large EEG datasets,
- and the nested Overleaf paper repo

out of the root repository.

## Recommended To Add

These files are good root-repo candidates and should be reviewed, then staged in the root GitHub repo.

### Modified tracked files

- `finetune_main.py`
- `preprocessing/preprocessing_physio.py`
- `scripts/FACED/submit_train.slurm`
- `scripts/ISRUC/submit_train.slurm`
- `scripts/PHYSIO-MI/train_physio_compact_shared.slurm`
- `scripts/SEED-V/submit_seedv_train.slurm`
- `scripts/TUEV/submit_train.slurm`

### Deleted tracked files

These are currently deletions relative to the root repo and should be confirmed before commit.

- `scripts/FACED_A1_ABLATION_PLAN.md`
- `scripts/analyze_facced_routing.py`

### New untracked files that are reasonable to add

- `docs/logs_output.md`
- `docs/logs_test_output_registry.md`
- `docs/modif_log.md`
- `docs/next_step.md`
- `docs/prompt.md`
- `docs/template.tex`
- `docs/github_migration_audit.md`
- `scripts/run_faced.sh`
- `scripts/run_seedv.sh`

## Recommended To Keep Ignored

These should stay out of the root GitHub repo.

### Runtime and cluster outputs

- `logs/`
- `output/`
- `checkpoints/`
- `*.pth`
- `*.pt`
- `*.ckpt`
- `wandb/`

### Local scratch/check files

- `.quota_check`

### Generated figure snapshot in the root figure folder

- `figure/faced_main_results.pdf`

The existing tracked PNG assets in `figure/` are fine to keep.

### Nested paper checkout

- `paper/`

Reason:

- `paper/69f14702cd352097cf163436/` is a separate Git repository with its own `.git`.
- It also contains `build/` outputs and compiled PDFs.
- Adding it from the root repo would create an embedded-repo situation and complicate migration.

## Recommended Separate Handling For The Paper

Do not add `paper/` to the root GitHub repo by default.

Instead choose one of these later:

1. Keep the paper only in the Overleaf repo.
2. Mirror the paper source into a separate GitHub repo.
3. Export a clean paper-source snapshot into a non-nested folder later, excluding:
   - `.git/`
   - `build/`
   - compiled PDFs unless intentionally desired

## Root Repo State Summary

The current root repo is small enough for GitHub migration.

- `docs/` is small
- `scripts/`, `models/`, `datasets/`, and `preprocessing/` are all lightweight
- no large EEG dataset lives inside the root repo
- no >20 MB root-repo files were found outside the nested paper checkout

The main large assets still live outside the repo and should be transferred separately to the new SSH server:

- preprocessed EEG datasets
- pretrained model weights
- selected checkpoints if needed

## Patched Ignore Rules

The root `.gitignore` has been updated to ignore:

- `.quota_check`
- `figure/*.pdf`
- `paper/`

This is intentionally conservative to reduce accidental migration of local artifacts.

## Recommended Next Commands

Review the exact staged candidates:

```bash
cd /gpfs/radev/project/xu_hua/mj756/EEG_F/model_rep/EEGxPlore

git status --short
git diff -- finetune_main.py preprocessing/preprocessing_physio.py scripts docs .gitignore
```

Stage the root-repo migration set:

```bash
git add .gitignore
git add finetune_main.py
git add preprocessing/preprocessing_physio.py
git add scripts/FACED/submit_train.slurm
git add scripts/ISRUC/submit_train.slurm
git add scripts/PHYSIO-MI/train_physio_compact_shared.slurm
git add scripts/SEED-V/submit_seedv_train.slurm
git add scripts/TUEV/submit_train.slurm
git add scripts/run_faced.sh
git add scripts/run_seedv.sh
git add docs
git rm scripts/FACED_A1_ABLATION_PLAN.md
git rm scripts/analyze_facced_routing.py
```

Double-check that the nested paper repo is not being added:

```bash
git status --short
git check-ignore -v paper figure/faced_main_results.pdf .quota_check
```

If the staged set looks right, commit and push the root repo.

## New Server Transfer Split

After the root GitHub push, migrate external assets separately:

1. clone the root repo on the new SSH server
2. recreate the conda/runtime environment
3. transfer datasets with `rsync` or `scp`
4. transfer pretrained weights
5. optionally transfer selected checkpoints
6. recreate the `logs/` symlink-to-scratch pattern on the new server
