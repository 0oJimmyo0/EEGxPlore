# Publish this code to your GitHub

This directory is already a **Git** repo. It was cloned from upstream [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod); `origin` may still point there. To store **your fork** on GitHub:

## 1. Create an empty repository on GitHub

On GitHub: **New repository** → e.g. `CBraMod-fork` or `CBraMod-attnres` — **do not** add a README/license if you will push existing history.

## 2. Point `origin` at *your* repo (recommended layout)

Keep upstream as `upstream` and use `origin` for your fork:

```bash
cd /path/to/CBraMod

# If origin is still upstream, rename it
git remote rename origin upstream

# Add your repo (replace USER and REPO)
git remote add origin https://github.com/USER/REPO.git
# or SSH: git remote add origin git@github.com:USER/REPO.git
```

To **replace** `origin` with your URL only:

```bash
git remote set-url origin https://github.com/USER/REPO.git
```

## 3. Commit local changes

```bash
git status
git add -A
git commit -m "Fork: AttnRes variants, finetune updates, SLURM scripts"
```

Resolve conflicts if you later `git pull upstream main`.

## 4. Push

If this is the **first** push of your branch to the new repo:

```bash
git push -u origin main
```

If GitHub defaulted the empty repo to `master`, either rename the branch or:

```bash
git push -u origin main:main
```

## 5. Optional: stay in sync with upstream

```bash
git fetch upstream
git merge upstream/main
# or: git rebase upstream/main
```

---

**License:** Original CBraMod is MIT (see `LICENSE`). Your fork should keep that file and give credit to the original authors in the README or a `FORK.md`.
