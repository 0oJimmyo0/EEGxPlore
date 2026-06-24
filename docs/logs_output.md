# Logs Audit and Ablation Execution Guide

## Scope

This note audits the current `EEGxPlore/logs/` tree and summarizes what the existing runs tell us right now.
It also explains which ablations are already supported by the current SLURM launchers and which ones still matter most for the paper.

This update supersedes the older log note in three places:
- the inventory counts are larger now, especially for `FACED`, `SEED-V`, and `Phy_MI`
- there is now a clean new `SEED-V` baseline family that directly compares dense, AttnRes-only, and selective MoE-style variants under one matched setup
- there are new focused ablation batches on `FACED` and `PhysioNet-MI` that materially sharpen the paper story

Important paper-status caveats:
- `Mumtaz` is still excluded from the paper because the current preprocessing is not subject-wise and can introduce subject leakage.
- `PhysioNet-MI` should still be treated as a boundary-case dataset rather than folded into an “improves everywhere” narrative.
- This paper should be positioned as a downstream adaptation method paper, not as a benchmark paper. The role of the five retained datasets is to test transfer breadth and generalization under heterogeneous downstream shift.

---

## Inventory Overview

| Dataset | Total `.out` logs | Runs with parsed args | Runs with final raw test metrics | Runs with final EMA metrics | Runs with newer summary path / retained summary bookkeeping |
|---|---:|---:|---:|---:|---:|
| FACED | 60 | 56 | 51 | 9 | 13 |
| ISRUC | 19 | 19 | 19 | 17 | 19 |
| Mumtaz | 8 | 8 | 7 | 1 | 7 |
| Phy_MI | 80 | 80 | 77 | 17 | 77 |
| SEED-V | 68 | 67 | 63 | 17 | 63 |
| TUEV | 70 | 69 | 63 | 2 | 63 |

Delta versus the previous audit:
- `FACED`: grew from 51 to 60 logs
- `Phy_MI`: grew from 71 to 80 logs
- `SEED-V`: grew from 61 to 68 logs
- `TUEV`: grew materially through the new `1785589--1799572` family and now contains a completed multi-seed dense/AttnRes/full control block for the paper
- `ISRUC` is unchanged in count but remains one of the strongest stable support datasets

High-level reading:
- `SEED-V` now has the cleanest new matched baseline family among all datasets.
- `FACED` now has a much more useful within-method ablation block for context and depth-summary content.
- `PhysioNet-MI` now has a clearer simplification/regularization study, which helps the boundary-case analysis.
- `ISRUC` remains one of the cleanest support datasets for the paper.
- `TUEV` remains one of the most important datasets for the main paper because it now provides a completed multi-seed control family with clean baseline toggles.

---

## Cross-Dataset Experimental Pattern

Across the repository, the current method family still has a clear center of gravity:
- backbone is almost always initialized from the CBraMod pretrained weights
- `attnres_variant` is usually `pre_attn`
- MoE is enabled in most recent selective runs
- the two dominant depth-summary patterns are:
  - `compact_shared + attn_delta4`
  - typed/block context modes using `auto`
- router dispatch is usually `soft`
- router architecture is usually `mlp`, with a smaller baseline-style branch using `linear`

That means the paper story is still coherent:
- the project has moved away from generic dense finetuning and toward structured selective adaptation
- the main questions being tested are now not arbitrary hyperparameters, but:
  - whether depth-aware routing helps
  - whether typed context structure helps
  - whether compact EEG router features help
  - whether simpler adaptation can already be enough on some tasks

---

## Dataset-by-Dataset Summary

## FACED

### Inventory and format status
- 60 total runs
- 56 runs with parseable `Namespace(...)`
- 51 runs with final raw test metrics
- 9 runs with final EMA test metrics
- 13 runs in the newer summary-tracked style

### What changed in this update
The new `1771927–1771939` family materially improves the FACED ablation story.
It adds:
- a gradient-path block:
  - `faced_grad_detached`
  - `faced_grad_delayed`
  - `faced_grad_trainable`
- a depth-summary content block:
  - `faced_content_delta4`
  - `faced_content_balanced`
  - `faced_content_latemix`
- a context block:
  - `faced_ctx_compact`
  - `faced_ctx_dual`
  - `faced_ctx_block`

### Parameter distribution snapshot
- `attnres_variant`: `pre_attn` 52, `full` 4, missing 4
- `use_ema`: missing 44, `True` 12, `False` 4
- `moe=True` in 56 runs, missing in the 4 oldest sparse logs
- `moe_attnres_depth_context_mode`: missing 44, `compact_shared` 11, `dual_query_block_typed_proj` 3, `block_shared_typed_proj` 2
- `moe_attnres_depth_summary_mode`: missing 31, `attn_delta4` 20, `auto` 5, `attn_mlp_balanced` 2, `attn_mlp_latemix` 2
- `moe_attnres_depth_summary_grad_mode`: missing 41, `detached` 12, `trainable` 5, `delayed_unfreeze` 2
- `moe_router_arch`: `mlp` 49, `linear` 7, missing 4
- `moe_router_dispatch_mode`: `soft` 48, `hard_capacity` 8, missing 4
- `moe_router_compact_feature_mode`: missing 44, `eeg_summary` 9, `none` 7

### Best observed FACED runs
Best historical raw run still visible in the retained logs:
- `faced_hier_1464851.out`
- raw test = `acc 0.60548 / kappa 0.55276 / f1 0.60721`
- older 50-epoch exploratory regime

Best historical EMA run still visible in the retained logs:
- `faced_hier_1504160.out`
- EMA test = `acc 0.59179 / kappa 0.53800 / f1 0.59419`
- configuration:
  - `compact_shared`
  - `attn_delta4`
  - `detached`
  - `use_ema=True`

### What the newest FACED batch indicates
#### Content ablations
The new content block is clean and interpretable:
- `faced_content_delta4`: raw `0.58025 / 0.52494 / 0.58347`, EMA `0.58038 / 0.52406 / 0.58274`
- `faced_content_balanced`: raw `0.57810 / 0.52275 / 0.58083`, EMA `0.58065 / 0.52516 / 0.58278`
- `faced_content_latemix`: raw `0.57958 / 0.52424 / 0.58406`, EMA `0.58253 / 0.52699 / 0.58428`

Reading:
- the new content-family differences are real but small
- `attn_mlp_latemix` is the strongest of the three in the new standardized content block
- `attn_delta4` remains competitive enough that we should not overclaim one universally superior summary content mode

#### Context ablations
The new context block is more informative than the content block:
- `faced_ctx_compact`: raw `0.58481 / 0.52917 / 0.58703`, EMA `0.58078 / 0.52457 / 0.58258`
- `faced_ctx_dual`: raw `0.58655 / 0.53216 / 0.58845`, EMA `0.58414 / 0.52877 / 0.58567`
- `faced_ctx_block`: raw `0.58910 / 0.53414 / 0.58920`, EMA `0.58655 / 0.53159 / 0.58765`

Reading:
- in the newest FACED context block, richer typed block context is slightly better than compact shared context
- `block_shared_typed_proj` is the strongest among the new context runs
- the gap is not huge, so the honest paper framing is “typed context is promising and slightly stronger on FACED,” not “typed context decisively wins everywhere”

#### Gradient-path ablations
The new gradient-path runs are present:
- `faced_grad_detached`
- `faced_grad_delayed`
- `faced_grad_trainable`

But in the currently retained `.out` files:
- the logs expose training and diagnostics
- they do not yet expose the same clean final post-test metric tail as the newer content/context runs in the retained files we checked

Reading:
- these runs look useful for appendix diagnostics
- but until we verify their final saved summaries or retained checkpoint metrics cleanly, they should not be treated as the strongest headline quantitative evidence

### FACED conclusion
FACED is now one of the most important paper-facing datasets even though it is not yet the cleanest strict baseline matrix.
The cleanest new conclusions are:
- the newest context ablation is more informative than the newest content ablation
- typed block context looks slightly stronger than compact shared context
- `attn_mlp_latemix` is a reasonable alternative summary mode, but not a dramatic jump over `attn_delta4`
- FACED is still more useful for within-method evidence than for a clean dense-vs-selective baseline matrix

---

## ISRUC

### Inventory and format status
- 19 total runs
- all 19 runs have parseable args
- all 19 runs have final raw metrics
- 17 runs have final EMA metrics
- all 19 runs are newer summary-tracked runs

### What changed in this update
- no new log-count growth since the previous audit
- ISRUC remains one of the cleanest and most trustworthy support datasets in the repo

### Best observed ISRUC run
Best raw test run:
- `isruc_hier_1521786.out`
- raw test = `acc 0.80681 / kappa 0.78019 / f1 0.83062`
- EMA test = `acc 0.79972 / kappa 0.76805 / f1 0.82202`
- configuration:
  - `attnres_variant=pre_attn`
  - `moe=True`
  - `moe_attnres_depth_context_mode=dual_query_block_typed_proj`
  - `moe_attnres_depth_summary_mode=auto`
  - `moe_attnres_depth_summary_grad_mode=detached`

### ISRUC conclusion
ISRUC remains one of the strongest support datasets for the paper and is especially valuable because its gains do not depend on the affective-EEG setting.
No change to the main reading:
- ISRUC remains a strong support dataset for the paper
- typed block context is competitive and sometimes strongest
- `compact_shared + attn_delta4` is still a stable fallback
- EMA remains useful enough that it should not be silently dropped in final reporting

---

## Mumtaz

Paper-status note:
- exclude Mumtaz from the paper
- reason: current preprocessing is not subject-wise and therefore risks subject leakage

Internal audit note only:
- 8 total runs
- 7 with final raw binary-task metrics
- 1 with EMA metrics
- best clear binary retained run still appears to be:
  - `mumtaz_hier_1529683.out`
  - `acc 0.88997 / pr_auc 0.96158 / auroc 0.95473`

No change to the paper decision: keep Mumtaz out.

---

## PhysioNet-MI (`Phy_MI`)

### Inventory and format status
- 78 total runs
- all 78 runs have parseable args
- 75 runs have final raw metrics
- 17 runs have final EMA metrics
- 75 runs are newer summary-tracked runs

### What changed in this update
The new `1771944–1771954` family is very useful.
It adds a clean small boundary-case block around:
- regularization:
  - `physio_reg_none`
  - `physio_bank_reg`
- component learning-rate scaling:
  - `physio_component_lr`
  - `physio_bankreg_componentlr`
- compact router feature mode:
  - `physio_compact_none`
  - `physio_compact_eeg`
  - `physio_compact_psd`

### Parameter distribution snapshot
- `attnres_variant`: always `pre_attn`
- `use_ema`: `False` 61, `True` 17
- `moe=True` in all 78 runs
- `moe_attnres_depth_context_mode`: `compact_shared` 71, `dual_query_block_typed_proj` 7
- `moe_attnres_depth_summary_mode`: `attn_delta4` 71, `auto` 7
- `moe_attnres_depth_summary_grad_mode`: `delayed_unfreeze` 77, `detached` 1
- `moe_router_arch`: `mlp` 62, `linear` 16
- `use_component_lr`: `True` 60, `False` 18
- `moe_router_compact_feature_mode`: `none` 69, `psd_summary` 5, `eeg_summary` 4

### Best observed PhysioNet-MI runs
Best historical raw run still visible in the full retained log tree:
- `physio_moe_1522379.out`
- raw test = `acc 0.61633 / kappa 0.48830 / f1 0.61507`

Best historical EMA run still visible:
- `physio_moe_1522891.out`
- EMA test = `acc 0.60685 / kappa 0.47571 / f1 0.60875`

### What the newest PhysioNet-MI batch indicates
#### Regularization / component-LR block
- `physio_reg_none`: `0.60081 / 0.46760 / 0.60036`
- `physio_bank_reg`: `0.60137 / 0.46836 / 0.60195`
- `physio_component_lr`: `0.59248 / 0.45651 / 0.59105`
- `physio_bankreg_componentlr`: `0.61292 / 0.48384 / 0.61224`

Reading:
- bank regularization alone is only a tiny bump over no regularization
- component-LR alone is worse in this new batch
- the combination `bankreg + component_lr` is the strongest run in this newest focused batch and approaches the historical best

#### Compact-feature block
- `physio_compact_none`: `0.60015 / 0.46683 / 0.59817`
- `physio_compact_eeg`: `0.58466 / 0.44614 / 0.58401`
- `physio_compact_psd`: `0.59521 / 0.46019 / 0.59462`

Reading:
- in the new focused PhysioNet-MI batch, compact router features do not help
- `eeg_summary` is clearly worst of the three
- `psd_summary` is better than `eeg_summary`, but still below `compact_none`

### PhysioNet-MI conclusion
The new PhysioNet-MI results strengthen the boundary-case interpretation.
The cleanest reading is:
- this dataset still does not reward the full structured routing story as consistently as FACED / ISRUC / TUEV
- simpler or lightly regularized variants remain competitive
- compact router features are not helping here and should not be framed as universally useful
- the new batch is good appendix and boundary-case material, and part of it should likely appear in the main paper boundary-case subsection

---

## SEED-V

### Inventory and format status
- 65 total runs
- 64 runs with parseable args
- 60 runs with final raw metrics
- 17 runs with final EMA metrics
- 60 runs with newer summary bookkeeping

### What changed in this update
The most useful current SEED-V reference set is no longer just the first `1769781–1769798` seed-42 block.
We now have a clearer 3-seed family-level bundle spanning:
- dense finetuning
- AttnRes-only
- MoE-only without depth router
- compact-shared context
- block-shared typed context
- dual-query typed context

This gives us the best current SEED-V control panel for deciding what to keep, simplify, or remove from the adaptation block before new reruns.

### Best historical retained SEED-V runs
Best raw-accuracy retained run in the full log history:
- `seedv_SEEDV_TUNE_1503802.out`
- raw test = `acc 0.40940 / kappa 0.26723 / f1 0.41527`

Best raw-kappa retained run in the full log history:
- `seedv_SEEDV_TUNE_1502031.out`
- raw test = `acc 0.40828 / kappa 0.26911 / f1 0.41581`

Best EMA retained run in the full log history:
- `seedv_SEEDV_TUNE_1503998.out`
- EMA test = `acc 0.41076 / kappa 0.26982 / f1 0.41866`

### SEED-V registry of current reference families

This table should be treated as the current canonical SEED-V reference registry in `logs/`.
It is deliberately based on family-level means over the retained 3-seed EMA runs where available.

| Family | Run id | Seeds | AttnRes | MoE | Dispatch | Depth router | Context mode | Block count | Compact feature | Mean acc | Mean kappa | Mean wF1 |
|---|---:|---:|---|---|---|---|---|---:|---|---:|---:|---:|
| Dense finetune | `1769781` | 3 | `none` | `False` | `hard_capacity` | `False` | `compact_shared` | 4 | `eeg_summary` | 0.40186 | 0.25807 | 0.40909 |
| AttnRes-only | `1769783` | 3 | `pre_attn` | `False` | `hard_capacity` | `False` | `compact_shared` | 4 | `eeg_summary` | 0.40673 | 0.26576 | 0.41447 |
| MoE-only, no depth router | `1785556` | 3 | `pre_attn` | `True` | `soft` | `False` | `compact_shared` | 4 | `none` | 0.40720 | 0.26616 | 0.41525 |
| Compact-shared context | `1769798` | 3 | `pre_attn` | `True` | `soft` | `True` | `compact_shared` | 8 | `none` | 0.40477 | 0.26302 | 0.41304 |
| Block-shared typed context | `1769799` | 3 | `pre_attn` | `True` | `soft` | `True` | `block_shared_typed_proj` | 8 | `none` | 0.40752 | 0.26640 | 0.41481 |
| Dual-query typed context | `1769800` | 3 | `pre_attn` | `True` | `soft` | `True` | `dual_query_block_typed_proj` | 8 | `none` | 0.40768 | 0.26684 | 0.41506 |
| Older full selective dual reference | `1769784` | 3 | `pre_attn` | `True` | `soft` | `True` | `dual_query_block_typed_proj` | 4 | `none` | 0.40708 | 0.26646 | 0.41556 |

### What the current SEED-V registry already tells us

The current SEED-V story is best understood as a mechanism-control story rather than a headline-gain story.

The strongest current indications are:
- `AttnRes-only` already gives a clear and stable gain over dense finetuning.
- `MoE-only` without the depth router is already competitive with the fuller typed-context variants.
- `block_shared_typed_proj` and `dual_query_block_typed_proj` are effectively tied on the current 3-seed means.
- `compact_shared` is weaker than both `block_shared` and `dual_query` in the current grouped runs.

This means the current evidence supports a paper claim like:
- selective upper-layer adaptation is useful,
- AttnRes is the clearest stable ingredient,
- typed specialist capacity may help,
- but the extra depth-routing/context machinery is still close enough that it should not be overclaimed without tighter controls.

### Older diagnostic reference runs

Some older SEED-V runs are still important for mechanism interpretation even though they are not the current paper-facing registry:

| Purpose | Run id | Why it matters |
|---|---:|---|
| Dual-query instability reference | `1501564` | Shows dual-query soft-dispatch with growing depth-summary projection norms and more unstable branch behavior. Useful as evidence that some modules may be under-utilized or poorly conditioned rather than simply useless. |
| Dual-query norm-gated stabilization reference | `1504001` | Shows that norm-gated dual-query depth summaries can keep projected norms much smaller and more stable, but this alone does not clearly outperform the simpler 3-seed reference families. |

### SEED-V conclusion
SEED-V remains one of the most important datasets for the paper, but it now argues for discipline more than for hype:
- it supports the adaptation paper,
- it does not support a blanket claim that the fullest selective structure always wins,
- it already gives enough evidence to justify a core-structure refinement pass before larger reruns.

The cleanest current SEED-V message is:
- `AttnRes` is real,
- selective specialist capacity is plausible,
- compact context looks weak,
- dual-query is not yet justified over block-shared typed context,
- and the next missing controls should target dispatch mode, router necessity, and shared-bank vs dual-bank structure.

---

## TUEV

### Inventory and format status
- 56 total runs
- 55 runs with parseable args
- 49 runs with final raw metrics
- 2 runs with final EMA metrics
- 49 runs with newer summary bookkeeping

### What changed in this update
- no new log-count growth since the previous audit
- TUEV remains one of the most important main-paper datasets because its launcher already supports clean baseline toggles and because the retained logs already contain strong selective-routing results

### Best observed TUEV runs
Best raw-accuracy retained run:
- `tuev_1539249.out`
- raw test = `acc 0.67978 / kappa 0.64796 / f1 0.81893`

Best raw-kappa retained run:
- `tuev_1541528.out`
- raw test = `acc 0.64985 / kappa 0.66805 / f1 0.82608`

Best retained EMA run:
- `tuev_1539697.out`
- EMA test = `acc 0.63193 / kappa 0.64266 / f1 0.81615`

### TUEV conclusion
No change to the main reading:
- TUEV is still a strong main-paper dataset
- it supports the selective-adaptation story better than PhysioNet-MI does
- it now serves as the second completed control-style dataset beside SEED-V

---

## Cross-Dataset Update Summary

The new logs change the paper-facing interpretation in four important ways.

### 1. FACED is now better for appendix-strengthening ablations
The new FACED block gives us a cleaner view of:
- content mode
- context mode
- gradient-path behavior

The strongest clean new FACED takeaway is:
- typed block context is slightly stronger than compact shared context in the new context family

### 2. ISRUC remains strong and stable
No major shift here.
ISRUC still supports the method story cleanly.

### 3. PhysioNet-MI now more strongly supports the boundary-case narrative
The new Physio batch says:
- compact router features are not helping here
- regularization plus component-LR tuning can help somewhat
- the dataset still prefers simpler or more carefully regularized behavior than the “richest” structured story might suggest

### 4. SEED-V is now more nuanced than before
This is the biggest conceptual update.
The new matched seed-42 baseline block says:
- `dense < AttnRes-only`
- `full selective` is not yet beating `AttnRes-only` in that clean family

That means the paper should not currently claim:
- “full model decisively beats all matched SEED-V baselines”

Instead it can safely say:
- AttnRes is a strong adaptation ingredient
- selective routing remains promising but configuration-sensitive
- the clean multi-seed version of this matrix is now available and supports a more disciplined control-style interpretation

---

## Paper-Facing Framing After the Latest Logs

The strongest honest presentation is no longer a single monotonic story in which one final recipe dominates every dataset. Instead, the logs support a two-layer evidence structure:

### 1. Confirmatory adaptation evidence
These are the datasets that best test whether the proposed adaptation mechanism can beat dense finetuning under tightly matched conditions.
- `SEED-V`: strongest completed multi-seed ablation family, but the gains over dense are modest and mostly captured by AttnRes plus specialist adaptation
- `TUEV`: strongest clinical-event control dataset and now a completed multi-seed control family

### 2. Strong supportive breadth evidence
These are the datasets that best show where the method can deliver visibly larger practical gains or richer within-method structure.
- `FACED`: strongest affective dataset for visibly larger gains and clear context/content comparisons
- `ISRUC`: strongest physiological-state support dataset, with consistently good typed-context runs
- `PhysioNet-MI`: boundary-case dataset showing that simpler or more regularized adaptation can be competitive

That means the paper should not be written as if only `SEED-V` and `TUEV` matter. A better reviewer-facing framing is:
- `SEED-V` and `TUEV` provide the strict matched adaptation controls
- `FACED` and `ISRUC` provide the clearest evidence that the method can yield stronger practical benefits under heterogeneous downstream shift
- `PhysioNet-MI` provides the failure-mode / boundary-case analysis

This is still scientifically defensible as long as the paper clearly labels which evidence is confirmatory and which is supportive. The strongest datasets should lead the narrative, but they should not be mixed deceptively into one status-obscuring benchmark table.

## What These New Logs Mean for the Paper

### Safe claims strengthened by the new logs
- `FACED` now has cleaner within-method context/content evidence
- `PhysioNet-MI` now has a clearer simplification/boundary-case story
- `SEED-V` now has a cleaner matched baseline block, even though its outcome is more mixed than the older broad sweep suggested

### Claims that became riskier
- any claim that “the full selective model clearly wins on SEED-V” is now too strong without multi-seed confirmation
- any claim that compact router features help uniformly is weaker now because the new PhysioNet-MI batch argues against that
- any claim that typed depth-aware routing is universally the main driver of gains should be softened until the clean matched ablation matrix is complete

### Best current paper framing after this update
The current evidence supports:
- AttnRes is a strong and likely necessary part of the adaptation story
- selective routing is promising and can help, but its gains are more configuration-sensitive than a simple headline would suggest
- typed context structure looks useful on FACED and ISRUC
- PhysioNet-MI remains a genuine boundary case where simpler or differently regularized adaptation may suffice

---

## Remaining Ablations That Matter Most

These are still the most important unfinished or not-yet-paper-clean comparisons.

Main-paper critical:
- multi-seed `SEED-V` matrix for:
  - dense
  - AttnRes-only
  - MoE-only without depth router
  - full selective
- completed multi-seed `TUEV` matrix for:
  - dense
  - AttnRes-only
  - full selective
- `PhysioNet-MI` simplification panel using the new focused family

Appendix-important:
- full FACED gradient-path comparison once its retained metrics are cleanly confirmed
- FACED content and context ablations
- full PhysioNet-MI compact-feature block
- full PhysioNet-MI regularization/component-LR block
- richer routing diagnostics for FACED / ISRUC / TUEV

---

## SLURM Execution Guidance

### What is already well supported
Best current launchers for clean baseline matrices:
- `scripts/SEED-V/submit_seedv_train.slurm`
- `scripts/TUEV/submit_train.slurm`

Best current launchers for within-method sweeps:
- `scripts/FACED/submit_train.slurm`
- `scripts/ISRUC/submit_train.slurm`
- `scripts/PHYSIO-MI/train_physio_compact_shared.slurm`

### What the newest logs say we should prioritize next
1. `SEED-V` multi-seed repeat of the new manual family
- dense
- AttnRes-only
- full selective
- compact-context selective

2. `TUEV` clean matched baseline matrix is now completed
- dense
- AttnRes-only
- full selective

3. one explicit `MoE-only without depth summary` control
- ideally on `SEED-V`
- alternatively on `TUEV`

4. `PhysioNet-MI` finalize the simplification table
- `reg_none`
- `bank_reg`
- `component_lr`
- `bankreg_componentlr`
- `compact_none`
- `compact_psd`
- `compact_eeg`

5. `FACED` keep the new context/content block for appendix and mechanism analysis

---

## Bottom Line

The new logs improve the paper, but they also make the story more honest.

Strongest updates:
- `FACED` now has cleaner within-method ablations
- `PhysioNet-MI` now has a much clearer boundary-case analysis block
- `SEED-V` now has a clean matched baseline family

Most important scientific consequence:
- the new clean `SEED-V` family says `AttnRes-only` is currently the strongest matched seed-42 baseline in that block, not the full selective model

So the paper should currently be formulated as:
- a study of structured downstream adaptation for pretrained EEG backbones
- with strong evidence that AttnRes matters
- with promising but configuration-sensitive evidence for richer selective routing
- and with an explicit boundary-case analysis showing when simpler adaptation may be enough

That is a stronger and more defensible paper than a forced “we win everywhere” story.
