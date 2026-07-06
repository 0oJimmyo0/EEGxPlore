# EEGxPlore Next-Step Plan

## Current Status

We have now shown that the current ACCRE setup can run:

- the original `EEGxPlore` SEED-V pipeline,
- LaBraM baseline finetuning on our existing SEED-V data format,
- LaBraM + EEGxPlore-style selective adaptation blocks on the same data.

The practical meaning of this is important:

- the infrastructure question is mostly solved for SEED-V,
- backbone substitution is feasible in code,
- the main remaining problem is now scientific and architectural rather than environment-related.

At the moment, the most honest summary is:

- `AttnRes`-style adaptation is stable and usually competitive,
- the full selective MoE path is feasible but fragile,
- gains over a dense LaBraM baseline are currently modest and configuration-sensitive on SEED-V.

That does **not** invalidate the project direction.
It just means the strongest paper claim should currently be:

> Our adaptation block is useful as a lightweight EEG-backbone adaptation mechanism, and it can provide gains on existing EEG backbones, but the expert-routing part still needs refinement to become consistently reliable.

---

## Immediate Challenge

The current challenge is **not** “can we run LaBraM?”.
We can.

The current challenge is:

> Why does the selective adaptation path sometimes underperform or become fragile relative to dense finetuning, and which part of the adaptation block is responsible?

The main symptoms seen so far on SEED-V are:

- specialist gradients are often much smaller than backbone gradients,
- routing tends to become shortcut-prone or collapse-prone,
- the shared FFN path may dominate too strongly,
- extra routing context can add complexity without guaranteed benefit.

So the next stage should focus on three direct hypotheses:

1. the shared path may be overpowering the specialists,
2. the experts may be too weak to matter even when they are selected,
3. the current router target may be more complicated than SEED-V needs.

---

## Immediate Engineering Response

The codebase should now support direct tests for:

- reduced shared-path dominance,
- increased expert contribution,
- simplified router input targets.

Concretely, the next LaBraM-on-SEED-V diagnosis runs should prioritize:

- lower shared-path contribution,
- higher effective expert contribution,
- simpler router inputs such as `delta_only` rather than the full baseline+attnres+delta summary,
- depth-router off vs on only when the comparison is paired and controlled.

We should treat these as **mechanism diagnosis** runs, not headline-result runs.

## Objective

The immediate priority is **not** to add more backbones.
The immediate priority is to determine whether the current CBraMod adaptation approach is actually well-designed.

The practical workflow should be:

1. use `SEED-V` as the small, clean mechanism sandbox,
2. simplify or revise the CBraMod method there first,
3. rerun the remaining 4 retained datasets only after the CBraMod structure stabilizes,
4. then add PEFT comparisons against the stabilized method,
5. only after that, port the finalized method to 1 and then 2 extra backbones.

The core research question for the next phase is:

> Which components of the current adaptation mechanism actually improve downstream transfer, and which are redundant, unstable, or unnecessary?

---

## Main Scientific Reframing

The paper should not overclaim that depth-aware routing is the entire contribution.

A stronger and more honest story may be:

- downstream EEG foundation-model adaptation benefits from **selective upper-layer modification**,
- the most robust gains come from **depth aggregation** and **structured lightweight specialist capacity**,
- routing and compact context should be presented as useful **only if confirmed by paired seed-complete ablations**.

This means:

- `AttnRes` may become the main stable ingredient,
- typed specialists may remain only if they beat parameter-matched generic PEFT or generic expert-bank controls,
- routing and compact context stay in the main method only if they survive controlled seed-complete comparisons.

---

## Components At Risk Of Removal

The following components should be treated as provisional until the decomposition audit is complete:

- compact EEG router context
- PSD router context
- dual-query routing
- learned depth-aware router
- `hard_capacity` dispatch if it is not clearly better than `soft`
- two-bank specialist design if parameter-matched generic adapters or generic expert banks perform similarly

These components are not guaranteed to remain in the official method.

---

## Phase Order

## Phase A: CBraMod Trust/Reproducibility Cleanup

Purpose:

- eliminate method/paper/code mismatch,
- make every main number traceable,
- ensure we are comparing the correct official implementation.

### Required tasks

1. inspect the paper text, code defaults, and training scripts for consistency
2. resolve whether official runs used:
   - `soft` dispatch
   - `hard_capacity` dispatch
3. align:
   - paper method description
   - `finetune_main.py` defaults
   - Slurm launchers
   - result tables
4. ensure every main-paper number maps to:
   - log file
   - seed
   - checkpoint id
   - generated summary row
5. stop using headline/best values as the main evidentiary object
6. move headline provenance to appendix
7. make main tables use:
   - mean ± std
   - paired delta vs dense finetuning

### Concrete audit targets

- dispatch mode in paper vs code
- `soft` vs `hard_capacity`
- result-generation scripts that currently use hand-curated constants
- seed-complete summaries vs headline values

### Success criteria

- no method-description mismatch remains,
- no official result is ambiguous about dispatch mode or seed provenance,
- the main table can be regenerated from log-derived summaries.

---

## Phase B: SEED-V-First CBraMod Mechanism Decomposition Audit

Purpose:

- isolate which components matter,
- remove unnecessary complexity,
- identify the simplest stable recipe,
- do it first on the smallest clean control dataset before spending compute on the other 4 datasets.

### Why SEED-V first

- it is already the cleanest current control dataset,
- it has the most interpretable dense vs AttnRes vs MoE-style family,
- it is cheaper and faster to iterate on than a full 5-dataset sweep,
- if a component fails here, it usually does not deserve immediate expansion.

### Seed policy

Preferred seeds:

- `42`
- `3407`
- `2024`
- `11`
- `123`

If compute is limited:

- use `42`, `3407`, `2024` first,
- but keep the same seeds across all variants.

### Main rule

Do **not** rerun all 5 datasets while the CBraMod method is still structurally moving.

If SEED-V reveals structural changes:

- simplify the method first,
- then rerun the remaining 4 CBraMod datasets with the revised structure.

---

## Prioritized Experiment Table

| Priority | Family | Variants | Datasets | Purpose |
|---|---|---|---|---|
| P0 | Trust audit | official script/default check; soft vs hard-capacity | SEED-V first, then TUEV if needed | determine what method was actually run |
| P1 | Baseline controls | frozen+head, dense, top-k finetuning | SEED-V | establish simple downstream controls |
| P1 | Depth aggregation | dense, AttnRes-only, AttnRes-lite/frozen if feasible | SEED-V | test whether AttnRes is the main robust gain |
| P1 | Capacity controls | generic expert bank, typed bank + uniform routing, typed bank + no learned routing | SEED-V | test whether typed capacity matters beyond extra capacity |
| P1 | Routing isolation | random, uniform, shuffled-depth, content-only, depth-only, content+depth, frozen router, learned router | SEED-V | test whether learned routing is meaningful |
| P1 | Specialist structure | shared bank, spatial-only, spectral-temporal-only, typed dual-bank | SEED-V | test whether dual-bank structure matters |
| P1 | Context controls | no compact context, EEG summary, PSD summary, dual-query, block-shared | SEED-V | test whether compact/dual-query context really helps |
| P1.5 | Early PEFT sanity check | adapter, LoRA | SEED-V | avoid late surprise that generic PEFT already matches gains |
| P2 | Official method selection | simplest stable recipe chosen from SEED-V audit | SEED-V | define official CBraMod method |
| P3 | 4-dataset CBraMod confirmation | dense, AttnRes-only, minimal official method, optional legacy full method | FACED, ISRUC, TUEV, PhysioNet-MI | confirm the revised method transfers across the current suite |
| P4 | Full PEFT matrix | adapter, LoRA, residual adapter matched in params if possible | all 5 datasets or at least 3 strongest ones | compare the stabilized method to standard downstream finetuning alternatives |
| P5 | First added backbone | finalized minimal method only | one backbone first | portability after method stabilization |
| P6 | Second added backbone | finalized minimal method only | if time/compute remain | stretch validation |

---

## Controlled SEED-V Audit Matrix

## A. Baseline controls

Run:

- frozen backbone + classifier head
- dense finetuning
- top-k finetuning

Questions answered:

- how much transfer is already available without complex adaptation?
- does selective adaptation beat simple selective finetuning?

## B. Depth aggregation controls

Run:

- dense finetuning
- AttnRes-only
- AttnRes with lite/frozen adaptation if feasible

Questions answered:

- is depth aggregation the main stable effect?
- is it already enough to explain most of the gain?

## C. Capacity controls

Run:

- generic expert bank or generic MoE without typed spatial/spectral-temporal separation
- typed spatial/spectral-temporal bank with uniform routing
- typed bank with no learned routing

Questions answered:

- is typed specialist structure better than generic extra capacity?
- is learned routing even necessary once typed banks exist?

## D. Routing-isolation controls

Run:

- random router
- uniform router
- shuffled depth-summary router
- content-only router
- depth-only router
- content + depth router
- frozen router
- learned router

Questions answered:

- is the learned router actually useful?
- is depth information meaningful?
- does current-layer content do most of the work?

## E. Specialist-structure controls

Run:

- one shared specialist bank
- spatial-only specialists
- spectral-temporal-only specialists
- typed dual-bank specialists

Questions answered:

- do we really need two typed banks?
- is one branch doing almost all the work?

## F. Context controls

Run:

- no compact context
- compact EEG summary context
- PSD summary context
- dual-query context
- block-shared context

Questions answered:

- does compact router context help at all?
- is dual-query better than block-shared on seed-complete evidence?

## G. Early PEFT sanity check

Run:

- standard adapter
- LoRA

Questions answered:

- are we at risk of discovering too late that generic PEFT already explains the current gain?
- should PEFT be prioritized earlier if the gap is small?

---

## Key Comparisons To Prioritize

The most important comparisons are:

1. `AttnRes-only` vs dense finetuning
2. typed uniform specialists vs generic expert bank
3. learned routing vs uniform / random / shuffled-depth / content-only routing
4. content + depth router vs content-only and depth-only router
5. full current recipe vs the simplest strong recipe
6. compact context vs no context
7. dual-query vs block-shared routing
8. early adapter / LoRA sanity check vs the best simplified CBraMod variant

These comparisons should determine the official method, not novelty preference.

---

## Decision Tree For Keeping Or Removing Components

## Step 1: Dispatch mode

If `soft` and `hard_capacity` give similar mean performance:

- keep the simpler, cleaner, easier-to-explain option
- make it the only official method

If one clearly wins on paired seed-complete means:

- keep the winner
- demote the other to appendix

## Step 2: Depth aggregation

If `AttnRes-only` consistently beats dense:

- keep `AttnRes`
- treat it as a core method component

If `AttnRes-only` does not improve paired seed means:

- reconsider whether depth aggregation is really central

## Step 3: Typed specialists

If typed specialists beat generic expert-bank controls and later beat PEFT controls:

- keep typed specialist structure

If typed specialists match but do not exceed generic alternatives:

- reframe them as one implementation choice, not the main novelty

## Step 4: Learned routing

If learned routing does not beat:

- uniform routing
- random routing
- shuffled-depth routing
- content-only routing

then:

- remove learned depth-aware routing from the official method
- or demote it to exploratory appendix evidence

## Step 5: Compact EEG / PSD context

If compact EEG/PSD context does not improve seed-complete mean performance:

- remove it from the main method

If it only improves selected headline runs:

- keep it out of the official method

## Step 6: Dual-query routing

If dual-query improves only selected headline runs but not paired seed means:

- remove it from the official method
- keep block-shared if it is simpler and comparably strong

## Step 7: Two-bank specialist design

If one shared bank performs similarly to typed dual-bank specialists:

- simplify to one shared bank

If spatial-only or spectral-temporal-only explains almost all benefit:

- consider simplifying the adaptation structure

## Final rule

Prefer the **simplest stable method** over the most complex method.

Novelty alone is not a reason to keep a component.

---

## Phase C: Select Minimal Official CBraMod Method

After the SEED-V decomposition audit, choose the minimal official CBraMod variant.

Possible outcome:

- `CBraMod + AttnRes + typed residual specialists + simple content/depth router`

without:

- compact EEG context
- PSD context
- dual-query routing
- unnecessary dispatch complexity

The official method should be the version that:

- gives stable paired gains,
- survives PEFT sanity checks,
- survives routing-isolation checks,
- is the simplest defensible mechanism.

---

## Phase D: 4-Dataset CBraMod Confirmation

Only after the official CBraMod method is selected on SEED-V should we rerun the remaining 4 retained datasets:

- `FACED`
- `ISRUC`
- `TUEV`
- `PhysioNet-MI`

### Methods to include

- dense finetuning
- AttnRes-only
- minimal official method
- optional full current method if still informative as a legacy comparison

### Goal

- confirm that the revised method transfers across the existing CBraMod evaluation suite,
- avoid spending PEFT or cross-backbone compute on a moving target.

---

## Phase E: Full PEFT Comparison On The Stabilized Method

Once the CBraMod structure is stable, add:

- adapter
- LoRA
- residual adapter matched in params if possible
- frozen + head if not already included in every dataset family
- top-k finetuning where useful

### Why PEFT comes here

- otherwise we risk comparing PEFT against a method that we later change,
- reviewer-facing PEFT results are more valuable once the official method is fixed.

### Metrics to report

- balanced accuracy
- Cohen’s kappa
- weighted F1
- trainable parameters
- total parameters
- peak GPU memory if available
- wall-clock training time if available
- mean ± std across seeds
- paired delta vs dense finetuning
- per-seed dot plots

---

## Phase F: Only Then Revisit Cross-Backbone Expansion

After Phases A-E:

- port the **finalized minimal method**
- first to one additional backbone
- then to a second backbone only if time and compute remain

This keeps cross-backbone work scientifically clean:

- we test portability of the distilled method,
- not a still-changing CBraMod-specific full recipe.

Preferred order:

1. `LaBraM`
2. `CSBrain` or `REVE-base` if implementation is clean
3. `EEGPT` as a fallback if newer model integration is cheaper or more stable

---

## Compact List Of Code Files Likely Needing Changes

## Core training/config

- `finetune_main.py`
- `finetune_trainer.py`
- `finetune_evaluator.py`

## Backbone/model definition

- `models/cbramod.py`
- `models/criss_cross_transformer.py`
- `models/moe.py`
- `models/attn_res.py`

## Dataset-specific wrappers

- `models/model_for_seedv.py`
- `models/model_for_tuev.py`
- `models/model_for_faced.py`
- `models/model_for_isruc.py`
- `models/model_for_physio.py`

## Launchers and experiment scripts

- `scripts/SEED-V/submit_seedv_train.slurm`
- `scripts/TUEV/submit_train.slurm`
- `scripts/FACED/submit_train.slurm`
- `scripts/ISRUC/submit_train.slurm`
- `scripts/PHYSIO-MI/train_physio_compact_shared.slurm`
- `scripts/run_seedv.sh`
- dataset-specific helper scripts for summary extraction

## Result/provenance generation

- paper result asset builders under `paper/.../tools/`
- especially `build_paper_assets.py`
- summary scripts that currently encode hand-curated values

## Paper text files

- `paper/.../sec/method.tex`
- `paper/.../sec/results.tex`
- `paper/.../sec/experiments.tex`
- `paper/.../appendix/implementation_details.tex`
- `paper/.../appendix/extended_ablations.tex`
- `paper/.../appendix/reproducibility.tex`

---

## Immediate Next Step

The immediate next step should be a **SEED-V trust + mechanism bootstrap**, not a large rerun.

## Immediate goal

Produce one small, reliable, **output-driven** SEED-V audit bundle that tells us:

1. what dispatch mode the official results actually used,
2. whether `AttnRes-only` is already the strongest simple method,
3. which adaptation modules are clearly ineffective,
4. which modules appear under-utilized rather than intrinsically bad,
5. whether learned routing adds anything over simple typed specialists,
6. whether compact context and dual-query are likely removable.

This immediate step should not just compare final metrics.
It should use the training outputs and routing diagnostics to decide whether each module should be:

- kept,
- simplified,
- reparameterized,
- or removed before the full SEED-V rerun.

## Immediate technical actions

### 1. Freeze the official SEED-V baseline recipe

Audit and document the exact current SEED-V families:

- dense
- AttnRes-only
- MoE-only / typed no-router
- full current method

Technically, this means:

- inspect `scripts/SEED-V/submit_seedv_train.slurm`
- inspect `scripts/run_seedv.sh`
- inspect `finetune_main.py` defaults
- inspect `models/model_for_seedv.py` logging
- map each reported family to actual run ids in `logs/SEED-V/out`

Deliverable:

- one small CSV or markdown table listing:
  - family name
  - run script
  - seed
  - dispatch mode
  - context mode
  - summary mode
  - test metrics

### 2. Add one explicit SEED-V experiment registry

Create a single machine-readable registry for the SEED-V audit variants.

This can be:

- a CSV under `docs/` or `output/`
- or a small JSON/YAML manifest

Fields should include:

- `family`
- `seed`
- `dispatch_mode`
- `attnres_variant`
- `use_moe`
- `specialist_structure`
- `router_variant`
- `context_variant`
- `model_dir`
- `log_path`

Why this matters:

- it prevents run drift,
- it makes the mechanism audit reproducible,
- it makes later summary generation much easier.

### 3. Add an output-driven module audit before structural edits

Before deciding what to delete or keep, extract the current SEED-V evidence from model outputs and diagnostics.

For each run family, collect:

- final metrics: `BA`, `kappa`, `wF1`
- paired seed deltas vs dense
- router assignment histograms
- effective-expert usage per bank
- routing entropy before and after dispatch
- spatial vs spectral residual norms
- compact-context gate values and warmup scales
- depth-summary norm, grad flow, and unfreeze behavior
- domain-bias magnitude, if enabled

Interpretation rule:

- a module is **ineffective** if it does not improve matched-seed metrics and its diagnostics do not show meaningful use,
- a module is **under-utilized** if the metrics are weak but the diagnostics show collapse, near-zero gates, frozen gradients, or branch imbalance that suggests the design is not being exercised properly,
- only after this audit should we decide whether to remove the module or rework it.

Deliverable:

- one SEED-V audit table or JSONL summary with one row per run family and seed,
- one short decision note for each candidate component:
  - `AttnRes`
  - typed specialists
  - learned router
  - compact EEG/PSD context
  - dual-query context
  - two-bank structure
  - dispatch mode

### 4. Implement the first SEED-V simplification batch

Run these first on seeds `42`, `3407`, `2024`:

- dense
- AttnRes-only
- typed specialists with uniform routing
- typed specialists with learned routing
- full current method
- no compact context
- block-shared
- dual-query

This is the smallest useful batch that can already answer:

- does AttnRes carry most of the gain?
- does learned routing help?
- do compact context and dual-query matter?

The first edit cycle after this batch should focus on:

- removing modules that are both weak in metrics and inactive in diagnostics,
- simplifying modules that show partial utility but unstable behavior,
- preserving only the smallest adaptation block that still gives a consistent gain.

### 5. Add one early PEFT sanity check

Before doing a full PEFT matrix, add at least:

- adapter
- LoRA

on SEED-V with the same 3 seeds.

Why:

- this prevents us from spending a week refining a method that generic PEFT already matches,
- but it still avoids a full PEFT sweep before the CBraMod method stabilizes.

## Immediate files to touch

Most likely first changes:

- `scripts/SEED-V/submit_seedv_train.slurm`
- `scripts/run_seedv.sh`
- `finetune_main.py`
- possibly `models/moe.py` or `models/cbramod.py` only if we need cleaner switchable routing ablations
- one new summary/registry file under `docs/` or `output/`

## Immediate success criteria

By the end of this immediate step, we should have:

1. a verified official SEED-V baseline recipe,
2. a small SEED-V ablation matrix with matching seeds,
3. a module-level audit saying which parts are ineffective vs under-utilized,
4. enough evidence to choose whether the current CBraMod structure needs simplification before rerunning the other 4 datasets.

---

## Recommended Order Of Runs For The Next 1–2 Weeks

## Week 1

### Day 1–2: SEED-V trust audit

1. confirm dispatch mode used in official SEED-V runs
2. audit paper/code/script mismatch
3. identify which result tables are log-derived vs hand-curated
4. create a SEED-V experiment registry

### Day 3–5: SEED-V mechanism bootstrap

Run on the same 3 seeds:

- dense
- AttnRes-only
- typed uniform specialists
- typed learned specialists
- full current method
- block-shared
- dual-query
- no compact context

### Day 6–7: early PEFT sanity check

Run:

- adapter
- LoRA

on the same SEED-V seeds.

Decision checkpoint at end of Week 1:

- keep/remove compact context
- keep/remove dual-query
- keep/remove learned routing
- decide whether AttnRes-only or a simple typed-specialist variant is the real stable core

## Week 2

### Day 8–10: select official minimal CBraMod method

Use the SEED-V audit to lock:

- dispatch mode
- routing structure
- context structure
- specialist structure

### Day 10–14: rerun remaining 4 CBraMod datasets

Run on:

- `FACED`
- `ISRUC`
- `TUEV`
- `PhysioNet-MI`

using:

- dense
- AttnRes-only
- minimal official method
- optional full current method if still informative

If Week 2 goes well, then prepare the broader PEFT sweep next.

---

## Bottom Line

The project should now optimize for:

1. trust
2. SEED-V-first mechanism clarity
3. simplicity
4. 4-dataset CBraMod confirmation
5. only then full PEFT comparison
6. only then portability to new backbones

The current question is not:

> How many backbones can we add?

The current question is:

> What is the minimal robust CBraMod adaptation method that genuinely improves downstream transfer?

Once that answer is clear, PEFT comparison and cross-backbone expansion become much easier to defend.
