# ICASSP paper framework

## Working title

**Task-Dependent Effects of Cross-Depth Aggregation and Typed Specialists for EEG Foundation Model Fine-Tuning**

The title may be shortened for the final submission, but it should preserve
the paper's actual scope: one CBraMod backbone, cross-depth aggregation with
typed specialists, and a four-dataset accuracy--compute evaluation.

## 1. Paper identity

The paper asks:

> Under a fixed CBraMod fine-tuning protocol, when does adding cross-depth
> AttnRes aggregation and typed spatial/spectral specialists help, and what
> computational cost does it introduce across EEG tasks?

The paper is a controlled study of **cross-depth aggregation and typed
specialist augmentation under full-backbone fine-tuning**. It is not a new
backbone, a broad routing search, or a strict parameter-efficient fine-tuning
method. The confirmed `specialist_augmented_full` condition fine-tunes the
full CBraMod backbone and adds one upper-layer specialist block. “Selective”
may remain as an internal executable condition name, but it should not be used
in the manuscript as a frozen-backbone or PEFT claim.

The central comparison is:

| Reference | Proposed condition |
|---|---|
| Full CBraMod fine-tuning | AttnRes + typed spatial/spectral specialists |

Both conditions use the same CBraMod backbone, dataset protocol, seed set,
checkpoint-selection rule, and evaluation metrics. The method recipe is locked;
depth-aware routing is disabled in the paper-facing condition.

The primary estimand is the effect of the **combined augmentation**. The
four-page paper should not imply a factorial causal separation of AttnRes and
specialists; the Full-vs-Augmented comparison is the planned component control.

## 2. Contributions to claim

The manuscript should make three compact contributions:

1. **A controlled augmentation comparison.** We evaluate the joint effect of
   cross-depth pre-attention aggregation and typed spatial/spectral specialists
   on CBraMod under matched training protocols.
2. **A multi-dataset, multi-seed evaluation.** We evaluate four EEG settings
   (FACED, ISRUC, SEED-V, and PhysioNet-MI) over three prespecified seeds and
   report balanced accuracy, weighted-F1, and Cohen's kappa as mean ± SD.
3. **A task-dependent accuracy–compute characterization.** The augmentation
   improves some task regimes, remains approximately neutral in another, and
   degrades the subject-disjoint PhysioNet-MI setting while increasing runtime
   and memory. This establishes a reproducible trade-off rather than a
   universal-improvement claim.

The contribution is strongest as a compact task-dependence study: the
augmentation's benefit is conditional on the dataset/task regime, and its cost
is measurable.

The frozen results support this interpretation directly: the specialist
condition improves the dataset mean on ISRUC and SEED-V, is effectively tied
with Full fine-tuning on FACED, and is lower on PhysioNet-MI.

## 3. Claims that must not appear

Do not claim that:

- specialists improve every dataset or every random seed;
- the method is parameter-efficient in the strict frozen-backbone sense;
- routing is universally beneficial;
- depth-conditioned routing is necessary or part of the main method;
- PhysioNet-MI and SEED-V absolute scores are directly comparable;
- the four datasets use identical split semantics;
- the results reproduce unavailable rejected-paper checkpoints;
- the paper extends the TMLR interaction-alignment study.

Use “dataset-dependent effect,” “controlled accuracy–compute trade-off,” and
“full-backbone fine-tuning with cross-depth aggregation and typed specialists.”

## 4. Four-page narrative

The official template provides four pages for the manuscript body and permits
an additional fifth page containing references only. The exact layout can move
slightly after the template is populated, but the information hierarchy should
remain fixed.

### Page 1: motivation, question, and contribution

**Opening problem.** EEG foundation models are transferred to downstream tasks
that differ substantially in label structure, subject variability, channel
configuration, and recording regime. Full fine-tuning is a strong baseline,
but whether additional structured adaptation capacity consistently helps
across such heterogeneous tasks remains unclear.

**Gap.** Existing work has demonstrated several structured adaptation
strategies, but their benefit relative to ordinary full fine-tuning across
heterogeneous EEG tasks and their computational cost remain insufficiently
characterized.

**Question.** Does the locked cross-depth AttnRes + typed-specialist
augmentation provide a reliable benefit over Full CBraMod fine-tuning, and
what is its overhead?

**Contributions.** Use the three contributions above in a short numbered list.

End the introduction with the key framing sentence:

> We study specialist augmentation as a task-dependent accuracy–compute
> trade-off, rather than assuming that additional cross-depth and specialist
> capacity is uniformly beneficial.

Avoid a long related-work section. Use one paragraph to position CBraMod
adaptation and specialist/routing methods, then move to the question.

### Page 2: method and locked protocol

Include one compact architecture figure showing:

`EEG input → CBraMod blocks → cross-depth AttnRes → typed spatial/spectral specialists → classifier`

Earlier hidden states should feed the cross-depth AttnRes path, followed by the
top-layer representation entering the spatial/spectral specialists. The
figure should show that the specialist block is an added upper-layer module,
the full backbone remains trainable, and no archived depth-routing branches
are part of the proposed method.

Describe only the implementation details needed to reproduce the comparison:
cross-depth pre-attention AttnRes beginning at the first encoder layer, one
upper specialist layer with four experts per spatial/spectral branch, and
sample-wise soft routing. We use the same locked specialist configuration
across datasets, without PSD/context features, domain bias, depth-aware
routing, or component-specific learning rates. Checkpoints are selected by
validation kappa and each condition uses seeds 3407, 2024, and 2027.

Evaluate four downstream tasks: FACED and SEED-V for emotion recognition,
ISRUC for sleep staging, and PhysioNet-MI for motor imagery. Dataset-specific
optimization and split protocols are fixed within each paired comparison;
PhysioNet-MI uses a subject-disjoint split. The exact epoch, batch, LMDB, and
manifest details remain in the released artifact. State explicitly that the
source split semantics differ by dataset and that absolute scores are not
compared across datasets.

### Page 3: main results

Use one full-width four-row table with the three primary metrics. Each method
cell reports BA / weighted-F1 / kappa as mean ± SD. A second full-width table
reports paired ΔBA, Δweighted-F1, Δkappa, positive-seed count, and compute
ratios. This exposes the multi-metric evidence without adding a third table.

Recommended table structure:

| Dataset | Full FT BA / wF1 / κ | + AttnRes + Specialists BA / wF1 / κ |
|---|---|---|
| FACED | mean ± SD / mean ± SD / mean ± SD | mean ± SD / mean ± SD / mean ± SD |
| ISRUC | mean ± SD / mean ± SD / mean ± SD | mean ± SD / mean ± SD / mean ± SD |
| SEED-V | mean ± SD / mean ± SD / mean ± SD | mean ± SD / mean ± SD / mean ± SD |
| PhysioNet-MI | mean ± SD / mean ± SD / mean ± SD | mean ± SD / mean ± SD / mean ± SD |

The current frozen values imply the following text:

- ISRUC shows the clearest positive effect: approximately +1.27 percentage
  points in balanced accuracy and +2.40 kappa points.
- SEED-V shows a smaller positive mean effect: approximately +0.77 points in
  balanced accuracy and +1.09 kappa points.
- FACED is effectively neutral/slightly negative.
- PhysioNet-MI is consistently negative for the specialist condition across
  all three seeds.

The result paragraph should emphasize the paired pattern rather than repeat
the table row by row: ISRUC and PhysioNet-MI have nearly equal but opposite
BA shifts, with consistent signs in all three seeds; SEED-V has a modest,
seed-sensitive gain; and FACED has no reproducible advantage at this seed
budget. BA, weighted-F1, and kappa agree in the direction of the mean effect
on all four datasets. This mixed pattern is the task-dependence result.

Write the Results section as two short subsections:

#### 4.1 Task-dependent performance

State the four dataset outcomes, followed by the positive-seed consistency
sentence. Do not interpret the combined augmentation as separate AttnRes or
specialist effects.

#### 4.2 Accuracy–compute trade-off

Use Table 2 and one paragraph to report parameter, runtime, and memory overhead.

### Frozen paper-quality result values

The following are the current manuscript-ready table values. Performance
values are BA / weighted-F1 / kappa in percent; deltas are in percentage
points. All values are mean ± sample SD over the three prespecified seeds
(`3407`, `2024`, and `2027`). Positive-seed counts refer to paired seeds with
a positive BA delta. Use the generated JSON/Markdown artifacts for unrounded
values and complete per-seed metrics.

| Dataset | Full CBraMod FT (BA / wF1 / κ) | AttnRes + Specialists (BA / wF1 / κ) |
|---|---|---|
| FACED | 58.56 ± 0.80 / 58.65 ± 0.84 / 53.03 ± 0.86 | 58.44 ± 0.44 / 58.48 ± 0.42 / 52.88 ± 0.47 |
| ISRUC | 77.52 ± 0.93 / 78.66 ± 1.25 / 72.56 ± 1.48 | 78.79 ± 1.02 / 80.56 ± 1.09 / 74.97 ± 1.31 |
| SEED-V | 39.36 ± 0.17 / 40.04 ± 0.18 / 24.74 ± 0.17 | 40.13 ± 0.67 / 40.97 ± 0.60 / 25.83 ± 0.90 |
| PhysioNet-MI | 63.11 ± 1.27 / 63.14 ± 1.25 / 50.80 ± 1.70 | 61.84 ± 1.02 / 61.86 ± 1.06 / 49.09 ± 1.35 |

The manuscript caption should say: “Test performance under matched full
fine-tuning and cross-depth specialist augmentation. Results are mean ± sample
SD over three seeds; all scores are multiplied by 100 for compact
presentation.”

The table should be read within each dataset. Absolute scores across datasets
are not comparable because the tasks, labels, splits, and recording regimes
differ. The important evidence is the paired delta under the same dataset
protocol. Weighted-F1 follows the same overall cross-task pattern and is
shown in Table 1 rather than being relegated to the artifact.

Do not include seed-42 development runs, failed jobs, smoke metrics, or
unavailable historical checkpoints in this table. The rejected-paper SEED-V
values may appear only in a short context sentence if needed, clearly labeled
as legacy reported values and never pooled with this table.

### Page 4: efficiency, interpretation, and conclusion

Use a paired-evidence and computational-overhead table in the body. Keep
absolute timing, memory, and parameter counts in the artifact or supplementary
material. The body table should have one clear job:

- trainable parameters increase by about 2.73M in every dataset;
- runtime increases by approximately 2.4×–3.0×;
- peak CUDA memory increases by approximately 1.9×–2.8×.

The manuscript caption should say: “Paired specialist-minus-full effects and
computational overhead. Performance deltas are mean ± sample SD over matched
seeds; time and memory are specialist-to-full ratios.”

The current measured cost summary is:

| Dataset | ΔBA | ΔwF1 | Δκ | +BA seeds | Time × | Memory × |
|---|---:|---:|---:|---:|---:|---:|
| FACED | −0.12 ± 1.24 | −0.17 ± 1.25 | −0.15 ± 1.33 | 1/3 | 2.41× | 1.86× |
| ISRUC | +1.27 ± 1.46 | +1.90 ± 1.66 | +2.40 ± 1.89 | 3/3 | 2.91× | 2.85× |
| SEED-V | +0.77 ± 0.74 | +0.93 ± 0.67 | +1.09 ± 1.06 | 2/3 | 2.55× | 2.38× |
| PhysioNet-MI | −1.27 ± 0.26 | −1.29 ± 0.24 | −1.71 ± 0.35 | 0/3 | 3.04× | 2.49× |

All specialist rows add approximately 2.73M trainable parameters. These
measurements should be presented as overhead under the locked A6000 execution
environment, not as hardware-independent complexity claims.

The efficiency paragraph should say that the proposed module adds capacity
and compute. It should not call the method lightweight or parameter-efficient.

Close with a concise interpretation:

> Cross-depth specialist augmentation is beneficial in some downstream regimes
> but neutral or harmful in others, while consistently increasing computational
> cost. Its use should therefore be treated as a task-dependent design choice
> rather than a uniformly stronger replacement for Full CBraMod fine-tuning.

Use the remaining body space for limitations and reproducibility: three seeds,
one locked protocol per dataset, one backbone, and no claim of universal
optimality. Keep detailed per-seed values, hashes, checkpoint paths, routing
diagnostics, and failed-run history in the released artifact.

## 5. Recommended figures and tables

### Figure 1: method schematic

One narrow architecture diagram. Highlight the added AttnRes and typed
specialist branch. Include a small note that the backbone is trainable.

### Table 1: paired performance

The four-row full-width table described above. Each method cell reports
BA / weighted-F1 / kappa as mean ± SD.

### Table 2: paired effects and computational overhead

Four rows, one per dataset, reporting paired deltas, positive-seed count,
runtime ratio, and memory ratio. Absolute measurements remain in the artifact.

The component-control extension has its own 12-cell manifest. If its results
are informative and the four-page layout permits, report only BA mean ± SD and
the two paired deltas in a compact table; keep exact per-seed deltas,
weighted-F1, macro-F1, and archived depth-routing or seed-42 diagnostics in
the artifact. The Full-vs-Augmented comparison remains the primary result.

## 6. Reviewer-feedback coverage

The outline addresses the main concerns with minimal new material:

| Concern | Evidence in this paper |
|---|---|
| Limited robustness across seeds | Three prespecified seeds per cell, mean ± SD |
| Baseline strength | Matched Full CBraMod control on every dataset |
| Dataset/task dependence | FACED, ISRUC, SEED-V, PhysioNet-MI |
| Split leakage/generalization | Explicit split provenance; PhysioNet-MI subject-disjoint manifest |
| Ambiguous method definition | Locked method recipe and explicit trainability statement |
| Reproducibility | Data/protocol/method/code/checkpoint manifests |
| Practical usefulness | Runtime, memory, and trainable-parameter measurements |
| Overclaiming | Report positive, neutral, and negative outcomes |

The paper should not attempt to answer every historical reviewer suggestion.
New backbones, broad hyperparameter sweeps, new router designs, and additional
datasets would expand the scope beyond what four pages can defend.

### Post-results analysis backlog

Do not launch additional training before the complete four-page draft is
reviewed. The highest-value remaining checks are analysis-only:

1. Run selected-checkpoint class-wise recall/confusion diagnostics for ISRUC
   and PhysioNet-MI. Add one or two sentences to the manuscript only if the
   pattern is clear; otherwise retain the result in the artifact.
2. Inspect ISRUC and PhysioNet-MI validation trajectories. Mention trajectory
   stability only if the method separation persists beyond the selected epoch.
3. If needed, run a full-set routing/utilization pass for artifact diagnostics.
   Last-validation-batch routing snapshots must not be used as global evidence.
4. The Full + AttnRes-only component-control extension is now prespecified as
   `full_attnres_only` in `paper_attnres_ablation_manifest_v1.csv`. It uses the
   same three seeds and dataset protocols as the main comparison, keeps the
   full backbone trainable, and disables specialists. It is analyzed
   separately from the frozen 24-cell main matrix.

## 7. TMLR separation firewall

The ICASSP paper has a different estimand and result package:

| TMLR | ICASSP |
|---|---|
| Native interaction-aligned adaptation | Task-dependent cross-depth specialist augmentation |
| CBraMod and LaBraM study | CBraMod only |
| Native-axis residual adapter | AttnRes + typed specialists |
| Structural alignment question | Cross-dataset task dependence and efficiency |

Do not reuse TMLR numerical results, checkpoints, figures, tables, prose, or
registries. Loader/preprocessing infrastructure may be reused when the
ICASSP row has independent execution provenance. Every table row in this
framework comes from the frozen ICASSP evidence view and retains its own
dataset, split, seed, code commit, protocol hash, method hash, checkpoint,
and metric provenance.

## 8. Frozen evidence source

The current four-dataset evidence was audited as 24/24 passing cells and 12
paired comparisons. The generated working artifacts are:

- `output/icassp2027_frozen_20260830/confirmatory_audit.json`
- `output/icassp2027_frozen_20260830/confirmatory_aggregate.json`
- `output/icassp2027_frozen_20260830/paper_main_table.md`
- `output/icassp2027_frozen_20260830/paper_paired_delta_table.md`
- `output/icassp2027_frozen_20260830/paper_efficiency_table.md`

The result set is paper-quality under the active audit: every row has a
complete summary, checkpoint, required metrics, accepted execution provenance,
and `paper_eligible=true`. The frozen view contains no smoke or seed-42 rows.
The Full rows for the earlier datasets are retained accepted legacy execution
artifacts, while the recent SEED-V specialist and PhysioNet-MI rows use the
hardened pinned execution snapshot. This provenance distinction is for the
artifact and internal audit; the manuscript should describe the common locked
protocol and retain exact commit identifiers in the released manifests rather
than implying that all 24 jobs ran from one identical checkout.

The analysis-contract correction is committed in `f7dd22c`. The frozen view
references the immutable training/execution worktrees rather than copying
large checkpoints. The paper must preserve those source paths and hashes in
the final artifact bundle.

## 9. Writing order

1. Insert the compact Table 1 values, then write the result paragraph without
   changing the interpretation.
2. Insert Table 2 and write the concise cost paragraph; keep absolute
   efficiency values in the artifact/supplement.
3. Draw Figure 1 from the locked method recipe only.
4. Write the introduction around the conditional-benefit question.
5. Compress protocol details into the method/experiment section.
6. Add the TMLR separation statement to the internal submission checklist,
   not as a prominent manuscript section unless required by policy.
7. Run a final claim audit: every quantitative statement must trace to a
   paper-eligible row in the frozen evidence view.

## 10. Submission-readiness checklist

- [ ] Title and abstract describe task-dependent effects and an accuracy–compute trade-off.
- [ ] Full-vs-Specialist comparison is stated before any auxiliary result.
- [ ] Four datasets and three seeds are clearly specified.
- [ ] Results are mean ± SD, with paired specialist-minus-full deltas.
- [ ] Positive, neutral, and negative dataset outcomes are all reported.
- [ ] Runtime, memory, and parameter overhead are visible.
- [ ] “Selective” is not used as a manuscript-facing frozen-backbone PEFT claim.
- [ ] Seed-42, smoke, failed, and historical rows are excluded from pooled
  confirmatory statistics.
- [ ] TMLR numbers, figures, prose, checkpoints, and registries are absent.
- [ ] Every paper number maps to a frozen, paper-eligible artifact row.
