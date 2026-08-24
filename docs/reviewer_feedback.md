# Consolidated interpretation of the three reviews and meta-review

## Executive takeaway

Across the four assessments, there is a surprisingly consistent scientific message:

> The original paper identified a relevant downstream-adaptation problem and a reasonable CBraMod-aligned specialist design, but the evidence did not establish depth-aware routing as a stable or necessary contribution.

The main lessons worth carrying into the revised paper are:

1. **Keep the broad research problem:** EEG foundation-model adaptation should not be treated as an automatic choice between frozen probing and full fine-tuning.
2. **Preserve the structural insight:** adaptation should account for how an encoder organizes channel and temporal-patch interactions.
3. **Remove depth-aware routing from the paper’s central identity:** its contribution was small, unstable, and occasionally negative.
4. **Replace the single-backbone claim with a cross-backbone design principle:** CBraMod and LaBraM become two prospective instantiations of one interaction-aligned adaptation rule.
5. **Use matched multi-seed means as the primary evidence:** no best-run headline results.
6. **Add the missing simple controls:** LoRA, generic bottleneck adapters, frozen probing, upper-layer fine-tuning, full fine-tuning, and parameter-matched axis-blind adaptation.
7. **Report accuracy, stability, and efficiency together.**
8. **Predefine a fallback:** if the aligned adapter does not beat simpler controls, publish the work as a controlled empirical study rather than forcing a method-paper claim.

NeurIPS’s own criteria support this interpretation: originality can come from a novel combination, framing, or empirical insight rather than a completely new primitive, but the claims must be technically supported. Reviewers are also instructed to be fair, precise, constructive, and actionable. ([NeurIPS][1])

---

# Reviewer 1

## Core takeaway

Reviewer 1 believed that:

* the writing was excessively complicated;
* the evidence-tier terminology obscured the actual results;
* the main presentation emphasized the strongest selected results;
* baseline training and comparison fairness were insufficiently explained;
* the motivation for EEG-specific adaptation was weak;
* the Bommasani citation did not support the specific EEG-adaptation claim.

The review’s scientific core was:

> The paper used elaborate terminology and strong headline presentation to support a result that looked considerably weaker under conservative multi-seed interpretation.

## What made sense and should be incorporated

### 1. The introduction was too jargon-heavy

This criticism is valid.

The introduction introduced:

* depth-aware selective adaptation;
* AttnRes-derived depth-summary routing;
* typed spatial/spectral-temporal specialists;
* structured representation geometry;
* contextual payoff evidence;
* matched reproduced control evidence;
* boundary-case evidence;

before giving the reader a concrete operational picture. The contributions paragraph does indeed place several newly coined concepts together. 

### Incorporation

The revised paper should use only a few immediately defined terms:

* **native interaction structure**;
* **realized token shape**;
* **interaction eligibility**;
* **interaction-aligned adapter**;
* **generic or axis-blind adapter**.

The introduction should first show a simple example:

> CBraMod explicitly separates channel-wise and patch-wise interactions, while LaBraM organizes channel-patch tokens differently. We test whether adapters aligned with these interaction structures outperform applying the same generic PEFT module to both encoders.

Only then should the formal framework be introduced.

---

### 2. The evidence-tier language was cognitively expensive

Reviewer 1’s sarcastic treatment was excessive, but the underlying criticism was right.

“Contextual payoff,” “matched reproduced control,” and “boundary-case evidence” were intended to be transparent. In practice, they made the paper appear to be constructing categories around heterogeneous evidence rather than presenting one uniform experiment.

### Incorporation

Remove the three-tier terminology completely.

Use:

> Primary results use matched within-dataset comparisons under the same preprocessing, split, seed set, training budget, and checkpoint-selection rule. Previously published values are shown separately for context and are not treated as matched evidence.

This says the same thing more clearly.

---

### 3. Headline values overstated the strength of the result

Reviewer 1 was imprecise in saying you simply reported “the best seed tried,” but the broader concern was correct.

The paper reported validation-(\kappa)-selected headline runs separately from multi-seed means. On SEED-V, the headline Full-dual BA was 0.4162, while its three-seed mean was 0.4077. On TUEV, the headline full-model BA was 0.6798, while its mean was 0.6196 with substantial variability. 

### Incorporation

For the revised paper:

* main tables must contain mean ± standard deviation;
* use the same seed set for every primary method;
* show per-seed values in the appendix;
* never use a strongest seed or strongest run as the headline;
* checkpoint selection within each seed may still use a prespecified validation metric.

---

### 4. The Bommasani citation was not well targeted

This criticism was reasonable. A broad foundation-model paper is not the strongest source for a specific claim about EEG montage, sampling-rate, and task heterogeneity.

### Incorporation

Motivate the problem using:

* EEG-specific foundation-model evaluations;
* EEG transfer and channel/montage studies;
* EEG PEFT or adaptation papers;
* empirical evidence that adaptation behavior differs across backbones and tasks.

Bommasani may remain as general background, but not as primary support for the specific technical motivation.

---

### 5. The paper needed a more explicit argument against generic PEFT

Reviewer 1 asked why full fine-tuning or generic PEFT should behave differently in EEG.

That is a fair question, and the original paper asserted an answer more strongly than it demonstrated one.

### Incorporation

The new paper must empirically test:

[
\text{interaction-aligned adapter}
\quad\text{versus}\quad
\text{LoRA, generic adapter, upper-layer tuning, and axis-blind adapter}.
]

Do not assume that EEG structure alignment is useful. Make it the falsifiable hypothesis.

---

## What did not make sense and should not guide the revision

### 1. “No details on baseline training are provided”

This is factually incorrect.

The paper defined which modules were trainable for dense, AttnRes, MoE-only, full selective, frozen, top-(k), and adapter-style conditions. It also provided epochs, batch sizes, learning rates, weight decay, seed conventions, checkpoint-selection rules, timing, compute, and reproduction information.  

### Appropriate response

Do not redesign the new paper based on the belief that no details existed. Instead:

* move the most essential baseline definitions into the main paper;
* place one compact optimization table in the main paper;
* retain full details in the appendix and code artifact.

The problem was discoverability and presentation, not complete absence.

---

### 2. “The paper does not solve a relevant problem”

This was an unsupported categorical judgment.

The other two reviewers and the meta-review all recognized downstream EEG foundation-model adaptation as relevant. NeurIPS reviewers are expected to substantiate significance and novelty claims rather than give vague categorical conclusions. ([NeurIPS][1])

### Appropriate response

Ignore the categorical dismissal, but strengthen the relevance argument by showing:

* existing adaptation practice is inconsistent;
* generic PEFT may not respect encoder interactions;
* different backbones expose different structures;
* the proposed framework yields a measurable performance–efficiency or diagnostic benefit.

---

### 3. “Close to unintelligible”

The paper was overcomplicated, but not literally unintelligible. Sections 3.3–3.5 formally defined AttnRes aggregation, depth summaries, router context, and typed branches. 

### Appropriate response

Treat this as a warning about first-pass readability, not as evidence that the underlying method lacked definition.

---

### 4. “This sounds like an LLM writing up a non-result”

This was speculative, personal, and not a useful scientific argument. It falls short of NeurIPS guidance to be fair, precise, substantiated, and constructive. ([NeurIPS][1])

### Appropriate response

Ignore the accusation entirely. Retain only the technical issue underneath it:

> Did the main presentation make a weak or unstable effect appear stronger than the seed-complete evidence justified?

That issue is valid and should be corrected.

---

# Reviewer 2

## Core takeaway

Reviewer 2 considered the paper:

* relevant and practically motivated;
* technically reasonable;
* transparent and careful;
* generally clear;
* but empirically too fragile for acceptance.

The reviewer’s main conclusion was:

> The paper is technically solid, but the strongest evidence is not sufficiently matched or stable, the most novel component is weaker than the simpler components, and critical generic PEFT and efficiency controls are missing.

This is the most balanced of the three reviews.

## What made sense and should be incorporated

### 1. Multi-seed matched evaluation must cover the primary claims

This is essential.

FACED and ISRUC were presented as supportive single-run payoff evidence, while PhysioNet-MI was a focused single-run analysis.  

### Incorporation

The new paper should not inherit the exact five-dataset structure merely for continuity. Instead:

* select a smaller set of primary shared datasets;
* evaluate both CBraMod and LaBraM on those datasets;
* use the same primary seed policy for all core methods;
* reserve additional datasets for mechanistic or supplementary analyses.

A rigorous three-dataset, two-backbone matrix is preferable to five unevenly evaluated datasets.

---

### 2. LoRA, generic adapters, top-(k), and frozen controls are mandatory

This is probably the single most important experimental lesson.

The original paper documented these baseline concepts but did not provide matched results for frozen, top-(k), or Adapter/LoRA controls in the main evidence. 

### Incorporation

Every primary backbone-dataset family should include:

* frozen encoder + head;
* full fine-tuning;
* upper-(k)-layer fine-tuning;
* LoRA;
* generic bottleneck adapter;
* parameter-matched axis-blind adapter;
* interaction-aligned adapter;
* interaction-aligned + depth aggregation.

The aligned-versus-generic comparison tests whether structure matters.

The aligned-versus-upper-(k) comparison tests whether simply adapting late layers is sufficient.

---

### 3. Performance and efficiency must be evaluated jointly

This criticism is fully valid.

The original paper reported dataset-family compute envelopes, but it did not provide a clean component-level trade-off across methods. 

### Incorporation

Report by method:

* trainable parameter count;
* percentage of backbone trained;
* total parameters;
* peak training memory;
* training time per epoch;
* total GPU-hours;
* inference memory;
* inference latency;
* additional FLOPs or MACs;
* main performance metrics.

The paper should show a Pareto plot, not only an accuracy table.

---

### 4. SEED-V subject/session overlap requires exact reporting

The reviewer correctly distinguished split consistency from subject-independent generalization.

### Incorporation

The new paper should report:

* whether participants overlap;
* whether sessions overlap;
* whether neighboring segments from one recording cross splits;
* exact split unit;
* exact intended generalization claim.

Use the existing protocol as a reproducibility condition if needed, but add a subject-disjoint evaluation for cross-subject claims.

---

### 5. A second backbone is necessary

This concern directly supports the CBraMod + LaBraM direction.

### Incorporation

Use the second backbone not merely as an additional benchmark, but as a test of the design rule:

> Given a new backbone’s interaction structure and token shape, can we derive the appropriate adapter before seeing downstream results?

CBraMod and LaBraM should instantiate the same operator family under different structural specifications.

---

### 6. The novel router was not the stable source of improvement

Reviewer 2 accurately identified that AttnRes and specialist capacity accounted for most of the stable SEED-V change, while the later router/context refinement was small and seed-sensitive.

### Incorporation

Depth aggregation and depth routing should become:

* one secondary ablation;
* a potential negative result;
* not part of the title;
* not part of the primary adapter;
* not a central contribution.

---

## What should not be adopted literally

Reviewer 2 made very few incorrect arguments. Most concerns were valid.

### 1. Do not interpret “more backbones” as “backbone count is the contribution”

The reviewer asked for one additional backbone to test generality. That does not mean adding as many backbones as possible.

The paper should prioritize:

* two deep implementations;
* matched controls;
* multiple seeds;
* shared datasets;
* mechanistic aligned-versus-generic evidence.

A shallow third backbone is less valuable than completing the two-backbone causal matrix.

---

### 2. Do not feel obligated to preserve all original datasets

The reviewer asked for multi-seed FACED, ISRUC, and PhysioNet-MI because those datasets supported claims in the submitted paper.

For the new paper, you may redesign the evaluation. You do not need all five original datasets if a smaller shared matrix provides cleaner evidence.

---

### 3. The originality score of 2 is subjective, not a scientific fact

NeurIPS explicitly recognizes novel combinations, new insights, and empirical discoveries as valid originality. ([NeurIPS][2])

The new paper can have stronger originality if it presents:

* one prospective derivation rule;
* aligned and mismatched controls;
* cross-backbone validation;
* a mechanistic token-geometry result;
* a principled negative finding about depth complexity.

---

# Reviewer 3

## Core takeaway

Reviewer 3 agreed that:

* the adaptation problem is meaningful;
* the CBraMod-aligned specialist decomposition is reasonable;
* the paper is generally coherent and transparent;
* depth-aware routing is less supported than the paper’s original identity suggests;
* missing upper-layer and generic PEFT controls leave the main causal question unresolved.

This reviewer added two particularly useful analyses:

1. direct evidence about whether AttnRes actually selects meaningful earlier depths;
2. per-class analysis explaining the TUEV BA/weighted-F1 divergence.

## What made sense and should be incorporated

### 1. Expert usage did not establish useful depth conditioning

Showing that experts receive routing mass only establishes that the specialist path is active. It does not show that:

* depth summaries affect routing;
* earlier layers are meaningfully selected;
* depth use is task dependent;
* the depth signal improves performance.

### Incorporation

Because depth will no longer be central, do not launch a large depth-analysis project.

Keep a bounded secondary analysis:

* learned depth-weight distribution;
* earlier-layer mass;
* depth entropy;
* uniform-depth control;
* aligned adapter without depth;
* aligned adapter with depth;
* paired multi-seed difference.

The likely conclusion may be:

> Depth aggregation was operational but did not consistently improve the simpler aligned adapter.

---

### 2. TUEV needs per-class analysis

A large balanced-accuracy increase with little weighted-F1 change suggests class-specific trade-offs.

### Incorporation

For TUEV report:

* class support;
* per-class recall;
* per-class precision;
* per-class F1;
* macro F1;
* weighted F1;
* balanced accuracy;
* confusion matrices;
* paired error transitions between dense, LoRA, and aligned adaptation.

This should be part of the main validity analysis, not an optional appendix detail.

---

### 3. Upper-layer fine-tuning is a critical baseline

This is a highly efficient and scientifically targeted control.

### Incorporation

Compare the aligned adapter to:

* top-one block fine-tuning;
* top-(k) block fine-tuning under a predefined (k);
* full fine-tuning.

This determines whether the adapter provides value beyond merely exposing late layers to downstream gradients.

---

### 4. Narrow the channel/patch claim

The reviewer correctly said that CBraMod’s decomposition was motivated by CBraMod’s architecture and was not demonstrated as universally optimal for EEG.

### Incorporation

The revised claim should be:

> We propose a design rule that aligns adaptation with each encoder’s native interaction organization. Channel and patch operators are the resulting instantiations for CBraMod and LaBraM, not a universal decomposition for all EEG encoders.

Do not argue that frequency-band, temporal-scale, subject-specific, or other expert decompositions are inferior. They are outside the tested claim.

---

## What should not guide the revision

### 1. The numerical rating was harsher than the written assessment

The prose described the work as:

* adequately motivated;
* structurally coherent;
* transparent;
* well documented;
* technically reasonable.

A score of 2 was severe relative to that language, but rating calibration is subjective. Do not spend research effort trying to explain the score inconsistency.

---

### 2. Do not extensively compare every possible expert decomposition

The reviewer asked why frequency-band, temporal-scale, or context-specific experts were not used if the claim was general.

The clean answer is to narrow the claim. You do not need a combinatorial expert-search study.

The new paper should not claim that channel/patch structure is globally optimal. It should claim that the adapter is derived from the tested encoder’s native organization.

---

### 3. Process caveat: hidden prompt phrase overlap

The submitted PDF contained hidden machine-readable instructions requiring three particular phrases, and Reviewer 3 used those phrases or near-equivalents.  

This creates a review-process concern, but it should be kept separate from scientific planning.

The technical points from Reviewer 3 independently overlap with Reviewer 2 and the meta-review, so they remain useful even if the production process for the review is questionable.

---

# Meta-review

## Core takeaway

The meta-review gave the clearest high-level diagnosis:

> The paper’s best-supported result was modest benefit from simpler structured adaptation components; the evidence did not establish depth-aware/contextual routing as the stable cause of improvement.

It then identified the exact requirements for a more defensible study:

* matched multi-seed evaluation;
* generic PEFT and upper-layer controls;
* a second backbone;
* cost and parameter accounting;
* direct depth evidence;
* TUEV per-class analysis;
* SEED-V split clarification;
* empirical-study fallback if the complex method does not beat simpler controls.

This should carry the greatest weight in planning.

## What made sense and should be incorporated

### 1. The most novel component was not the best-supported component

This is the central lesson.

The new paper should not preserve depth routing merely because it was the most distinctive architectural component.

### Incorporation

Primary model:

* simple interaction-aligned adapter;
* no routing;
* no compact EEG/PSD context;
* no specialist MoE;
* no depth summary.

Secondary ablation:

* aligned adapter + depth aggregation.

---

### 2. Main evidence must use conservative multi-seed summaries

The meta-review correctly criticized the mismatch between headline selected runs and seed-complete means.

### Incorporation

The main table should report only:

* mean ± standard deviation;
* paired seed differences;
* number of favorable seeds;
* confidence intervals where appropriate.

Selected checkpoint values remain internal to each seed, not a separate headline category.

---

### 3. The missing baseline set is essential

This is fully correct and should become part of the paper’s central experimental design rather than an appendix add-on.

---

### 4. Second-backbone evidence materially improves the claim

This directly validates the move to CBraMod and LaBraM.

The new paper must show that the adapter is derived from one rule rather than separately handcrafted after observing each dataset.

---

### 5. Accuracy, stability, and cost require one unified analysis

This should become one of the main figures:

* x-axis: trainable parameters or GPU cost;
* y-axis: mean balanced accuracy or macro F1;
* error bars: seed variability;
* markers: frozen, dense, upper-(k), LoRA, generic adapter, aligned adapter.

---

### 6. Use an explicit method-paper versus empirical-study decision gate

This is one of the most useful recommendations.

### Method-paper outcome

Use this framing only if the aligned method:

* works across both backbones;
* improves at least two primary datasets;
* survives multi-seed averaging;
* beats or Pareto-dominates LoRA, generic adapters, and upper-(k) fine-tuning;
* beats a parameter-matched axis-blind control;
* demonstrates a measurable efficiency advantage.

### Empirical-study outcome

Use this framing if:

* generic PEFT matches the aligned method;
* upper-layer tuning explains most gains;
* only one backbone benefits;
* effects depend strongly on dataset geometry;
* depth complexity is consistently unnecessary.

Possible title:

> **When Does Encoder-Structure-Aware Adaptation Help EEG Foundation Models?**

NeurIPS recognizes empirical insights and negative results as potentially original, but a negative-results paper requires deeper analysis than merely observing that a method failed. ([NeurIPS][2])

---

## What in the meta-review should be corrected or interpreted carefully

### 1. “TUEV mean balanced accuracy does not improve over dense”

Taken literally, this is numerically incorrect.

The paper reports:

* Dense mean BA: 0.6040 ± 0.0231
* Full selective mean BA: 0.6196 ± 0.0391

So full selective has a numerically higher mean BA by approximately 1.56 percentage points. Its mean weighted F1 is lower:

* Dense: 0.7982 ± 0.0272
* Full selective: 0.7933 ± 0.0223. 

The defensible interpretation is:

> TUEV did not show a stable, metric-consistent improvement; mean BA increased, but variability was large and weighted F1 decreased.

Do not repeat the stronger claim that mean BA itself failed to improve numerically.

---

### 2. “AttnRes aggregation and typed specialists modestly help”

This is too broad if treated as a general conclusion.

On SEED-V, AttnRes and specialists modestly improved the mean results. On TUEV, AttnRes mean BA was lower than dense. 

Use the narrower conclusion:

> Simpler structured components showed modest benefit in some datasets, especially SEED-V, but did not produce uniform cross-dataset gains.

---

### 3. Do not interpret the meta-review as requiring the same original paper with more experiments

The revised research question is broader and cleaner. You do not need to preserve:

* every original dataset;
* the evidence tiers;
* the full specialist router;
* the original title or framing.

The review should guide the new experimental standards, not lock you into the rejected architecture.

---

# Cross-review consensus: what should definitely enter the new paper

| Consensus issue                                | Revised-paper action                                                  |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| Depth routing is weak and unstable             | Remove from primary model; retain bounded ablation                    |
| Headline runs overstated evidence              | Main results use multi-seed mean ± SD                                 |
| Generic controls are missing                   | Add LoRA, bottleneck, axis-blind, frozen, upper-(k), dense            |
| One backbone is insufficient                   | Use CBraMod and LaBraM                                                |
| CBraMod decomposition may be backbone-specific | Present one design rule with backbone-specific instantiations         |
| Evidence categories are too complicated        | Use one matched primary protocol                                      |
| Cost-benefit is unclear                        | Report parameters, memory, runtime, latency, FLOPs                    |
| TUEV metrics diverge                           | Add macro/per-class analysis and confusion matrices                   |
| SEED-V generalization is unclear               | Audit participant/session overlap and add subject-disjoint evaluation |
| Mechanism evidence is weak                     | Compare aligned versus parameter-matched generic/axis-blind adapters  |
| Claims may exceed evidence                     | Predefine method-paper and empirical-study outcomes                   |

---

# Arguments that should be discarded or de-emphasized

These should **not** drive the research plan:

1. The problem is irrelevant.
2. The entire paper was unintelligible.
3. No baseline-training information existed.
4. The work was merely “LLM-written.”
5. A large number of backbones is automatically better than two rigorous backbones.
6. Every original dataset must remain in the resubmission.
7. Every conceivable spatial, frequency, temporal, or subject-specific expert decomposition must be tested.
8. Depth aggregation requires an extensive new investigation even after it is removed from the central claim.
9. TUEV mean BA was numerically worse than dense; it was not, although the overall result was unstable and metric-dependent.
10. A paper is unoriginal merely because it combines existing components. NeurIPS explicitly allows originality through novel combinations, framing, and empirical insight. ([NeurIPS][2])

---

# How the reviews map onto the revised paper

## Proposed identity

> **Interaction-Aligned Adaptation for EEG Foundation Models**

## Research question

> Should internal PEFT operators be derived from an EEG encoder’s native interaction structure and realized downstream token shape, rather than applying one generic adapter uniformly across backbones?

## Central innovation

One prospective adapter-construction rule:

1. identify semantic representation axes;
2. identify native interaction sites;
3. record the realized ([C,S,D]) tensor shape;
4. exclude degenerate interaction axes;
5. instantiate a common low-rank residual operator along eligible axes;
6. allocate a fixed parameter budget;
7. freeze the mapping before final experiments.

CBraMod and LaBraM are two instantiations of this rule, not two unrelated adapters.

## Primary empirical test

[
\text{interaction-aligned}

>

\text{LoRA, generic bottleneck, axis-blind, and upper-layer tuning}
]

under:

* matched data and splits;
* matched seed sets;
* comparable parameter budgets;
* controlled optimization searches;
* common performance and efficiency reporting.

## Depth result

> Explicit depth aggregation is tested as an optional extension and is retained only if it provides stable incremental value. Current pilot evidence suggests that it does not.

---

# Mentor discussion brief

## What we learned from the reviews

The original paper’s strongest scientific idea was not the depth router. It was the intuition that adaptation should respect encoder structure.

The failure was primarily one of **claim–evidence alignment**:

* the paper identity emphasized depth routing;
* the stable evidence supported simpler adaptation;
* baseline comparisons did not isolate structural alignment;
* the evaluation was too heterogeneous;
* efficiency was not part of the central evidence.

## What we propose changing

1. Replace the depth-routing paper with an interaction-aligned PEFT paper.
2. Use CBraMod and LaBraM as two structurally distinct backbones.
3. Implement one common low-rank axis operator.
4. Derive its placement and eligibility prospectively.
5. Evaluate shared datasets under one matched protocol.
6. Add all simple PEFT and fine-tuning controls.
7. Make depth a secondary negative/neutral ablation.
8. Predefine whether final results justify a method paper or empirical study.

## Decisions to request from the mentor

### Scientific identity

* Is **interaction-aligned adaptation** the right central claim?
* Should the paper be positioned as a general ML method or an EEG-specific empirical study?
* Is the proposed backbone-to-adapter derivation rule sufficiently novel and falsifiable?

### Method design

* Should both LaBraM axes be enabled whenever (C>1) and (S>1), or should native architectural interaction sites impose a stricter rule?
* Should the common operator use low-rank attention, a bottleneck MLP, or another axis mixer?
* What parameter-matching tolerance is acceptable?

### Evaluation scope

* Which three datasets should be shared across both backbones?
* Is SEED-V best treated as a separate geometry/generalization analysis?
* Are five final seeds feasible?
* Is subject-disjoint SEED-V required for the primary paper?

### Publication strategy

* What results would justify a method-paper claim?
* What outcome should trigger the empirical-study fallback?
* What is the hard date for freezing the architecture and beginning confirmatory runs before the December 2026 application cycle?

## Recommended mentor-facing conclusion

> The reviews do not imply that the research problem was unimportant. They indicate that the original paper centered its identity on the least-supported mechanism. We propose preserving the meaningful insight—that downstream adaptation should account for encoder structure—while replacing depth-aware routing with a simpler and falsifiable cross-backbone framework. The revised work will test whether a prospectively derived interaction-aligned adapter provides a better performance–efficiency trade-off than generic PEFT and simple fine-tuning on CBraMod and LaBraM. If it does not, we will report the project as a controlled empirical study of when structure-aware adaptation helps rather than forcing a novel-method claim.

[1]: https://neurips.cc/Conferences/2026/MainTrackHandbook?utm_source=chatgpt.com "Main Track Handbook 2026"
[2]: https://neurips.cc/Conferences/2026/ReviewerGuidelines?utm_source=chatgpt.com "2026 Reviewer Guidelines"

---

# Current revised-plan status and TMLR readiness

This section supersedes the earlier “mentor discussion brief” as the current
execution record. The four reviewer interpretations above are retained: the
reviewers were correct that the legacy paper over-centered depth routing,
under-isolated the causal comparison against simple PEFT and upper-layer
training, and mixed single-run evidence with seed-complete evidence. Their
unsupported categorical statements about irrelevance, unintelligibility,
absence of all baseline details, or “LLM-written” work remain rejected and do
not guide the revised study.

## Current paper identity

The working paper identity is:

> **Interaction-Aligned Adaptation for EEG Foundation Models**

The central claim is deliberately conditional:

> A lightweight residual adapter can be derived from a backbone’s native,
> non-degenerate interaction axes. Its usefulness depends on the backbone,
> realized token geometry, dataset, and whether the pretrained backbone is
> frozen or fully trainable.

This is not a claim that channel/patch adapters universally outperform dense
fine-tuning. CBraMod must be run only in `EEGxPlore/EEGxPlore`; LaBraM must be
run only in `LaBraM`. The repositories must never be wired together for a
paper result. `LaBraM-depth` remains historical and is not a third active
backbone.

## What the current evidence already says

### LaBraM SEED-V

- The realized LaBraM geometry is `[B,62,1,D]`; channel interaction is
  eligible, while temporal patch attention is degenerate.
- Frozen channel adaptation is the correct structure-aware SEED-V result;
  frozen patch is retained only as a singleton-axis capacity control.
- The existing within-subject results show that freezing can reduce the
  early-overfitting failure mode, but a subject-disjoint evaluation is still
  required before making a cross-subject generalization claim.

### LaBraM ISRUC

The verified ISRUC contract is six bipolar channels, 30 temporal patches per
epoch, sequence length 20, batch size 16, the CBraMod-matched preprocessing,
and validation-kappa checkpoint selection.

Completed development evidence:

| Cell | Development status | Primary test BA summary |
| --- | --- | ---: |
| Dense full fine-tuning | Seed-42 test anchor complete | `0.7997` |
| Rank-32 channel, patch, channel+patch, alpha `0.5` | Seeds `42,1024,3407` complete | `0.7924`, `0.7911`, `0.7917` mean |
| Frozen generic token adapter | Seeds `42,1024,3407` complete | `0.6636 ± 0.0023` |
| Upper-2 fine-tuning | Seeds `42,1024,3407` complete | `0.7939 ± 0.0053` |
| LoRA qkv, rank 8 | Seeds 42 and 1024 retry complete; 3407 pending | pending final packet |

The native low-rank adapters are active and connected, but their validation
peaks occur early and their test means do not exceed the dense anchor. Upper-2
fine-tuning nearly matches dense performance with a smaller train/validation
gap. Generic frozen adaptation underfits in its current configuration. The
responsible conclusion is therefore:

> On ISRUC, native residual adaptation is clearly useful in the frozen-backbone
> regime, but the current fully trainable low-rank adapter does not improve over
> dense fine-tuning. This is a conditional result, not evidence that the
> design rule is universally ineffective.

The earlier frozen ISRUC gain used a full-width frozen patch adapter. It must
not be silently presented as proof that the final rank-32 common primitive has
already won a matched frozen comparison.

Current jobs:

- `12760330`: repaired LoRA seed-3407 completion, pending.
- `12760559`: one bounded channel run with backbone LR scale `0.1`, adaptor
  LR scale `1.0`, and alpha LR scale `0.5`, pending. This tests whether
  reducing backbone drift can expose a useful adaptor effect.

## Remaining experiments before the paper-grade matrix is complete

### 1. Finish and close ISRUC LaBraM

Required:

1. Complete LoRA seed 3407 and record the complete three-seed result.
2. Inspect `12760559` as a single targeted early-overfitting test. Promote it
   to multiseed only if validation trajectory, train/validation gap, and test
   behavior all improve coherently.
3. Add a parameter-matched axis-blind residual control. The current generic
   token adapter is not parameter-matched to the rank-32 aligned branch and
   is therefore a useful negative control, not the final causal test.
4. Run the frozen version of the final rank-32 aligned primitive against
   frozen dense and frozen generic controls, using the same trainable-parameter
   budget and development seeds.
5. Stop ISRUC tuning after these gates. Do not continue broad LR, depth, or
   adapter-variant searches if the aligned method remains below dense and
   upper-layer controls.

### 2. Complete LaBraM across the planned datasets

- **FACED:** the existing FACED work is complete as an internal experiment
  block. Before using it as primary evidence, verify that the final registry
  contains the same protocol hash, seed policy, dense/frozen/upper-k/LoRA/
  generic/axis-blind/aligned cells, and efficiency fields. Rerun only missing
  cells.
- **TUEV:** still requires preprocessing and class-support audit, dense
  baseline, the complete control ladder, and per-class recall/precision/F1,
  macro metrics, weighted metrics, and confusion matrices.
- **SEED-V:** retain the within-subject reproducibility result, but add the
  subject-disjoint/grouped evaluation required for any cross-subject claim.
- **PhysioNet-MI:** optional; do not start it until the three shared primary
  datasets are complete and its subject-split contract is verified.

### 3. Complete the independent CBraMod path

In `EEGxPlore/EEGxPlore`, independently implement and verify the same
low-rank residual family and the same control definitions. Do not import the
LaBraM implementation or substitute LaBraM into CBraMod. The required CBraMod
matrix is dense, frozen, upper-k, LoRA, generic, parameter-matched axis-blind,
aligned, and one bounded depth extension on the selected shared datasets.

### 4. Add the reviewer-required analysis layer

Every primary cell must report:

- mean ± standard deviation across the declared seeds, per-seed values, and
  paired differences;
- balanced accuracy, macro F1, accuracy, kappa, and weighted F1;
- trainable and total parameters, peak memory, wall-clock time, GPU-hours,
  throughput, inference latency, and FLOPs/MACs where practical;
- realized `[C,S,D]`, eligibility mask, alpha, residual ratio, gradients,
  update norms, and adapter-on/off effects;
- TUEV per-class results and confusion matrices;
- exact split units and overlap audits, especially for SEED-V.

## TMLR submission gate

TMLR does not require a universal performance gain or a new state-of-the-art.
Its official acceptance questions are whether the claims are supported by
accurate, convincing, clear evidence and whether the findings would interest
some part of its audience. A systematic study of method strengths, weaknesses,
robustness, or generalization can satisfy that standard when it produces
actionable insight. See the [TMLR acceptance criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
and [editorial policies](https://jmlr.org/tmlr/editorial-policies.html).

There is no numeric BA or accuracy threshold. For this project, the minimum
defensible TMLR submission should satisfy all of the following:

1. At least two backbones are complete, with CBraMod and LaBraM kept in their
   separate repositories and derived from the same prospective design rule.
2. At least two shared primary datasets have the complete matched matrix;
   three shared datasets (FACED, ISRUC, TUEV) are preferred.
3. Each primary cell includes dense, frozen, upper-k, LoRA, generic,
   parameter-matched axis-blind, and aligned controls under the same split,
   budget, checkpoint rule, and declared seed set.
4. Final claims use seed-complete means and uncertainty, not the best seed.
   This project uses the fixed three-seed packet `{42, 1024, 3407}` for both
   multiseed results and the locked final test block; no five-seed block is
   planned.
5. The paper reports the performance–efficiency trade-off and explains both
   positive and negative results, including ISRUC’s frozen-versus-trainable
   regime difference and SEED-V’s geometry boundary condition.
6. The code, protocol manifests, checkpoints, hashes, trainability summaries,
   and analysis scripts are reproducible, with final runs made from a committed
   and clean code state.

### Method-paper threshold

Use the stronger method-paper framing only if the aligned adapter has a
positive and stable mean effect on both backbones, support on at least two
primary datasets, a matched-budget advantage over axis-blind adaptation, and
an efficiency or parameter-efficiency advantage over LoRA and upper-k tuning.

### Controlled-study threshold

If the aligned adapter does not beat dense or upper-k fine-tuning, the work can
still be submitted to TMLR as a controlled empirical study if the matrix is
complete and the paper establishes actionable findings such as:

- native-axis eligibility predicts when an adapter can have a meaningful
  mechanism;
- frozen and fully trainable adaptation have different failure modes;
- generic PEFT, LoRA, upper-layer tuning, and aligned adaptation occupy
  different performance–efficiency regimes;
- depth complexity is not consistently necessary;
- realized token geometry and split protocol materially change the conclusion.

Under that outcome, the title and claims should change to something like:

> **When Does Encoder-Structure-Aware Adaptation Help EEG Foundation Models? A
> Controlled Study Across CBraMod and LaBraM**

The current project is not yet at either submission gate: ISRUC is close, but
LoRA completion, parameter matching, the CBraMod matrix, efficiency reporting,
and the remaining dataset analyses are still required. The correct near-term
action is to finish the bounded ISRUC closeout, then move dataset-by-dataset
and backbone-by-backbone without reopening broad adapter tuning.
