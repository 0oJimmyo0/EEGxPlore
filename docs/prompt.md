You are helping refine the EEGxPlore paper and codebase for the SEED-V branch:

Repository:
https://github.com/0oJimmyo0/EEGxPlore/tree/SEED-V

Current concern:
The current submitted paper proposes depth-aware selective adaptation for pretrained EEG foundation models using CBraMod as the main backbone. However, the current results show only modest gains over the base pretrained CBraMod/dense finetuning baseline, and some gains appear seed-sensitive. In particular, the current evidence suggests:

1. AttnRes/depth aggregation gives the clearest and most stable improvement.
2. Typed spatial/spectral-temporal specialists may add some benefit, but the effect is small.
3. Depth-aware routing, dual-query context, and compact EEG/PSD context may be second-order or seed-sensitive.
4. Some components may be redundant, overcomplicated, or not actually responsible for the performance improvement.
5. Before expanding to LaBraM, EEGPT, CSBrain, or other backbones, we need to revisit the CBraMod adaptation mechanism and identify the minimal robust method.

Please update the next-step research/refinement plan accordingly.

Primary objective:
Do not prioritize cross-backbone expansion yet. First, audit and simplify the current CBraMod adaptation approach. The next phase should answer:

“Which components of the current adaptation mechanism actually improve downstream transfer, and which components are redundant, unstable, or unnecessary?”

Please revise the plan around the following priorities.

P0: Trust and reproducibility fixes

1. Inspect the paper text, config defaults, and code implementation to resolve any mismatch between reported method and implementation.
2. Specifically check whether the official reported runs used soft dispatch or hard_capacity dispatch.
3. Make the paper, config files, training scripts, and method description consistent.
4. Ensure all main-paper numbers can be traced to logs, seeds, checkpoint identifiers, and generated summary tables.
5. Stop using headline/best values as the main evidentiary object. Main paper tables should prioritize seed-complete mean ± std and paired deltas versus dense finetuning. Headline/best checkpoint provenance should move to the appendix.

P1: CBraMod decomposition audit

Design a controlled ablation matrix on CBraMod before adding new backbones. Start with SEED-V and TUEV because they are the current matched-control datasets. Use the same seeds across all variants, ideally:

42, 3407, 2024, 11, 123

If compute is limited, use 3 seeds first, but keep the same seeds across all variants within a dataset.

Run the following families:

A. Baseline controls

* frozen backbone + classifier head
* dense finetuning
* top-k finetuning

B. Generic PEFT controls

* standard adapter
* LoRA
* generic residual adapter with parameter count matched to our specialist module if possible

C. Depth aggregation controls

* dense finetuning
* AttnRes-only
* AttnRes with frozen/lite adaptation if available

D. Capacity controls

* generic MoE or generic expert bank without typed spatial/spectral-temporal separation
* typed spatial/spectral-temporal bank with uniform routing
* typed bank with no learned routing

E. Routing-isolation controls

* random router
* uniform router
* shuffled depth-summary router
* content-only router
* depth-only router
* content + depth router
* frozen router
* learned router

F. Specialist-structure controls

* one shared specialist bank
* spatial-only specialists
* spectral-temporal-only specialists
* typed dual-bank specialists

G. Context controls

* no compact context
* compact EEG summary context
* PSD summary context
* dual-query context
* block-shared context

The key comparisons should be:

1. AttnRes-only vs dense finetuning.
2. Typed uniform specialists vs generic adapter/MoE.
3. Learned routing vs uniform/random/shuffled routing.
4. Content + depth router vs content-only and depth-only router.
5. Full current recipe vs the simplest strong recipe.
6. Compact context vs no context.
7. Dual-query vs block-shared routing.

P2: Decide the minimal official method

After the ablation audit, identify the simplest variant that gives stable gains across seeds.

Possible outcome:
The official method may become something simpler than the current full method, such as:

CBraMod + AttnRes + typed residual specialists + simple content/depth router, without compact EEG/PSD context and without dual-query routing.

Do not preserve components just because they are novel. Preserve only components that improve paired seed-complete performance or provide clear mechanism value.

Decision rules:

1. Remove compact EEG/PSD context from the main method if it does not improve seed-complete mean performance on at least two datasets.
2. Remove dual-query routing from the main method if it improves only selected headline runs but not paired seed means.
3. Reframe or remove depth-aware routing if learned routing does not beat uniform, random, shuffled-depth, and content-only routing.
4. Reframe typed specialists if a parameter-matched generic adapter or generic MoE performs the same.
5. Prefer the simplest stable method over the most complex method.

P3: Main CBraMod matched matrix

Once the minimal official method is selected, run the final matched CBraMod matrix on all five retained datasets:

* FACED
* SEED-V
* ISRUC
* TUEV
* PhysioNet-MI

For each dataset, run:

* frozen + head
* dense finetuning
* AttnRes-only
* generic PEFT baseline, preferably adapter or LoRA
* MoE-only or typed specialists without depth-aware routing
* minimal official method
* full current method only if still relevant

Report:

* balanced accuracy
* Cohen’s kappa
* weighted F1
* trainable parameters
* total parameters
* peak GPU memory if available
* wall-clock training time if available
* mean ± std across seeds
* paired delta versus dense finetuning
* per-seed dot plots

P4: Only after P1–P3, revisit cross-backbone expansion

Do not port the full current method to LaBraM/EEGPT/CSBrain before we know which CBraMod components actually matter.

After identifying the minimal official method, then test portability on one additional backbone first.

Preferred order:

1. LaBraM as the stable structured EEG backbone comparator.
2. CSBrain or REVE-base as the newer spatial-temporal/setup-flexible comparator if implementation is clean.
3. EEGPT as a stable fallback if newer model integration is too expensive.

But cross-backbone work should be framed as a later validation step, not the immediate next step.

P5: Update the next_step plan document

Please rewrite the current next_step plan so that the order becomes:

Phase A: CBraMod trust/reproducibility cleanup.
Phase B: CBraMod mechanism decomposition audit.
Phase C: Select minimal official method.
Phase D: Run final seed-complete CBraMod matrix.
Phase E: Add one additional backbone only after the minimal method is finalized.
Phase F: Add a second backbone only if time and compute remain.

The revised plan should explicitly state that the current priority is not “add more backbones,” but “determine whether the current adaptation approach is actually well-designed.”

Please also include a section titled “Components at risk of removal,” listing:

* compact EEG/PSD router context
* dual-query routing
* learned depth-aware router
* hard_capacity dispatch if it is not clearly better than soft dispatch
* two-bank specialist design if parameter-matched generic adapters perform similarly

Please include a section titled “Main scientific reframing,” with the following idea:

The paper should not overclaim that depth-aware routing is the entire contribution. A stronger and more honest story may be that downstream EEG foundation-model adaptation benefits from selective upper-layer modification, with the most robust gains coming from depth aggregation and structured lightweight specialist capacity. Routing/context should be presented as useful only if confirmed by paired seed-complete ablations.

Please produce:

1. a revised next_step plan,
2. a prioritized experiment table,
3. a decision tree for keeping/removing each component,
4. a compact list of code files likely needing changes,
5. a recommended order of runs for the next 1–2 weeks.
