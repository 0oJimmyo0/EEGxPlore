# LaBraM Backbone Substitution Plan

## Goal
Replace the current `CBraMod` backbone in the SEED-V selective-adaptation pipeline with a LaBraM backbone, while preserving as much of the current adaptation machinery as possible:

- dataset loaders and split protocol
- training / evaluation loops
- pretrained-weight initialization flow
- classifier heads
- AttnRes / MoE-style selective adaptation blocks where still sensible

The key research question is:

> Do the current adaptation blocks remain effective when the base EEG foundation model is changed from CBraMod to LaBraM?

## Why this is scientifically interesting

The current branch couples two ingredients:

1. a pretrained EEG foundation model backbone (`CBraMod`)
2. selective adaptation on top of that backbone (AttnRes depth aggregation, typed MoE routing, block-context features)

Swapping the backbone lets us separate:

- gains caused by the adaptation strategy itself
- gains caused by CBraMod-specific representation structure

If the adaptation methods still help on LaBraM, that argues they are more general-purpose adaptation mechanisms rather than CBraMod-specific tricks.

## Current SEED-V stack

Current SEED-V execution path:

```text
submit_seedv_train.slurm
  -> finetune_main.py
    -> datasets/seedv_dataset.py
    -> models/model_for_seedv.py
      -> models/cbramod.py
      -> models/criss_cross_transformer.py
      -> models/attn_res.py
      -> models/moe.py
    -> finetune_trainer.py
    -> finetune_evaluator.py
```

Backbone-specific logic is concentrated in:

- `models/cbramod.py`
- `models/criss_cross_transformer.py`
- `models/model_for_seedv.py`

Training logic is largely backbone-agnostic as long as the model:

- accepts `x` and optional `batch_meta`
- returns logits for classification
- exposes trainable parameter names in a predictable way

## Backbone substitution strategy

### Stage 1: Minimal LaBraM integration

Objective:

- make SEED-V run end-to-end with LaBraM in the existing pipeline

Requirements:

- create `models/model_for_seedv_labram.py`
- create a thin LaBraM wrapper with:
  - pretrained checkpoint load
  - feature extraction forward
  - classifier head reuse
- keep `finetune_main.py`, dataset code, trainer code unchanged if possible

Deliverable:

- one working dense finetune baseline with LaBraM

### Stage 2: Interface normalization

Objective:

- define a shared backbone adapter interface so CBraMod and LaBraM are swappable

Recommended interface:

- `forward_features(x, batch_meta=None) -> features`
- `load_foundation_weights(...)`
- `feature_shape_spec`
- `pretrained_param_names`

This reduces future branching in:

- `model_for_seedv.py`
- optimizer parameter grouping
- checkpoint loading

### Stage 3: Adaptation transfer

Objective:

- port only the adaptation blocks that make sense on LaBraM

Three candidate levels:

1. **Head-only adaptation**
   - reuse current classifier heads
   - no AttnRes, no MoE
   - establishes LaBraM dense baseline

2. **Upper-layer selective adaptation**
   - adapt only top transformer layers
   - insert specialist FFN / routed blocks where LaBraM structure allows it

3. **Depth-context adaptation**
   - test whether the current block-summary / depth-router ideas transfer
   - may require a new definition of "depth summary" if LaBraM hidden-state structure differs from CBraMod

## Recommended experiment ladder

### Phase A: Backbone-only swap

Run:

1. `CBraMod dense baseline`
2. `LaBraM dense baseline`

Purpose:

- quantify the backbone substitution effect before adaptation

### Phase B: Simple adaptation transfer

Run:

1. `LaBraM + current classifier only`
2. `LaBraM + partial unfreeze`
3. `LaBraM + lightweight selective top-layer adaptation`

Purpose:

- establish whether adaptation helps at all on LaBraM before porting complex routing

### Phase C: Full selective adaptation port

Run:

1. `LaBraM + MoE without depth-context router features`
2. `LaBraM + MoE + adapted depth-context summaries`
3. compare against CBraMod selective adaptation

Purpose:

- isolate which adaptation pieces are backbone-agnostic

## Key implementation risks

### 1. Representation shape mismatch

CBraMod naturally uses `[B, C, S, D]`.

LaBraM may expose features in a different token layout. If so:

- classifier heads may need a new flattening / pooling rule
- AttnRes depth summaries may need new aggregation semantics

### 2. Hidden-state access

Current AttnRes / MoE context uses internal depth states from CBraMod layers.

If LaBraM does not expose analogous per-layer hidden states cleanly, then:

- depth-summary routing may need a simpler fallback
- initial LaBraM adaptation should avoid forcing exact CBraMod internals

### 3. Pretraining mismatch

CBraMod foundation weights are native to this codebase.

LaBraM checkpoints may assume:

- different channel ordering
- different normalization
- different patching / tokenization

We should validate:

- input contract
- sampling rate assumptions
- montage / channel map compatibility

### 4. Fair comparison

To claim adaptation transfer fairly:

- use the same SEED-V split protocol
- keep head capacity comparable
- keep optimizer family and training budget close

## Recommended first engineering milestone

Implement this first:

1. add a new `LaBraMSeedVModel` wrapper
2. support `--backbone {cbramod,labram}`
3. run dense LaBraM finetune on the exact same SEED-V LMDB split
4. verify checkpoint save, metrics, and runtime behavior

Only after that:

5. add selective adaptation variants on LaBraM

## Success criteria

### Engineering success

- SEED-V training completes on LaBraM
- checkpoint loads / saves work
- metrics are logged identically to CBraMod runs

### Research success

- at least one selective adaptation variant improves over LaBraM dense baseline
- improvement remains visible across more than one random seed
- comparison against CBraMod clarifies whether adaptation gains are backbone-specific or transferable

## Immediate next tasks

1. inspect LaBraM checkpoint format and expected input tensor contract
2. identify LaBraM feature tensor shape at the output used for classification
3. add a minimal SEED-V LaBraM wrapper with dense head only
4. run a smoke test identical to the CBraMod SEED-V smoke flow
5. only then design the adaptation-block port
