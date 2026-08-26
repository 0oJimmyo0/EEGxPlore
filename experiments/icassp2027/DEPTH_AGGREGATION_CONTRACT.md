# ICASSP 2027 Depth Aggregation Contract

Status: implementation profile
Branch: `icassp2027-routing`

## Scientific identity

Working title: **Selective Depth Aggregation for Subject-Disjoint EEG Foundation Model Transfer**

Central question:

> Can lightweight learned aggregation across pretrained transformer depth provide useful adaptation for unseen-subject EEG transfer without directly updating the pretrained backbone?

## Locked scope

- Backbone: CBraMod only.
- Datasets: SEED-V, FACED, ISRUC, and PhysioNet-MI.
- Split regime: the frozen subject-disjoint manifests already audited in this repository.
- Checkpoint selection: validation Cohen's kappa only.
- Final seeds: `42`, `1024`, and `3407`.
- No MoE, routing, specialist banks, PSD/context features, domain metadata, LaBraM, or TUEV.

## Fixed method matrix

| Method | Configuration | Role |
|---|---|---|
| Frozen | `attnres_variant=none`, classifier only | Representation reference |
| DepthAgg | `pre_attn`, start layer `8`, ungated | Proposed cross-depth aggregation |
| Upper4 | dense original parameters in layers 8--11 | Selective plasticity reference |
| Full | all original backbone parameters | Maximum-plasticity reference |

DepthAgg trains only:

```text
backbone.encoder.layers.{8,9,10,11}.pre_attn_res.{norm.weight,query}
classifier.*
```

The four active `FullAttnRes` modules contain exactly 1,600 trainable
depth-specific parameters for CBraMod (`4 x (200 + 200)`), excluding the
task head. The current model may instantiate additional frozen AttnRes modules
below layer 8; those are not trainable DepthAgg parameters.

## Optimizer contract

- Optimizer: AdamW.
- Base learning rate: `1e-4`.
- Depth multiplier: `1.0` (`1e-4` resolved depth LR).
- Classifier multiplier: `3.5` (`3.5e-4` resolved classifier LR).
- Upper4/Full backbone multiplier: `0.5` (`5e-5` resolved backbone LR).
- Weight decay: `5e-2`.
- Label smoothing: `0.1`.
- Class weighting: disabled.
- Warmup: disabled initially.

All resolved optimizer groups and trainable parameter counts must be recorded
in the run summary.

The JSON provenance record must additionally include the foundation checkpoint
path and SHA-256, the complete trainable parameter-name list, and an initial
snapshot of every resolved optimizer group before scheduler updates. The CSV
summary carries the foundation hash, AttnRes start layer, and JSON-encoded
versions of the same trainable-name and optimizer-group fields.

## Evidence firewall

- The routing outputs and routing contract are historical development artifacts.
- No routing result, TMLR result, checkpoint, registry, figure, or manuscript
  text may enter the DepthAgg evidence package.
- All DepthAgg results use `output/icassp2027_depth/` or a descendant run root.
- Every reported result is regenerated under the locked ICASSP manifests.
