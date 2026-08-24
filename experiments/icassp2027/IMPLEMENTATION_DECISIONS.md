# ICASSP 2027 Implementation Decisions

This document records which parts of the repository analysis are adopted for
the ICASSP branch. It is intentionally narrower than the historical rejected
paper and does not alter the legacy `typed_capacity_domain` implementation.

## Adopted

- Add a new `typed_conditional` route mode for the ICASSP study.
- Use one `TypedConditionalMoEFFN` parameter schema for both conditions:
  `router_policy=static` and `router_policy=sample`.
- Use the same router architecture and learned constant in both conditions.
  Static uses `softmax(R(c))`; Routed uses `softmax(R(u+c))`, where `u` is
  the mean of the current normalized pre-FFN sample representation over
  channel and patch axes.
- Keep the shared FFN, spatial specialists, and spectral-temporal specialists
  in both conditions. Use soft dispatch only.
- Adapt the upper four CBraMod transformer blocks. The ICASSP profile rejects
  AttnRes, PSD features, depth/context features, domain metadata, compact EEG
  summaries, router jitter, warmups, and router-specific regularizers.
- Freeze the pretrained shared FFN for Static/Routed while training the
  specialists, routers, constants, and task head. This makes the comparison
  estimate routing/capacity adaptation beyond the same frozen foundation;
  trainability and parameter counts must be written to the run registry.
- Add exact-isomorphism tests: identical state-dict keys/shapes, identical
  initialization under a matched seed, and identical trainable parameter
  counts for Static/Routed.
- Report macro-F1 in addition to the existing balanced accuracy, weighted-F1,
  and kappa. Keep timing and memory as registry diagnostics, not paper claims.
- Add SEED-V-only per-sample routing export for mechanism analysis.
- Fix SEED-V external-manifest shape validation so expected shape is derived
  from the actual serialized examples rather than an unrelated legacy
  manifest.

## Already resolved and retained

- Fresh subject-disjoint manifests, key audits, hashes, and metadata tables
  for SEED-V, FACED, ISRUC, and PhysioNet-MI.
- PhysioNet-MI subject provenance audit.
- ISRUC container-level manifest mapping. This is more precise than a simple
  subject-list override because an ISRUC subject contains multiple sequence
  containers.
- Frozen CBraMod forward probes across all four datasets.

## Rejected or deferred

- Do not modify or repurpose `typed_capacity_domain`; it remains legacy code.
- Do not use TMLR numerical results, checkpoints, registries, native-axis
  adapters, LaBraM, TUEV, or AttnRes in an ICASSP run.
- Do not build routing exports for every dataset. Only SEED-V is needed for
  the planned mechanism figure.
- Do not make runtime or memory a primary scientific result in a four-page
  paper; record them for reproducibility and sanity checks.
- Upper-4, LoRA, and bottleneck baselines remain required follow-up code, but
  are implemented after the Static/Routed path passes its contract tests.

## Implementation firewall

An ICASSP run must fail fast unless all of the following hold:

```text
backbone = CBraMod
moe_route_mode = typed_conditional
moe_num_layers = 4
attnres_variant = none
moe_use_psd_router_features = false
moe_use_attnres_depth_router_features = false
moe_domain_bias = false
moe_router_compact_feature_mode = none
moe_router_dispatch_mode = soft
moe_router_entropy_coef = 0
moe_router_balance_kl_coef = 0
moe_router_z_loss_coef = 0
moe_router_jitter_std = 0
moe_router_jitter_final_std = 0
moe_router_soft_warmup_epochs = 0
moe_uniform_dispatch_warmup_epochs = 0
moe_shared_blend_warmup_epochs = 0
```

