# Cross-Backbone Execution Plan Pointer

Last updated: 2026-07-23

The canonical scientific plan is:

`/data/neurogroup/mingyangjiang/EEGxPlore/LaBraM/docs/cross_backbone_execution_plan.md`

This repository owns the **CBraMod** execution path. Do not launch LaBraM
paper experiments from this repository. The existing `--backbone labram` path
and `models/labram_backbone.py` are retained as historical substitution
engineering and must not be mixed into the revised CBraMod results.

Repository-specific audit anchors:

- backbone selection: `finetune_main.py:120-127`;
- historical LaBraM substitution flags: `finetune_main.py:145-290`;
- CBraMod native patch grid: `models/cbramod.py:358-382`;
- native depth/block summaries: `models/criss_cross_transformer.py:165-180` and
  `557-750`;
- AttnRes: `models/attn_res.py:17-31`;
- typed specialist routing: `models/moe.py:190-1415`;
- dispatch choices: `models/moe.py:14-17`;
- dataset loaders: `datasets/faced_dataset.py`, `isruc_dataset.py`,
  `seedv_dataset.py`, `tuev_dataset.py`, and `physio_dataset.py`.

All CBraMod plans, controls, and final runs must follow the canonical
interaction-aligned design rule and the contract in
`docs/experiment_contract.yaml`. This pointer is intentionally not a second
source of scientific truth.
