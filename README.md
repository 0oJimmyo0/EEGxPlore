<div align="center">

# CBraMod


_A Criss-Cross Brain Foundation Model for EEG Decoding_


[![Paper](https://img.shields.io/badge/arXiv-2412.07236-red)](https://arxiv.org/abs/2412.07236)
[![Paper](https://img.shields.io/badge/Paper-ICLR-008B8B)](https://openreview.net/forum?id=NPNUHgHF2w)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/weighting666/CBraMod)
![GitHub Repo stars](https://img.shields.io/github/stars/wjq-learning/CBraMod)

</div>


<div align="center">
<img src="figure/CBraMod_logo.png" style="width: 15%;" />
</div>


<p align="center">
    🔍&nbsp;<a href="#-about">About</a>
    | 🔨&nbsp;<a href="#-setup">Setup</a>
    | 🚢&nbsp;<a href="#-pretrain">Pretrain</a>
    | ⛵&nbsp;<a href="#-finetune">Finetune</a>
    | 🧭&nbsp;<a href="#-current-workflow-eegxplore-branch">Workflow</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>
🔥 NEWS: Thanks to over 100 stars! We've further refined the code for improved stability. Appreciate your patience as we refine the implementation — ongoing EEG research continues to shape the development of a standardized pipeline.

🔥 NEWS: The paper "_CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding_" has been accepted by ICLR 2025!

## 🔍 About
We propose **CBraMod**, a novel EEG foundation model, for EEG decoding on various clinical and BCI application.
The preprint version of our paper is available at [arXiv](https://arxiv.org/abs/2412.07236). 
The camera-ready version of the paper will be available at [OpenReview](https://openreview.net/forum?id=NPNUHgHF2w).
<div align="center">
<img src="figure/model.png" style="width:100%;" />
</div>



## 🔨 Setup
Install [Python](https://www.python.org/downloads/).

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:
```commandline
pip install -r requirements.txt
``` 


## 🚢 Pretrain
You can pretrain CBraMod on our pretraining dataset or your custom pretraining dataset using the following code:
```commandline
python pretrain_main.py
```
We have released a pretrained checkpoint on [Hugginface🤗](https://huggingface.co/weighting666/CBraMod).

## ⛵ Finetune
You can finetune CBraMod on our selected downstream datasets using the following code:
```commandline
python finetune_main.py
```

## 🧭 Current Workflow (EEGxPlore Branch)

This branch is currently centered on FACED + SEED-V finetuning with selective adaptation (AttnRes + typed MoE).

### Script Surface (Reviewed)

- `scripts/run_seedv.sh`: local single-node SEED-V run (defaults to strict A1 block-summary setup).
- `scripts/SEED-V/submit_seedv_train.slurm`: cluster launcher for SEED-V experiments.
- `scripts/SEED-V/audit_seedv_lmdb_split.py`: checks split schema and subject overlap.
- `scripts/SEED-V/build_seedv_subject_disjoint_manifest.py`: wrapper for canonical manifest builder.
- `scripts/SEED-V/verify_block_context_diagnostics.py`: validates that block-depth diagnostics are present and non-trivial.
- `scripts/run_faced.sh` and `scripts/FACED/*`: FACED launch/analysis utilities.

### Current A1 Definition

`A1` now means the SEED-V block-summary depth-context ablation with a strict one-change contract:

1. `moe_attnres_depth_context_mode=block_shared_typed_proj`
2. fixed `moe_attnres_depth_block_count=4`
3. fixed `moe_attnres_depth_router_dim=15`
4. compact-summary-only knobs are not mixed into this run
5. CBraMod SEED-V cohort: `/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb` with LMDB `__keys__` split

`scripts/SEED-V/submit_seedv_train.slurm` enforces this by default with `A1_STRICT_BLOCK_ABLATION=1`.

### Start A New SEED-V Slurm Run

```bash
cd scripts/SEED-V
sbatch --export=ALL,SEEDV_PROTOCOL=cbramod_benchmark,RUN_NAME=seedv_a1_block_$(date +%Y%m%d_%H%M%S) submit_seedv_train.slurm
```

Useful overrides:

- `MODEL_DIR=/path/to/output_dir`
- `DATASET_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb`
- `EPOCHS=40`

For strict CBraMod-comparable cohort, keep `SEEDV_SPLIT_MANIFEST` unset.

Example:

```bash
cd scripts/SEED-V
sbatch --export=ALL,SEEDV_PROTOCOL=cbramod_benchmark,DATASET_DIR=/gpfs/radev/pi/xu_hua/shared/datasets/SEED-V/processed_lmdb,MODEL_DIR=/scratch/$USER/seedv_a1_run01,RUN_NAME=seedv_a1_run01,EPOCHS=40 submit_seedv_train.slurm
```

### Progress Snapshot

- Done: SEED-V launcher passes depth context mode and block count end-to-end.
- Done: strict A1 launcher mode now keeps the experiment surface clean.
- Done: diagnostics export includes block stats, router context norms, and routing stats.
- In progress: replacing fixed block mean pooling with a richer learned block selector/readout.

### Verify Block Context Is Actually Used

After training, verify exported diagnostics:

```bash
python scripts/SEED-V/verify_block_context_diagnostics.py --run_dir <MODEL_DIR>
```

Compare two runs:

```bash
python scripts/SEED-V/verify_block_context_diagnostics.py --run_dir <MODEL_DIR_RUN1> --run_dir <MODEL_DIR_RUN2>
```


## 🚀 Quick Start
You can fine-tune the pretrained CBraMod on your custom downstream dataset using the following example code:
```python
import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CBraMod().to(device)
model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
model.proj_out = nn.Identity()
classifier = nn.Sequential(
  Rearrange('b c s p -> b (c s p)'),
  nn.Linear(22*4*200, 4*200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(4 * 200, 200),
  nn.ELU(),
  nn.Dropout(0.1),
  nn.Linear(200, 4),
).to(device)

# mock_eeg.shape = (batch_size, num_of_channels, time_segments, points_per_patch)
mock_eeg = torch.randn((8, 22, 4, 200)).to(device)

# logits.shape = (batch_size, num_of_classes)
logits = classifier(model(mock_eeg))
```



## 🔗 Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@inproceedings{wang2025cbramod,
    title={{CB}raMod: A Criss-Cross Brain Foundation Model for {EEG} Decoding},
    author={Jiquan Wang and Sha Zhao and Zhiling Luo and Yangxuan Zhou and Haiteng Jiang and Shijian Li and Tao Li and Gang Pan},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=NPNUHgHF2w}
}
```

## ⭐ Star History
<div align="center">
    <a href="https://star-history.com/#wjq-learning/CBraMod&Date">
        <img src="https://api.star-history.com/svg?repos=wjq-learning/CBraMod&type=Date" style="width: 80%;" />
    </a>
</div>