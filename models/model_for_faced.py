import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.backbone = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
            attnres_variant=param.attnres_variant,
        )

        # Track which params were actually restored from checkpoint
        self.pretrained_param_names = set()

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            ckpt = torch.load(param.foundation_dir, map_location=map_location)

            # In case a future checkpoint is saved as {"state_dict": ...}
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            # -----------------------------
            # Baseline path: original behavior
            # -----------------------------
            if param.attnres_variant == 'none':
                msg = self.backbone.load_state_dict(ckpt, strict=True)
                print(msg)

                # All backbone params are considered pretrained in the baseline
                self.pretrained_param_names = {
                    f'backbone.{k}' for k in self.backbone.state_dict().keys()
                }

                print("[FACED] Baseline mode: strict=True checkpoint loading")

            # -----------------------------
            # AttnRes path: partial warm-start
            # -----------------------------
            else:
                backbone_state = self.backbone.state_dict()
                loadable = {
                    k: v for k, v in ckpt.items()
                    if k in backbone_state and backbone_state[k].shape == v.shape
                }

                msg = self.backbone.load_state_dict(loadable, strict=False)
                print(msg)

                self.pretrained_param_names = {
                    f'backbone.{k}' for k in loadable.keys()
                }

                missing_backbone = sorted(
                    set(backbone_state.keys()) - set(loadable.keys())
                )

                print(
                    f"[FACED] AttnRes mode ({param.attnres_variant}): "
                    f"strict=False partial checkpoint loading"
                )
                print(f"[FACED] Loaded backbone tensors: {len(loadable)}")
                print(f"[FACED] Missing backbone tensors: {len(missing_backbone)}")
                if len(missing_backbone) > 0:
                    preview = missing_backbone[:20]
                    print(f"[FACED] First missing keys: {preview}")

        # IMPORTANT: do this AFTER loading backbone checkpoint
        # so baseline strict=True still matches the original CBraMod weights
        self.backbone.proj_out = nn.Identity()

        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 10 * 200, 10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        else:
            raise ValueError(f"Unknown classifier: {param.classifier}")

        all_param_names = {n for n, _ in self.named_parameters()}
        self.new_param_names = all_param_names - self.pretrained_param_names

        print(f'Loaded pretrained params: {len(self.pretrained_param_names)}')
        print(f'New params: {len(self.new_param_names)}')

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out