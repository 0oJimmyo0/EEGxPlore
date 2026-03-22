import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod, backbone_finetune_kwargs, load_foundation_into_backbone


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
            **backbone_finetune_kwargs(param),
        )
        print(f"[FACED] attnres_variant = {param.attnres_variant}")
        print(f"[FACED] attnres_gated = {param.attnres_gated}")
        print(f"[FACED] attnres_gate_init = {param.attnres_gate_init}")
        print(f"[FACED] attnres_start_layer = {param.attnres_start_layer}")
        if getattr(param, 'moe', False):
            init_all = not getattr(param, 'moe_expert_zero_only', False)
            mode = "shared_specialist+dense" if getattr(param, 'moe_shared_specialist', False) else "replace"
            print(
                f"[FACED] MoE ({mode}): top-{param.moe_num_layers} layers, "
                f"experts/specialists={param.moe_num_experts}, top_k={param.moe_top_k}, "
                f"router_mode={getattr(param, 'moe_router_mode', 'token')}, "
                f"router_arch={getattr(param, 'moe_router_arch', 'linear')}, "
                f"mlp_h={getattr(param, 'moe_router_mlp_hidden', 128)}, "
                f"psd_router={getattr(param, 'moe_use_psd_router_features', False)}, "
                f"warm_start_all_experts={init_all}"
            )
            if getattr(param, 'moe_shared_specialist', False):
                print(
                    "[FACED] specialist linear1: "
                    + (
                        "dense copy"
                        if not getattr(param, 'moe_specialist_rand_linear1', False)
                        else "kaiming"
                    )
                )
        self.pretrained_param_names = set()

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            ckpt = torch.load(param.foundation_dir, map_location=map_location)

            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            loaded_bb = load_foundation_into_backbone(self.backbone, param, ckpt)
            self.pretrained_param_names = {f'backbone.{k}' for k in loaded_bb}

            if param.attnres_variant == 'none' and not getattr(param, 'moe', False):
                print("[FACED] Baseline mode: strict foundation load")
            elif getattr(param, 'moe', False):
                print(f"[FACED] MoE mode: partial load + dense FFN warm-start into experts")
            else:
                print(
                    f"[FACED] AttnRes mode ({param.attnres_variant}): "
                    f"partial foundation load"
                )
            print(f"[FACED] Backbone tensors marked pretrained: {len(self.pretrained_param_names)}")

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
