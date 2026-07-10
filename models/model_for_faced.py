import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

try:
    from timm.models.layers import trunc_normal_
except ImportError:  # pragma: no cover - active LaBraM env should provide timm
    trunc_normal_ = None

from .cbramod import CBraMod, backbone_finetune_kwargs, load_foundation_into_backbone
from .labram_backbone import LaBraMBackbone, load_labram_foundation_into_backbone


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.backbone_name = str(getattr(param, 'backbone', 'cbramod')).strip().lower()
        self.labram_head_mode = str(getattr(param, 'labram_head_mode', 'external_pooled_linear')).strip().lower()
        if self.backbone_name == 'cbramod':
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
        elif self.backbone_name == 'labram':
            self.backbone = LaBraMBackbone(param)
        else:
            raise ValueError(f"Unsupported backbone for FACED: {self.backbone_name}")
        print(f"[FACED] backbone = {self.backbone_name}")
        print(f"[FACED] attnres_variant = {param.attnres_variant}")
        print(f"[FACED] attnres_gated = {param.attnres_gated}")
        print(f"[FACED] attnres_gate_init = {param.attnres_gate_init}")
        print(f"[FACED] attnres_start_layer = {param.attnres_start_layer}")
        if getattr(param, 'moe', False):
            print(
                f"[FACED] MoE (typed_capacity_domain): top-{param.moe_num_layers} layers, "
                f"experts/bank={param.moe_num_experts}, "
                f"route_mode={getattr(param, 'moe_route_mode', 'typed_capacity_domain')}, "
                f"capacity_factor={getattr(param, 'moe_capacity_factor', 1.0)}, "
                f"psd_router={getattr(param, 'moe_use_psd_router_features', False)}, "
                f"attnres_depth_router={getattr(param, 'moe_use_attnres_depth_router_features', False)}, "
                f"attnres_depth_dim={getattr(param, 'moe_attnres_depth_router_dim', 26)}, "
                f"attnres_depth_context_mode={getattr(param, 'moe_attnres_depth_context_mode', 'compact_shared')}, "
                f"attnres_depth_block_count={getattr(param, 'moe_attnres_depth_block_count', 4)}, "
                f"attnres_depth_summary_mode={getattr(param, 'moe_attnres_depth_summary_mode', 'auto')}, "
                f"attnres_depth_probe_mlp_for_router={getattr(param, 'moe_attnres_depth_probe_mlp_for_router', False)}, "
                f"attnres_depth_summary_grad_mode={getattr(param, 'moe_attnres_depth_summary_grad_mode', 'detached')}, "
                f"attnres_depth_summary_unfreeze_epoch={getattr(param, 'moe_attnres_depth_summary_unfreeze_epoch', 16)}, "
                f"router_temp={getattr(param, 'moe_router_temperature', 1.0)}, "
                f"router_entropy_coef={getattr(param, 'moe_router_entropy_coef', 0.0)}, "
                f"router_balance_kl_coef={getattr(param, 'moe_router_balance_kl_coef', 0.0)}, "
                f"domain_bias={getattr(param, 'moe_domain_bias', False)}, "
                f"domain_emb_dim={getattr(param, 'moe_domain_emb_dim', 16)}, "
                f"moe_load_balance={getattr(param, 'moe_load_balance', 0.0)}, "
                f"moe_domain_bias_reg={getattr(param, 'moe_domain_bias_reg', 0.0)}"
            )
        self.pretrained_param_names = set()

        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}') if torch.cuda.is_available() else torch.device('cpu')
            ckpt = torch.load(param.foundation_dir, map_location=map_location, weights_only=False)

            if self.backbone_name == 'cbramod':
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]
                loaded_bb = load_foundation_into_backbone(self.backbone, param, ckpt)
            else:
                loaded_bb = load_labram_foundation_into_backbone(self.backbone, ckpt)
            self.pretrained_param_names = {f'backbone.{k}' for k in loaded_bb}

            if self.backbone_name == 'labram':
                adapter_type = getattr(param, 'labram_adapter_type', 'cbramod_stack')
                if param.attnres_variant == 'none' and not getattr(param, 'moe', False):
                    print("[FACED][LaBraM] dense baseline mode: loaded LaBraM foundation weights")
                elif getattr(param, 'moe', False):
                    print(f"[FACED][LaBraM] selective mode: loaded LaBraM foundation + adapter_type={adapter_type}")
                else:
                    print(
                        f"[FACED][LaBraM] AttnRes mode ({param.attnres_variant}): "
                        f"loaded LaBraM foundation + adapter_type={adapter_type}"
                    )
            elif param.attnres_variant == 'none' and not getattr(param, 'moe', False):
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
        classifier_name = str(param.classifier)
        if self.backbone_name == 'labram':
            if self.labram_head_mode == 'native_head':
                classifier_name = 'native_head'
                print("[FACED][LaBraM] using native internal LaBraM head for dense parity.")
            elif classifier_name != 'labram_pooled_linear':
                print(
                    f"[FACED][LaBraM] remapping classifier {classifier_name!r} -> "
                    "'labram_pooled_linear' to match the original LaBraM pooled-head finetuning path"
                )
                classifier_name = 'labram_pooled_linear'

        if classifier_name == 'native_head':
            self.classifier = nn.Identity()
        elif classifier_name == 'labram_pooled_linear':
            self.classifier = nn.Linear(200, param.num_of_classes)
        elif classifier_name == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif classifier_name == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(param.num_of_classes),
            )
        elif classifier_name == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif classifier_name == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(10 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        if self.backbone_name == 'labram' and classifier_name == 'labram_pooled_linear':
            self._init_labram_pooled_classifier(param)

        all_param_names = {n for n, _ in self.named_parameters()}
        self.new_param_names = all_param_names - self.pretrained_param_names

        print(f'Loaded pretrained params: {len(self.pretrained_param_names)}')
        print(f'New params: {len(self.new_param_names)}')

    def forward(self, x, batch_meta=None):
        feats = self.backbone(x, batch_meta=batch_meta)
        out = self.classifier(feats)
        return out

    def _init_labram_pooled_classifier(self, param) -> None:
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError("LaBraM pooled FACED classifier is expected to be nn.Linear.")
        init_scale = float(getattr(param, 'labram_init_scale', 0.001))
        if init_scale < 0:
            with torch.no_grad():
                w = self.classifier.weight.detach().float()
                b = self.classifier.bias.detach().float()
                print(
                    "[FACED][LaBraM] pooled head init: preserving default nn.Linear initialization "
                    f"weight_mean={w.mean().item():.6g} weight_std={w.std(unbiased=False).item():.6g} "
                    f"bias_mean={b.mean().item():.6g} bias_std={b.std(unbiased=False).item():.6g}",
                    flush=True,
                )
            return
        if trunc_normal_ is None:
            raise ImportError(
                "timm is required to apply LaBraM-style pooled head initialization in EEGxPlore."
            )
        trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.constant_(self.classifier.bias, 0.0)
        self.classifier.weight.data.mul_(init_scale)
        self.classifier.bias.data.mul_(init_scale)
        with torch.no_grad():
            w = self.classifier.weight.detach().float()
            b = self.classifier.bias.detach().float()
            print(
                "[FACED][LaBraM] pooled head init: "
                f"init_scale={init_scale} "
                f"weight_mean={w.mean().item():.6g} weight_std={w.std(unbiased=False).item():.6g} "
                f"bias_mean={b.mean().item():.6g} bias_std={b.std(unbiased=False).item():.6g}",
                flush=True,
            )
