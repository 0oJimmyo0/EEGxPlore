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
        print(f"[SEED-V] attnres_variant = {param.attnres_variant}")
        print(f"[SEED-V] attnres_gated = {param.attnres_gated}")
        print(f"[SEED-V] attnres_gate_init = {param.attnres_gate_init}")
        print(f"[SEED-V] attnres_start_layer = {param.attnres_start_layer}")
        if getattr(param, 'moe', False):
            depth_context_mode = getattr(param, 'moe_attnres_depth_context_mode', 'compact_shared')
            depth_summary_mode = getattr(param, 'moe_attnres_depth_summary_mode', 'auto')
            depth_probe_mlp = getattr(param, 'moe_attnres_depth_probe_mlp_for_router', False)
            if str(depth_context_mode) == 'block_shared_typed_proj':
                depth_summary_mode = 'block_typed_learned'
                depth_probe_mlp = False
            if str(depth_context_mode) == 'dual_query_block_typed_proj':
                depth_summary_mode = 'dual_query_block_typed_learned'
                depth_probe_mlp = False
            print(
                f"[SEED-V] MoE (typed_capacity_domain): top-{param.moe_num_layers} layers, "
                f"experts/bank={param.moe_num_experts}, "
                f"route_mode={getattr(param, 'moe_route_mode', 'typed_capacity_domain')}, "
                f"capacity_factor={getattr(param, 'moe_capacity_factor', 1.0)}, "
                f"psd_router={getattr(param, 'moe_use_psd_router_features', False)}, "
                f"attnres_depth_router={getattr(param, 'moe_use_attnres_depth_router_features', False)}, "
                f"attnres_depth_dim={getattr(param, 'moe_attnres_depth_router_dim', 26)}, "
                f"attnres_depth_context_mode={depth_context_mode}, "
                f"attnres_depth_block_count={getattr(param, 'moe_attnres_depth_block_count', 4)}, "
                f"attnres_depth_summary_mode={depth_summary_mode}, "
                f"attnres_depth_probe_mlp_for_router={depth_probe_mlp}, "
                f"attnres_depth_router_init={getattr(param, 'moe_attnres_depth_router_init', 'xavier')}, "
                f"attnres_depth_router_norm_gate={getattr(param, 'moe_attnres_depth_router_norm_gate', True)}, "
                f"attnres_depth_router_gate_init={getattr(param, 'moe_attnres_depth_router_gate_init', 0.075)}, "
                f"attnres_depth_router_norm_eps={getattr(param, 'moe_attnres_depth_router_norm_eps', 1e-6)}, "
                f"attnres_depth_block_separation_coef={getattr(param, 'moe_attnres_depth_block_separation_coef', 0.0)}, "
                f"attnres_depth_block_separation_target_js={getattr(param, 'moe_attnres_depth_block_separation_target_js', 0.03)}, "
                f"attnres_depth_grad_mode={getattr(param, 'moe_attnres_depth_summary_grad_mode', 'delayed_unfreeze')}, "
                f"attnres_depth_unfreeze_epoch={getattr(param, 'moe_attnres_depth_summary_unfreeze_epoch', 8)}, "
                f"uniform_warmup_epochs={getattr(param, 'moe_uniform_dispatch_warmup_epochs', 0)}, "
                f"shared_blend_warmup_epochs={getattr(param, 'moe_shared_blend_warmup_epochs', 0)}, "
                f"shared_blend_start={getattr(param, 'moe_shared_blend_start', 1.0)}, "
                f"shared_blend_end={getattr(param, 'moe_shared_blend_end', 0.0)}, "
                f"branch_mode={getattr(param, 'moe_specialist_branch_mode', 'both')}, "
                f"compact_router_mode={getattr(param, 'moe_router_compact_feature_mode', 'none')}, "
                f"compact_router_dim={getattr(param, 'moe_router_compact_feature_dim', 8)}, "
                f"expert_init_noise_std={getattr(param, 'moe_expert_init_noise_std', 0.0)}, "
                f"domain_bias={getattr(param, 'moe_domain_bias', False)}, "
                f"domain_emb_dim={getattr(param, 'moe_domain_emb_dim', 16)}, "
                f"moe_load_balance={getattr(param, 'moe_load_balance', 0.0)}, "
                f"moe_domain_bias_reg={getattr(param, 'moe_domain_bias_reg', 0.0)}"
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
                print("[SEED-V] Baseline mode: strict foundation load")
            elif getattr(param, 'moe', False):
                print("[SEED-V] MoE mode: partial load + dense FFN warm-start into experts")
            else:
                print(f"[SEED-V] AttnRes mode ({param.attnres_variant}): partial foundation load")
            print(f"[SEED-V] Backbone tensors marked pretrained: {len(self.pretrained_param_names)}")

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
                nn.LazyLinear(param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
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

    def forward(self, x, batch_meta=None):
        feats = self.backbone(x, batch_meta=batch_meta)
        out = self.classifier(feats)
        return out
