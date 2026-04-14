import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(f"[ISRUC] attnres_variant = {param.attnres_variant}")
        print(f"[ISRUC] attnres_gated = {param.attnres_gated}")
        print(f"[ISRUC] attnres_gate_init = {param.attnres_gate_init}")
        print(f"[ISRUC] attnres_start_layer = {param.attnres_start_layer}")
        if getattr(param, 'moe', False):
            print(
                f"[ISRUC] MoE (typed_capacity_domain): top-{param.moe_num_layers} layers, "
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
            map_location = torch.device(f'cuda:{param.cuda}')
            ckpt = torch.load(param.foundation_dir, map_location=map_location)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']

            loaded_bb = load_foundation_into_backbone(self.backbone, param, ckpt)
            self.pretrained_param_names = {f'backbone.{k}' for k in loaded_bb}
            if param.attnres_variant == 'none' and not getattr(param, 'moe', False):
                print('[ISRUC] Baseline mode: strict foundation load')
            elif getattr(param, 'moe', False):
                print('[ISRUC] MoE mode: partial load + dense FFN warm-start into experts')
            else:
                print(f"[ISRUC] AttnRes mode ({param.attnres_variant}): partial foundation load")
            print(f"[ISRUC] Backbone tensors marked pretrained: {len(self.pretrained_param_names)}")

        self.backbone.proj_out = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(6*30*200, 512),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, param.num_of_classes)

        all_param_names = {n for n, _ in self.named_parameters()}
        self.new_param_names = all_param_names - self.pretrained_param_names
        print(f'Loaded pretrained params: {len(self.pretrained_param_names)}')
        print(f'New params: {len(self.new_param_names)}')

        # self.apply(_weights_init)

    def forward(self, x, batch_meta=None):
        bz, seq_len, ch_num, epoch_size = x.shape

        x = x.contiguous().view(bz * seq_len, ch_num, 30, 200)
        epoch_features = self.backbone(x, batch_meta=batch_meta)
        epoch_features = epoch_features.contiguous().view(bz, seq_len, ch_num*30*200)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        out = self.classifier(seq_features)
        return out
