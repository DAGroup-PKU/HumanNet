"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn


# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")
    
    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)
        
class EfficientChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        hidden_dim = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(hidden_dim, dim, bias=qkv_bias)
        
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        q = self.q_proj(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        k = x.reshape(B, N, H, D).permute(0, 2, 1, 3)
        
        q = q.permute(0, 1, 3, 2)
        attn_logits = (q @ k) * self.scale
        attn = torch.sigmoid(attn_logits) * self.temperature
        
        v = self.mlp_fc1(x)
        v = self.mlp_act(v)
        v = self.mlp_fc2(v)
        v = self.norm(v)
        
        v = v.reshape(B, N, H, D).permute(0, 2, 1, 3)
        v_t = v.transpose(-1, -2)
        x_out = attn @ v_t
        
        x_out = x_out.transpose(-1, -2).permute(0, 2, 1, 3).reshape(B, N, C)
        x_out = self.proj(x_out)
        
        return x_out
    
class FusedFANProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        
        self.initial_projection_dim = fused_vision_dim * 4
        self.clean_body = nn.Sequential(
            nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
            nn.GELU()
        )
        self.fan_in_proj = nn.Linear(fused_vision_dim, llm_dim, bias=True)
        self.act = nn.GELU()
        self.fan_block = EfficientChannelAttention(llm_dim, num_heads=16, mlp_ratio=4.0)
        self.shared_tail = nn.Linear(llm_dim, llm_dim, bias=True)
        self.layer_norm = nn.LayerNorm(llm_dim)
        self.drop_prob = drop_prob
        self.dropout = nn.Dropout(drop_prob)
        self.register_buffer('gate', torch.tensor([0.3]))    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_clean = self.clean_body(x)
        
        if self.training:
            feat_clean = self.dropout(feat_clean)
            
        feat_fan = self.fan_in_proj(x)
        feat_fan = self.act(feat_fan)
        feat_robust = self.fan_block(feat_fan)
        
        combined_feat = feat_clean + torch.tanh(self.gate) * feat_robust
        out = self.layer_norm(combined_feat)
        out = self.shared_tail(out)
        
        return out