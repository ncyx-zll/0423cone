import torch 
import torch.nn as nn
from .mlp import MLP
from .ldm.modules.diffusionmodules.util import (
    linear,
    conv_nd,
    zero_module,

)
import einops

def exists(x):
    return x is not None


class TimeStepBlock(nn.Module):
    def __init__(self,
                 channels,
                 emb_channels,
                 out_channels = 256,
                 dims = 1,
                 dropout = 0.2,
                 use_scale_shift_norm = True):
        super(TimeStepBlock, self).__init__()
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        # self.out_norm = nn.LayerNorm(self.out_channels)
        # self.out_norm = nn.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6)
        # self.out_layers = nn.Sequential(
        #     # nn.LayerNorm(self.out_channels),
        #     nn.SiLU(),
        #     nn.Dropout(p=dropout),
        #     zero_module(
        #         conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        #     ),
        # )

    def forward(self, x, time_embed):
        if time_embed is None:
            return x
        if x.dim() not in (2, 3):
            raise ValueError(f"Unsupported x dim {x.dim()}, expected 2D or 3D tensor, got {tuple(x.shape)}")

        emb_out = self.emb_layers(time_embed).type(x.dtype)

        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=-1)
            if scale.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"Hidden dim mismatch between x ({x.shape[-1]}) and modulation ({scale.shape[-1]})"
                )

            # Align modulation tensors to x layout (e.g. x: [num_queries, batch, dim]).
            if x.dim() == 3:
                if scale.dim() == 2:
                    if scale.shape[0] != x.shape[1]:
                        raise ValueError(
                            f"Expected 2D modulation as [B,D] with B={x.shape[1]}, got {tuple(scale.shape)}"
                        )
                    # [batch, dim] -> [1, batch, dim], broadcast on query axis.
                    scale = scale.unsqueeze(0)
                    shift = shift.unsqueeze(0)
                elif scale.dim() == 3:
                    if scale.shape[:2] == x.shape[:2]:
                        pass
                    elif scale.shape[0] == x.shape[1] and scale.shape[1] == x.shape[0]:
                        # [batch, num_queries, dim] -> [num_queries, batch, dim]
                        scale = scale.permute(1, 0, 2)
                        shift = shift.permute(1, 0, 2)
                    else:
                        raise ValueError(
                            f"Incompatible shapes: x={tuple(x.shape)}, scale={tuple(scale.shape)}"
                        )
                else:
                    raise ValueError(
                        f"Unsupported scale dim {scale.dim()} for x dim {x.dim()}"
                    )
            else:
                if scale.dim() != 2 or scale.shape != x.shape:
                    raise ValueError(
                        f"Expected 2D x and modulation to share shape, got x={tuple(x.shape)}, scale={tuple(scale.shape)}"
                    )

            return x * (1 + scale) + shift

        return x

class BBoxEmbed(nn.Module):
    def __init__(self,embed_dim,
                time_embed_channels,):
        super(BBoxEmbed, self).__init__()
        # 【修改这里】：把 4 改成 2，因为我们只预测 [t_center, d] 两个值
        self.pred = MLP(embed_dim, embed_dim, 2, 3)
        self.norm = nn.LayerNorm(embed_dim)
        # self.time_embedding_layer = 
        self.time_step_embed = TimeStepBlock(
                    channels = embed_dim,
                    emb_channels = time_embed_channels,
                    # channels = 256,
                )
    def forward(self , x , time_embed = None):
        # x = self.time_step_embed(x , time_embed)
        # x = self.pred(self.norm(x))
        x = self.pred(x)
        return x
    
class ClassEmbed(nn.Module):
    def __init__(self,embed_dim,
                time_embed_channels,
                num_classes):
        super(ClassEmbed , self).__init__()
        # self.attention = CrossAttention(query_dim=embed_dim, context_dim=context_dim,
        #                             heads=n_heads, dim_head=d_head, dropout=dropout)
        self.num_classes = num_classes
        self.pred = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
                    channels = embed_dim,
                    emb_channels = time_embed_channels,
                    # channels = 256,
                )

    def forward(self , x , time_embed=None):
        # x = self.time_step_embed(x , time_embed)
        # x = self.pred(self.norm(x))
        x = self.pred(x)
        return x
