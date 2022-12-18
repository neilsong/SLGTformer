from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.attention import MHSA, RPE_MHSA, DepthWiseConv2d, Mlp, bn_init, conv_init, import_class
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

class attn_block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  attn_drop=0.0, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grpe=True, s_pos_emb=False, t_pos_emb=False, **kwargs):
        super().__init__()
        dim = dim * num_heads
        self.norm1 = norm_layer(dim)
        self.use_grpe = use_grpe
        self.pre_proj = nn.Sequential(
            nn.Conv2d(dim // num_heads, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        ) if num_heads > 1 else nn.Identity()
        self.s_pos_emb = s_pos_emb
        self.t_pos_emb = t_pos_emb
        if self.s_pos_emb:
            self.spatial_pos_embed_layer = nn.Parameter(torch.zeros(1, kwargs['num_point'], dim))
            trunc_normal_(self.spatial_pos_embed_layer, std=.02)
        if self.t_pos_emb:    
            self.temporal_pos_embed_layer = nn.Parameter(torch.zeros(1, kwargs['window_size'], dim))
            trunc_normal_(self.temporal_pos_embed_layer, std=.02)
        if self.use_grpe:
            self.attn = RPE_MHSA(dim, out_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, num_point=kwargs['num_point'])
        else:
            self.attn = MHSA(dim, out_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.residual = nn.Linear(dim, out_dim) if out_dim != dim else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.pre_proj(x)
        B, T = x.shape[0], x.shape[2]

        # rearrange spatial joints as tokens
        x = rearrange(x, 'b c t v -> (b t) v c')

        if self.s_pos_emb:
            x = x + self.spatial_pos_embed_layer
        if self.t_pos_emb:
            x = rearrange(x, '(b t) v c -> (b v) t c', b=B)
            if T != self.temporal_pos_embed_layer.size(1):
                time_embed = self.temporal_pos_embed_layer.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.temporal_pos_embed_layer
            x = rearrange(x, '(b v) t c -> (b t) v c', b=B)

        # attention + residual
        x = self.residual(x) + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, '(b t) v c -> b c t v', b=B)
        return x

class unit_sattn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, num_subset=3, s_pos_emb=False, s_num_heads=1, t_pos_emb=False, **kwargs):
        super(unit_sattn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [
                                      3, 1, num_point, num_point]), dtype=torch.float32, requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        if s_pos_emb:
            self.attention_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                attn_block(
                    out_channels, 
                    out_channels * num_subset, 
                    num_heads=s_num_heads, 
                    mlp_ratio=4., 
                    qkv_bias=False, 
                    qk_scale=None, 
                    attn_drop=0.0, 
                    drop=0., 
                    drop_path=0.,
                    act_layer=nn.GELU,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                    s_pos_emb=s_pos_emb, 
                    t_pos_emb=t_pos_emb, 
                    num_point=num_point, 
                    **kwargs
                )
            )
        else:
            self.attention_block = attn_block(
                in_channels, 
                out_channels * num_subset, 
                num_heads=s_num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                attn_drop=0.0,
                drop=0., 
                drop_path=0., 
                act_layer=nn.GELU, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_point= num_point,
            )

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False), requires_grad=False)  # [c,25,25]

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(
            1, self.out_channels // self.groups, 1, 1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1, ...]), self.norm(
            learn_A[1:2, ...]), self.norm(learn_A[2:3, ...])], 0)

        x = self.attention_block(x0)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x