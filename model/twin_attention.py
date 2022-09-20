from functools import partial
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.attention import MHSA, DepthWiseConv2d, Mlp, bn_init, conv_init, import_class
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from model.grpe_attention import unit_sattn
from model.twins_attention_orig import TwinSVT

class tattn_block(nn.Module):
    def __init__(self, in_channels, out_channels, t_num_heads=4, kernel_size=1, stride=1, num_point=25, block_size=41, depth=1, wss=6, sr_ratios=8, **kwargs):
        super(tattn_block, self).__init__()

        h_dim = in_channels * t_num_heads

        self.pre_proj = nn.Sequential(
            nn.Conv2d(in_channels, h_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_dim),
        )
        self.post_proj = nn.Sequential(
            nn.Conv2d(h_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )


        # time_size = kwargs["window_size"]
        # i_layer = kwargs["i"]

        # patches_resolution = time_size // 2 ** i_layer

        self.attn = TwinSVT(
            embed_dims=[ h_dim ], 
            num_heads=[ t_num_heads ], 
            mlp_ratios=[ 4 ], 
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[ depth ], 
            wss=[ wss ], 
            sr_ratios=[ sr_ratios ],
        )

        self.pool = DepthWiseConv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=0, stride=(stride, 1)) # change kernel size to (t_num_heads, 1) to see if test efficacy of depthwise conv
        self.bn = nn.BatchNorm2d(out_channels)

        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.pre_proj(x)
        B, T = x.shape[0], x.shape[2]


        x = rearrange(x, 'b c t v -> (b v) t c')
        x = self.attn(x, T, 1).flatten(-2)
        x = rearrange(x, '(b v) c t -> b c t v', b=B)

        x = self.post_proj(x)
        x = self.bn(self.pool(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(unit_tcn_skip, self).__init__()
        
        self.pool = DepthWiseConv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=0, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.pool(x))
        return x


class SATTN_TCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True, use_grpe=True, s_pos_emb=False, s_num_heads=1, t_pos_emb=False, t_num_heads=4, is_first=False, t_depth=1, **kwargs):

        super(SATTN_TCN_unit, self).__init__()

        self.spatial_attn1 = unit_sattn(
                in_channels, 
                out_channels if is_first else in_channels,
                A, groups, num_point, 
                s_pos_emb=s_pos_emb, 
                s_num_heads=s_num_heads, 
                t_pos_emb=t_pos_emb, 
                is_first=is_first,
                use_grpe=use_grpe,
                **kwargs
        )

        self.temporal_attn1 = tattn_block(
            out_channels if is_first else in_channels,
            out_channels, 
            t_num_heads=t_num_heads, 
            stride=stride, 
            num_point=num_point, block_size=block_size, 
            depth=t_depth,
            window_size = kwargs["window_size"],
            i = kwargs["i"],
            wss = kwargs["wss"],
            sr_ratios = kwargs["sr_ratios"],
        )

        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False), requires_grad=False)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride)
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob):
        y = self.spatial_attn1(x)
        y = self.temporal_attn1(y, keep_prob, self.A)
        x_skip = self.dropT_skip(self.dropSke(self.residual(x), keep_prob, self.A), keep_prob)
        return self.relu(y + x_skip)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(), in_channels=3, use_grpe=True, inner_dim=64, drop_layers=3, depth=4, s_num_heads=1, t_num_heads=8, window_size=120, s_pos_emb=True, t_pos_emb=True, t_depths=[2, 2, 2, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1]):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.drop_layers = depth - drop_layers

        inner_dim_expansion = [ 2 ** i for i in range(0, depth) ]

        self.layers = nn.ModuleList(
            [
                SATTN_TCN_unit(
                        in_channels, 
                        inner_dim, 
                        A, 
                        groups, 
                        num_point, 
                        block_size, 
                        residual=False, 
                        use_grpe=use_grpe, 
                        s_num_heads=s_num_heads, 
                        t_num_heads=t_num_heads, 
                        t_depth = t_depths[0],
                        
                        s_pos_emb=s_pos_emb, 
                        t_pos_emb=t_pos_emb, 
                        window_size=window_size, 
                        i=i,
                        is_first=True, 

                        wss=wss[0],
                        sr_ratios=sr_ratios[0],
                )
                if i == 0 else
                SATTN_TCN_unit(
                    inner_dim * inner_dim_expansion[i-1], 
                    inner_dim * inner_dim_expansion[i], 
                    A, groups, num_point, block_size, 
                    stride=inner_dim_expansion[i] // inner_dim_expansion[i-1], 
                    residual=True, 
                    use_grpe=use_grpe,
                    s_num_heads=s_num_heads, 
                    t_num_heads=t_num_heads,
                    t_depth = t_depths[i],
                    
                    window_size=window_size, 
                    i=i,

                    wss=wss[i],
                    sr_ratios=sr_ratios[i],
                )
                for i in range(depth)
            ]
        )

        self.fc = nn.Linear(inner_dim * inner_dim_expansion[-1], num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for u, blk in enumerate(self.layers):
            x = blk(x, 1.0 if u < self.drop_layers else keep_prob)

        # N*M,C,T,V
        c_new = x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
