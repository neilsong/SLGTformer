from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.attention import DepthWiseConv2d, bn_init, conv_init, import_class
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from timm.models.layers import trunc_normal_
from einops import rearrange
from model.new_attention import Attention, FeedForward, PreNorm
from model.twins_attention import PEG, PatchEmbedding, Transformer

class tattn_block(nn.Module):
    def __init__(self, in_channels, out_channels, t_num_heads=4, num_point=25, block_size=41, t_depth=1, t_patch=2, global_k=5, is_last=False):
        super(tattn_block, self).__init__()

        self.layers = nn.Sequential(
                PatchEmbedding(dim = in_channels, dim_out = out_channels, patch_size = 4),
                Transformer(dim = out_channels, depth = 1, local_patch_size = t_patch, global_k = global_k, has_local = not is_last, heads=t_num_heads),
                PEG(dim = out_channels),
                Transformer(dim = out_channels, depth = t_depth,  local_patch_size = t_patch, global_k = global_k, has_local = not is_last, heads=t_num_heads),
        )

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.layers(x)
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


class attn_block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4, s_pos_emb=False, t_pos_emb=False, **kwargs):
        super().__init__()
        self.s_pos_emb = s_pos_emb
        self.t_pos_emb = t_pos_emb

        if self.s_pos_emb:
            self.spatial_pos_embed_layer = nn.Parameter(torch.zeros(1, kwargs['num_point'], dim))
            trunc_normal_(self.spatial_pos_embed_layer, std=.02)
        if self.t_pos_emb:    
            self.temporal_pos_embed_layer = nn.Parameter(torch.zeros(1, kwargs['window_size'], dim))
            trunc_normal_(self.temporal_pos_embed_layer, std=.02)

        self.attn = PreNorm(dim, Attention(dim, out_dim, heads = num_heads))
        self.ff = PreNorm(out_dim, FeedForward(out_dim, mult=mlp_ratio))

        self.residual = nn.Linear(dim, out_dim) if out_dim != dim else nn.Identity()

    def forward(self, x):
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
        x = self.residual(x) + self.attn(x)
        x = x + self.ff(x)
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
                attn_block(out_channels, out_channels * num_subset, num_heads=s_num_heads, mlp_ratio=4, s_pos_emb=s_pos_emb, t_pos_emb=t_pos_emb, num_point=num_point, **kwargs)
            )
        else:
            self.attention_block = attn_block(in_channels, out_channels * num_subset, num_heads=s_num_heads,  mlp_ratio=4, s_pos_emb=s_pos_emb, t_pos_emb=t_pos_emb, num_point=num_point,)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(num_point))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(
            eye_array), requires_grad=False, device='cuda'), requires_grad=False)  # [c,25,25]

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


class SATTN_TCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, num_point, block_size, stride=1, residual=True, s_pos_emb=False, s_num_heads=1, t_pos_emb=False, t_num_heads=4, t_depth=1, t_patch=2, global_k=5, is_last=False, **kwargs):
        super(SATTN_TCN_unit, self).__init__()
        self.spatial_attn1 = unit_sattn(in_channels, out_channels, A, groups, num_point, s_pos_emb=s_pos_emb, s_num_heads=s_num_heads, t_pos_emb=t_pos_emb, **kwargs)
        self.temporal_attn1 = tattn_block(out_channels, out_channels, t_num_heads=t_num_heads, num_point=num_point, block_size=block_size, t_depth=t_depth, t_patch=t_patch, global_k=global_k, is_last=is_last)
        self.relu = nn.ReLU()

        self.A = nn.Parameter(torch.tensor(np.sum(np.reshape(A.astype(np.float32), [
                              3, num_point, num_point]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

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
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(), in_channels=3, inner_dim=64, drop_layers=3, depth=4, s_num_heads=1, t_num_heads=[2, 4, 8, 16], window_size=120, s_pos_emb=True, t_pos_emb=False, t_depth_arr=[1,1,5,4], t_patch_arr=[8, 4, 2, 1], global_k=5):
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
                SATTN_TCN_unit(in_channels, inner_dim, A, groups, num_point, block_size, residual=False, s_pos_emb=s_pos_emb, s_num_heads=s_num_heads, t_pos_emb=t_pos_emb, t_num_heads=t_num_heads[0], window_size=window_size, t_depth=t_depth_arr[0], t_patch=t_patch_arr[0], global_k=global_k)
                if i == 0 else
                SATTN_TCN_unit(inner_dim * inner_dim_expansion[i-1], inner_dim * inner_dim_expansion[i], A, groups, num_point, block_size, stride=inner_dim_expansion[i] // inner_dim_expansion[i-1], residual=True, s_pos_emb=s_pos_emb, t_pos_emb=t_pos_emb, s_num_heads=s_num_heads, t_num_heads=t_num_heads[i], t_depth=t_depth_arr[i], t_patch=t_patch_arr[i], global_k=global_k, is_last=(i == depth - 1))
                for i in range(depth)
            ]
        )

        self.fc = nn.Linear(inner_dim * inner_dim_expansion[-1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
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
