import torch
import torch.nn as nn
import numpy as np
import math
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from model.slgt_layer import SLGTLayer
from einops import rearrange
from model.encoder import FeatureEncoder


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, groups=8, block_size=41, graph=None, graph_args=dict(), in_channels=3, 
        num_layers=4, num_heads=4, dim_h=52, attn_dropout=0.5,
        dim_pe=8, enc_layers=2, enc_n_heads=4, enc_post_layers=0, max_freqs=8):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.encoder = FeatureEncoder(in_channels, dim_h, dim_pe, layers=enc_layers, n_heads=enc_n_heads, post_layers=enc_post_layers, max_freqs=max_freqs)

        layers = []
        for _ in range(num_layers):
            layers.append(SLGTLayer(
                dim_in=dim_h,
                dim_out=dim_h,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                residual=True,
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.fc = nn.Linear(dim_h, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, batch, keep_prob=0.9):
        x = batch.x
        x = rearrange(x, '(n t v) (c m) -> n c t v m', m=1, v=27, c=x.size(1), t=batch.window[0])
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        batch.x = rearrange(x, '(n m) c t v -> (n t v) (c m)', n=N, m=M, t=T, v=V, c=C)

        batch  = self.encoder(batch)
        batch = self.layers(batch)
        # N*M,C,T,V
        c_new = batch.x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = batch.x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
