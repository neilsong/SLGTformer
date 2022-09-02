from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
# from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from model.gated_gcn_layer import GatedGCNLayer

class SLGTLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_in, dim_out, num_heads, attn_dropout=0.0, residual=True):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.residual = residual

        self.local_model = GatedGCNLayer(dim_in, dim_out,
                                            dropout=0,
                                            residual=self.residual)

        self.self_attn = torch.nn.MultiheadAttention(
            dim_out, num_heads, dropout=self.attn_dropout, batch_first=True)

        self.norm1_local = nn.BatchNorm1d(dim_out)
        self.norm1_attn = nn.BatchNorm1d(dim_out)
        self.norm2 = nn.BatchNorm1d(dim_out)
        # self.norm1_time_attn = nn.BatchNorm1d(dim_out)
        
        # self.dropout_local = nn.Dropout(dropout)
        # self.dropout_attn = nn.Dropout(dropout)
        # self.dropout_time_attn  = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_out, dim_out * 2)
        self.ff_linear2 = nn.Linear(dim_out * 2, dim_out)
        # self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            es_data = None
            local_out = self.local_model(Batch(batch=batch,
                                                x=h,
                                                edge_index=batch.edge_index,
                                                edge_attr=batch.edge_attr,
                                                pe_EquivStableLapPE=es_data))
            # GatedGCN does residual connection and dropout internally.
            h_local = local_out.x
            batch.edge_attr = local_out.edge_attr

            h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head spatial attention
        if self.self_attn is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            h_attn = self._sa_block(h_dense, None, ~mask)[mask]

            # h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ta_block(self, x, attn_mask, key_padding_mask):
        """Temporal attention blocks.
        """
        x = self.temporal_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        # x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        # return self.ff_dropout2(self.ff_linear2(x))

        # no dropout
        x = self.activation(self.ff_linear1(x))
        return self.ff_linear2(x)

    def extra_repr(self):
        s = f'summary: dim_out={self.dim_out}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
