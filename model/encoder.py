import torch
import torch.nn as nn

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dim_out, dim_pe, layers, n_heads, post_layers, max_freqs):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        # Encode integer node features via nn.Embeddings
        self.node_encoder = LapPENodeEncoder(dim_in, dim_out, dim_pe=dim_pe, layers=layers, n_heads=n_heads, post_layers=post_layers, max_freqs=max_freqs)
        # Encode integer edge features via nn.Embeddings
        self.edge_encoder = LinearEdgeEncoder(dim_out)

    def forward(self, batch):
        batch = self.node_encoder(batch)
        batch = self.edge_encoder(batch)
        return batch


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_in, dim_out, dim_pe, layers, n_heads, post_layers, max_freqs, raw_norm_type="none", pass_as_var=False, expand_x=True, model_type="DeepSet"):
        super().__init__()

        dim_pe = dim_pe  # Size of Laplace PE embedding
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = layers  # Num. layers in PE encoder model
        n_heads = n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = post_layers  # Num. layers to apply after pooling
        max_freqs = max_freqs  # Num. eigenvectors (frequencies)
        norm_type = raw_norm_type # Raw PE normalization layer type
        self.pass_as_var = pass_as_var  # Pass PE also as a separate variable

        if dim_out - dim_pe < 1:
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_out}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_out - dim_pe)
        self.expand_x = expand_x

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        if model_type == 'Transformer':
            # Transformer model for LapPE
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                       nhead=n_heads,
                                                       batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            layers = []
            if n_layers == 1:
                layers.append(nn.ReLU())
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch

class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.in_dim = 1
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))
        return batch