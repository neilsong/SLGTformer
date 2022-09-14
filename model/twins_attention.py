import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size
        self.proj = nn.Conv2d(patch_size * dim, dim_out, 1)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (t p) v -> b (c p) t v', p = p)
        return self.proj(fmap)

class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = (kernel_size, 1), padding = (kernel_size // 2, 0), groups = dim, stride = 1))

    def forward(self, x):
        return self.proj(x)

class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, t, V, h = *shape, self.heads
        t = t // p

        fmap = rearrange(fmap, 'b c (t p) v -> (b t v) p c', p=p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b p (h d) -> (b h) p d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = - 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b t v h) p d -> b (h d) (t p) v', h = h, t = t, v = V, p = p)
        return self.to_out(out)

class GlobalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, (k, 1), stride = (k, 1), bias = False)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, V, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        q, k, v = map(lambda t: rearrange(t, 'b (h d) t v -> (b v h) t d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b v h) t d -> b (h d) t v', h = h, v = V)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_head = 64, mlp_mult = 4, local_patch_size = 7, global_k = 7, dropout = 0., has_local = True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = global_k))) if has_local else nn.Identity(),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))) if has_local else nn.Identity(),
                Residual(PreNorm(dim, GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = local_patch_size))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout)))
            ]))
    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x