import mlx.core as mx
import mlx.nn as nn
import math
from typing import Tuple, Optional

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        mu = mx.mean(x**2, axis=-1, keepdims=True)
        x = x * mx.rsqrt(mu + self.eps)
        return (x.astype(orig_dtype) * self.weight)

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(end).astype(mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.cos(freqs), mx.sin(freqs)

def apply_rotary_emb(xq, xk, cos, sin):
    # xq: (B, T, nh, hs), cos: (T, hs/2), sin: (T, hs/2)
    B, T, nh, hs = xq.shape
    _, _, nkh, _ = xk.shape
    
    xq_real = xq[..., ::2]
    xq_imag = xq[..., 1::2]
    xk_real = xk[..., ::2]
    xk_imag = xk[..., 1::2]
    
    # Broadcast cos/sin
    cos = cos[None, :T, None, :]
    sin = sin[None, :T, None, :]
    
    xq_out_real = xq_real * cos - xq_imag * sin
    xq_out_imag = xq_real * sin + xq_imag * cos
    xk_out_real = xk_real * cos - xk_imag * sin
    xk_out_imag = xk_real * sin + xk_imag * cos
    
    xq_out = mx.stack([xq_out_real, xq_out_imag], axis=-1).reshape(B, T, nh, hs)
    xk_out = mx.stack([xk_out_real, xk_out_imag], axis=-1).reshape(B, T, nkh, hs)
    
    return xq_out, xk_out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_embd: int, n_kv_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = n_embd // n_head
        self.n_groups = n_head // n_kv_head
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(B, T, self.n_head, self.head_dim)
        xk = xk.reshape(B, T, self.n_kv_head, self.head_dim)
        xv = xv.reshape(B, T, self.n_kv_head, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # (B, nh, T, hs)
        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        if self.n_groups > 1:
            xk = mx.repeat(xk, self.n_groups, axis=1)
            xv = mx.repeat(xv, self.n_groups, axis=1)

        # Scaled dot product attention
        scores = (xq @ xk.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(xq.dtype)
        scores = self.dropout(scores)
        out = scores @ xv

        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        return self.dropout(self.w3(nn.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, n_embd, n_kv_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.attention_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)

    def __call__(self, x, cos, sin, mask):
        x = x + self.attention(self.attention_norm(x), cos, sin, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, n_kv_head: int, dropout: float, block_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = [Block(n_embd, n_head, n_kv_head, dropout) for _ in range(n_layer)]
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.cos, self.sin = precompute_freqs_cis(n_embd // n_head, block_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)

        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(x.dtype)
        
        cos = self.cos[:T]
        sin = self.sin[:T]

        for block in self.blocks:
            x = block(x, cos, sin, mask)
            
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))

        return logits, loss
