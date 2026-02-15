import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadAttention(nn.Module):
    """Grouped-Query Attention (GQA) with RoPE."""
    def __init__(self, n_head, n_embd, block_size, dropout, n_kv_head=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = n_embd // n_head
        self.n_groups = self.n_head // self.n_kv_head
        
        self.wq = nn.Linear(n_embd, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, n_embd, bias=False)
        
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat KV heads if using GQA
        if self.n_kv_head != self.n_head:
            xk = torch.repeat_interleave(xk, self.n_groups, dim=2)
            xv = torch.repeat_interleave(xv, self.n_groups, dim=2)

        # (B, nh, T, hs)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Efficient implementation with FlashAttention support
        out = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.wo(out))
        return out

class FeedForward(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3)
        # Llama-style SwiGLU: (xW1 * swish(xW2))W3
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    """Transformer block with RMSNorm and GQA support."""
    def __init__(self, n_embd, n_head, block_size, dropout, n_kv_head=None):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, n_embd, block_size, dropout, n_kv_head)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.attention_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPT(nn.Module, PyTorchModelHubMixin):
    """Full decoder-only Transformer (Llama-style)."""
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.2, n_kv_head=None):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, n_kv_head) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(n_embd // n_head, block_size))
        
        print(f"Model created with {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B, T, C)
        
        freqs_cis = self.freqs_cis[:T]
        
        for block in self.blocks:
            x = block(x, freqs_cis)
            
        x = self.norm(x)       # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
