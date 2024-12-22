import math
import torch 
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class SinusoidalPositionalEncoding(nn.Module):
    """
    This class implements absolute positional encoding from the paper "Attention is all you need""
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]) # (batch, seq_len, d_model)
        return x
    

class AbsolutePositionalEncoding(nn.Module):
    """
    This class implements learnable absolute positional encoding
    """
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.emb = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) 
        pos_emb = self.emb(pos).unsqueeze(0)
        print(pos_emb.shape)
        x = x + pos_emb 
        return x

class MultiHeadAttention(nn.Module):
    """
    This class implements classical Multi-Head Causal Attention
    """
    def __init__(self, d_model, n_heads, qkv_bias = True, is_causal=True):
        super().__init__()
        self.is_causal = is_causal
        assert d_model % n_heads == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.key = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.value = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.key(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, scale=self.scale)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)

        out = self.out(out)
        return out
    

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class OnlyAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, qkv_bias)

    def forward(self, x):
        out = x + self.attn(self.norm(x))
        return out
    

class TransformerDecoder(nn.Module):
    def __init__(self, BasicBlock, vocab_size, d_model, n_heads, n_blocks, qkv_bias=True, seq_len=8192, PositionalEncoding=None):
        super().__init__()
        if PositionalEncoding is None:
            self.pos = None
        else:
            self.pos = PositionalEncoding(d_model, seq_len)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[BasicBlock(d_model, n_heads, qkv_bias) for _ in range(n_blocks)])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # Здесь косячок. Инициализирует только часть слоев!
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.to(next(iter(self.parameters())).device)
        emb = self.embedding(x)
        if self.is_pos is not None:
            emb = self.pos(emb)

        x = self.blocks(emb)

        out = self.lm_head(self.final_ln(x))

        return out
    


