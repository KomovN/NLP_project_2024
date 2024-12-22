import torch 
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    seq_len: int = 512 # max sequence length
    vocab_size_in: int = 50257 # number of input tokens
    vocab_size_out: int = 50257 # number of output tokens 
    n_blocks: int = 12 # number of layers
    n_heads: int = 12 # number of heads
    d_model: int = 768 # embedding dimension
    is_causal: bool = True # causal or not
    qkv_bias: bool = True # whether to use bias for the key and value


class RotaryEmbedding(nn.Module):
    """
    This class implements RoPE (Rotary Position Embedding)
    """
    def __init__(self, d_head, end=8192, theta=10000.0):
        super().__init__()
        self.d_head = d_head
        self.end = end 
        self.theta = theta 
        self.freqs_cis = self.precompute_freqs_cis_()

    def precompute_freqs_cis_(self):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2)[: (self.d_head // 2)].float() / self.d_head))
        t = torch.arange(self.end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'{freqs_cis.shape} - {(x.shape[1], x.shape[-1])}'
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
        
    def forward(self, q, k):
        B, L, _, _ = q.shape
        freqs_cis = self.freqs_cis[:L].to(q.device)
        xq_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(q), xk_out.type_as(k)
    

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model is not divisible by h"
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_bias = config.qkv_bias
        self.query = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.key = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.value = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.pos = RotaryEmbedding(self.d_head)

    def forward(self, x):
        B, L, D = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.d_head)
        k = self.key(x).view(B, L, self.n_heads, self.d_head)
        v = self.value(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        q, k = self.pos(q, k)
        q, k = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)

        out = self.out(out)
        return out
    

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x
    

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = RoPEMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class OnlyAttentionBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.attn = RoPEMultiHeadAttention(config)

    def forward(self, x):
        out = x + self.attn(self.norm(x))
        return out
    

class TransformerDecoder(nn.Module):

    def __init__(self, config, BasicBlock=TransformerBlock):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size_in, config.d_model)
        self.blocks = nn.Sequential(*[BasicBlock(config) for _ in range(config.n_blocks)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size_out)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
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

        x = self.blocks(emb)

        out = self.lm_head(self.final_ln(x))

        return out
    
    @classmethod
    def only_attention(cls, config):
        return cls(config, OnlyAttentionBlock)
    
