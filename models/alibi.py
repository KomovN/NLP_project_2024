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


class ALiBiEmbedding(nn.Module):
    """
    This class implements the ALiBi positional embedding
    """
    def __init__(self, config):
        super().__init__()
        x = (2 ** 8) ** (1 / config.n_heads)
        self.m = (
            torch.tensor([1 / x ** (i + 1) for i in range(config.n_heads)])
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def forward(self, seq_len):
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return self.m * (x - y)
    

class ALiBiMultiHeadAttention(nn.Module):
    """
    This class implements the ALiBi multi-head attention
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model is not divisible by h"
        self.is_causal = config.is_causal
        self.qkv_bias = config.qkv_bias
        self.pos_embedding = ALiBiEmbedding(config)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.query = nn.Linear(self.d_model,self.d_model, bias = self.qkv_bias)
        self.key = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.value = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, L, D = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.key(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn_mask = self.pos_embedding(L).unsqueeze(0).to(q.device)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal, scale=self.scale)
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
        self.attn = ALiBiMultiHeadAttention(config)
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
        self.attn = ALiBiMultiHeadAttention(config)

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
    
