import math
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
    n_buckets: int = 32 # number of buckets
    is_causal: bool = True # causal or not
    qkv_bias: bool = True # whether to use bias for the key and value


class RelativePositionBias(nn.Module):
    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16
    """
    def __init__(self, config):
        super(RelativePositionBias, self).__init__()
        self.is_causal = config.is_causal
        self.num_buckets = config.n_buckets
        self.max_distance = config.seq_len
        self.n_heads = config.n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, is_causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not is_causal:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            is_causal=self.is_causal,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1])  # shape (num_heads, qlen, klen)
        return values
    

class RelativeMultiHeadAttention(nn.Module):
    """
    This class implements the ALiBi multi-head attention
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model is not divisible by h"
        self.qkv_bias = config.qkv_bias
        self.pos_embedding = RelativePositionBias(config)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.query = nn.Linear(self.d_model,self.d_model, bias = self.qkv_bias)
        self.key = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.value = nn.Linear(self.d_model, self.d_model, bias = self.qkv_bias)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.scale = self.d_head ** -0.5
        self.is_causal = config.is_causal


    def forward(self, x: torch.tensor) -> torch.tensor:
        B, L, D = x.shape
        q = self.query(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.key(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.value(x).view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        attn_mask = self.pos_embedding(L, L).unsqueeze(0).to(q.device)
        if self.is_causal:
            temp_mask = torch.ones(L, L, dtype=torch.bool).tril(diagonal=0)
            attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
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
        self.attn = RelativeMultiHeadAttention(config)
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
        self.attn = RelativeMultiHeadAttention(config)

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
    
