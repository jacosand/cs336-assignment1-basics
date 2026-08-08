import torch
from jaxtyping import Bool, Float, Int
from typing import Literal
from torch import Tensor
from torch import nn
from einops import einsum, rearrange

from cs336_basics import nn_utils

class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        sigma = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(
        self,
        x: Float[Tensor, "... in_features"],
    ) -> Float[Tensor, "... out_features"]:
        return einsum(x, self.weight, '... in_features, out_features in_features -> ... out_features')


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3, b=3)
    
    def forward(
        self,
        token_ids: Int[Tensor, "..."],
    ) -> Float[Tensor, "... embedding_dim"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
    
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
    ) -> Float[Tensor, "... d_model"]:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        inv_rms = (einsum(x**2, "... d_model -> ...") / self.d_model + self.eps) ** -0.5
        
        result = einsum(x, inv_rms, self.weight, "... d_model, ..., d_model -> ... d_model")

        return result.to(in_dtype)


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Literal["swiglu", "silu"] = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        if activation not in {"swiglu", "silu"}:
            raise ValueError(f"Activation function {activation} not supported, must be swiglu or silu.")

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        if activation == "swiglu":
            self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        
        if self.activation == "swiglu":
            hidden = silu(self.w1(x)) * self.w3(x)
        elif self.activation == "silu":
            hidden = silu(self.w1(x))
        return self.w2(hidden)


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        assert d_k % 2 == 0, "d_k must be divisible by 2"

        pos_index = torch.arange(max_seq_len, device=device)
        vec_index = torch.arange(d_k // 2, device=device)

        inv_freq = theta ** (-2 * vec_index / d_k)
        theta_ik = einsum(pos_index, inv_freq, 'i, k -> i k')

        self.register_buffer("cos_theta", torch.cos(theta_ik), persistent=False)
        self.register_buffer("sin_theta", torch.sin(theta_ik), persistent=False)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_k"]:
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        seq_len = x.size(-2)
        
        if token_positions is not None:
            cos_values = self.cos_theta[token_positions]
            sin_values = self.sin_theta[token_positions]
        else:
            cos_values = self.cos_theta[:seq_len]
            sin_values = self.sin_theta[:seq_len]

        result = torch.empty_like(x)

        result[..., ::2] = x_even * cos_values - x_odd * sin_values
        result[..., 1::2] = x_even * sin_values + x_odd * cos_values 

        return result


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None,
) -> Float[Tensor, "... queries d_v"]:
    
    d_k = Q.size(-1)
    att = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / d_k ** 0.5
    if mask is not None:
        att -= torch.where(~mask, float('inf'), 0)
    att = nn_utils.softmax(att, dim=-1)
    return einsum(att, V, '... queries keys, ... keys d_v -> ... queries d_v')


class CausalMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        position_encoder: nn.Module | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.position_encoder = position_encoder
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        q = rearrange(q, "... queries (num_heads head_size) -> ... num_heads queries head_size", num_heads=self.num_heads)
        k = rearrange(k, "... keys (num_heads head_size) -> ... num_heads keys head_size", num_heads=self.num_heads)
        v = rearrange(v, "... keys (num_heads head_size) -> ... num_heads keys head_size", num_heads=self.num_heads)
        if self.position_encoder is not None:
            q = self.position_encoder(q, token_positions)
            k = self.position_encoder(k, token_positions)
        mask = torch.tril(torch.ones(q.size(-2), k.size(-2), device=x.device, dtype=torch.bool))
        y = scaled_dot_product_attention(q, k, v, mask=mask)
        y = rearrange(y, "... num_heads queries head_size -> ... queries (num_heads head_size)")

        return self.output_proj(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        position_encoder: nn.Module | None = None,
        layer_norm: Literal["pre", "post", "none"] = "pre",
        activation: Literal["swiglu", "silu"] = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.position_encoder = position_encoder
        self.layer_norm = layer_norm
        self.activation = activation

        if layer_norm not in {"pre", "post", "none"}:
            raise ValueError(f"Layer norm type {layer_norm} not supported, must be pre, post, or none.")

        if activation not in {"swiglu", "silu"}:
            raise ValueError(f"Activation function {activation} not supported, must be swiglu or silu.")

        if layer_norm != "none":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiheadSelfAttention(d_model, num_heads, position_encoder=position_encoder, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, activation = activation, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        if self.layer_norm == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        elif self.layer_norm == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        elif self.layer_norm == "none":
            x = x + self.attn(x, token_positions)
            x = x + self.ffn(x)
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        position_encoding: Literal["rope", "none"] = "rope",
        rope_theta: float = 10_000,
        layer_norm: Literal["pre", "post", "none"] = "pre",
        activation: Literal["swiglu", "silu"] = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.layer_norm = layer_norm
        self.position_encoding = position_encoding
        self.activation = activation

        assert d_model % num_heads == 0, "num_heads must divide d_model"

        if layer_norm not in {"pre", "post", "none"}:
            raise ValueError(f"Layer norm type {layer_norm} not supported, must be pre, post, or none.")

        if activation not in {"swiglu", "silu"}:
            raise ValueError(f"Activation function {activation} not supported, must be swiglu or silu.")

        if position_encoding not in {"rope", "none"}:
            raise ValueError(f"Position encoding type {position_encoding} not supported, must be rope or none.")
        elif position_encoding == "rope":
            self.position_encoder = RotaryPositionEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
        else:
            self.position_encoder = None

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, position_encoder=self.position_encoder, layer_norm=layer_norm, activation=activation, device=device, dtype=dtype) for _ in range(num_layers)
        )
        if layer_norm == "pre":
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(
        self,
        x: Int[Tensor, "... seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        if self.layer_norm == "pre":
            x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "input_seq_len"],
        max_new_tokens: int,
        eot_token: int | None = None,
        temperature: float = 1,
        top_p: float = 0.9,
    ) -> Int[Tensor, "output_seq_len"]:

        if temperature <= 0:
            raise ValueError("temperature must be positive")
        
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in (0, 1]")
        
        for _ in range(max_new_tokens):

            context = x[-self.context_length:]
            
            logits = self(context)
            logits = logits[-1, :] / temperature
            probs = nn_utils.softmax(logits, dim=-1)
            
            probs_sorted, token_idx_sorted = torch.sort(probs, dim=-1, descending=True)
            cum_probs = torch.cumsum(probs_sorted, dim=-1)
            cum_probs_before = cum_probs - probs_sorted
            probs_filtered = torch.where(cum_probs_before < top_p, probs_sorted, 0.0)
            
            next_idx = torch.multinomial(probs_filtered, num_samples=1)
            next_token = token_idx_sorted[next_idx]

            x = torch.concat((x, next_token), dim=-1)

            if eot_token != None and next_token.item() == eot_token:
                break
            
        return x
