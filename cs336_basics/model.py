import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
from einops import einsum, rearrange

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
        
        inv_rms = (einsum(x**2, "... d_model -> ...") / self.d_model + self.eps) ** -0.5
        
        return einsum(x, inv_rms, self.weight, "... d_model, ..., d_model -> ... d_model")