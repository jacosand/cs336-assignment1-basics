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
        
        sigma = (2 / (in_features + out_features)) ** 0.5

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(
        self,
        x: Float[Tensor, "... in_features"],
    ) -> Float[Tensor, "... out_features"]:
        return einsum(x, self.weight, '... in_features, out_features in_features -> ... out_features')