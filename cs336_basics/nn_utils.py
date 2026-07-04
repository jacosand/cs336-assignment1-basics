import torch
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, '...'], dim: int) -> Float[Tensor, '...']:

    y = x - x.max(dim=dim, keepdim=True).values
    y = torch.exp(y)

    return y / y.sum(dim=dim, keepdim=True)