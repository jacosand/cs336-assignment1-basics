import torch
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x: Float[Tensor, '...'], dim: int) -> Float[Tensor, '...']:

    y = x - x.max(dim=dim, keepdim=True).values
    y = torch.exp(y)

    return y / y.sum(dim=dim, keepdim=True)


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
    ) -> float:

    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
    target_logits = torch.gather(
        shifted_logits, dim=-1, index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return torch.mean(torch.logsumexp(shifted_logits, dim=-1) - target_logits)