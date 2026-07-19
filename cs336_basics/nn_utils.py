import torch
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Iterable


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


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> float:

    parameters = list(parameters)

    global_norm_squared = 0
    for param in parameters:
        if param.grad is None:
            continue
        global_norm_squared += torch.sum(param.grad.data ** 2)
    
    global_norm = global_norm_squared ** 0.5
    if global_norm  >= max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad.data = param.grad.data * max_l2_norm / (global_norm + eps)
    
    return global_norm.item()