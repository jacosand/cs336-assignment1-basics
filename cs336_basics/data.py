import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Int


def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    
    start_indices = np.random.randint(low = 0, high = len(x) - context_length, size = batch_size)
    range_indices = np.arange(context_length)
    idx = start_indices[:, None] + range_indices[None, :]
    in_tokens = torch.as_tensor(x[idx], dtype=torch.int32, device=device)
    out_tokens = torch.as_tensor(x[idx + 1], dtype=torch.int32, device=device)

    return in_tokens, out_tokens