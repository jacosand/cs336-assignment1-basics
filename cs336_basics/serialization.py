import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration,
        }, out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device | None = None,
) -> int:
    
    chk = torch.load(src, map_location = device, weights_only=True)
    model.load_state_dict(chk['model'])
    if optimizer is not None:
        optimizer.load_state_dict(chk['optimizer'])

    return chk['iteration']