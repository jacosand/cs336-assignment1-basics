import os
from typing import IO, BinaryIO
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    
    torch.save(
        {
            'model': model.state_dict(),
            'optimizers': [opt.state_dict() for opt in optimizers],
            'iteration': iteration,
        }, out
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer] | None,
    device: str | torch.device | None = None,
) -> int:
    
    chk = torch.load(src, map_location = device, weights_only=True)
    model.load_state_dict(chk['model'])
    if optimizers is not None:
        for opt, state_dict in zip(optimizers, chk['optimizers']):
            opt.load_state_dict(state_dict)

    return chk['iteration']