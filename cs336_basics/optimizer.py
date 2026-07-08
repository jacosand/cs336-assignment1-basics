from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.Tensor], lr: float=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self,
        params: Iterable[torch.Tensor],
        lr: float=1e-3,
        betas: tuple[float,float]=(0.9, 0.999),
        weight_decay: float=1e-2,
        eps: float=1e-8,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] <= 0 or betas[0] >= 1:
            raise ValueError(f"Invalid first moment update hyperparameter: {betas[0]}")
        if betas[1] <= 0 or betas[1] >= 1:
            raise ValueError(f"Invalid second moment update hyperparameter: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay rate: {weight_decay}") 
        if eps <= 0:
            raise ValueError(f"Invalid stability parameter: {eps}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
        }

        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                grad = p.grad.data

                p.data -= lr * weight_decay * p.data

                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                alpha_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

                p.data -= alpha_t * m / (v ** 0.5 + eps)

                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss