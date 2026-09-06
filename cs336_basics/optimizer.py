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


def get_lr_cosine(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif it < cosine_cycle_iters:
        cosine_frac = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate-min_learning_rate) * (1 + math.cos(cosine_frac * math.pi))
    else:
        return min_learning_rate


def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):

    def __init__(self,
        params: Iterable[torch.Tensor],
        lr: float=1e-3,
        mu: float=0.9,
        weight_decay: float=1e-2,
        eps: float=1e-7,
        split_qkv = False,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu <= 0 or mu >= 1:
            raise ValueError(f"Invalid first moment update hyperparameter: {mu}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay rate: {weight_decay}") 
        if eps <= 0:
            raise ValueError(f"Invalid stability parameter: {eps}")
        
        defaults = {
            "lr": lr,
            "mu": mu,
            "weight_decay": weight_decay,
            "eps": eps,
            "split_qkv": split_qkv,
        }

        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            mu = group["mu"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            split_qkv = group["split_qkv"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                assert p.ndim == 2

                state = self.state[p]
                if "b" not in state:
                    state["b"] = torch.zeros_like(p)

                b = state["b"]

                grad = p.grad.data

                b = mu * b + grad
                o = mu * b + grad

                if split_qkv:
                    q, k, v = torch.chunk(o, 3, dim=0)

                    q = newtonschulz5(q, eps=eps)
                    q *= max(1, q.size(0)/q.size(1)) ** 0.5

                    k = newtonschulz5(k, eps=eps)
                    k *= max(1, k.size(0)/k.size(1)) ** 0.5

                    v = newtonschulz5(v, eps=eps)
                    v *= max(1, v.size(0)/v.size(1)) ** 0.5

                    o = torch.cat([q, k, v])
                
                else:
                    o = newtonschulz5(o, eps=eps)
                    o *= max(1, o.size(0)/o.size(1)) ** 0.5

                p.data -= lr * weight_decay * p.data
                p.data -= lr * o

                state["b"] = b

        return loss
