import torch
from torch import nn
from cs336_basics import optimizer

def optimize_mean_squared(
    lr: float = 1,
    n_iter: int = 100,
):

    weights = nn.Parameter(5 * torch.randn((10, 10)))
    opt = optimizer.SGD([weights], lr=lr)
    for t in range(n_iter):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())

        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


def main():
    for lr in [1e0, 1e1, 1e2, 1e3]:
        print(f"Learning Rate: {lr}")
        print()
        optimize_mean_squared(lr=lr, n_iter=10)
        print()
        print()
    
if __name__ == "__main__":
    main()