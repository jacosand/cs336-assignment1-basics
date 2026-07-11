import argparse
import numpy as np
import torch
from cs336_basics import data, model, optimizer, nn_utils, serialization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train a transformer language model")

    # Input/output parameters
    parser.add_argument("--train-data", type=str, required=True, help="Path to the binary np.uint16 training data")
    parser.add_argument("--valid-data", type=str, required=True, help="Path to the binary np.uint16 validation data")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save model checkpoint files")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=int, default=10_000)

    # Optimizer arguments
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--valid-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")

    seed_everything(args.seed)

    transformer_lm = model.TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = device,
    )

    opt = optimizer.AdamW(
        transformer_lm.parameters(),
        lr = args.lr,
        betas = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
    )

    train_tokens = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(args.valid_data, dtype=np.uint16, mode="r")
    
    transformer_lm.train()

    for i in range(1, args.num_iterations + 1):
        
        opt.zero_grad()
        x, y = data.get_batch(train_tokens, batch_size=args.batch_size, context_length=args.context_length, device=device)
        preds = transformer_lm(x)
        loss = nn_utils.cross_entropy(preds, y)
        loss.backward()
        opt.step()

        if i % args.log_every == 0:
            print(f"iteration = {i}, training loss = {loss.item():.4f}")
        
        if i % args.valid_every == 0:
            transformer_lm.eval()

            with torch.no_grad():
                x, y = data.get_batch(valid_tokens, batch_size=args.batch_size, context_length=args.context_length, device=device)
                preds = transformer_lm(x)
                loss = nn_utils.cross_entropy(preds, y)
                print(f"iteration = {i}, validation loss = {loss.item():.4f}")
            
            transformer_lm.train()
        
        if i % args.save_every == 0:
            serialization.save_checkpoint(transformer_lm, opt, i, f"{args.save_path}/model_iter{i}.bin")

if __name__ == "__main__":
    main()