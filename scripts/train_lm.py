import os
import argparse
import time
import numpy as np
import torch
from cs336_basics import data, model, optimizer, nn_utils, serialization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train a transformer language model")

    # Input/output parameters
    parser.add_argument("--train-data", type=str, required=True, help="Path to the binary np.uint16 training data")
    parser.add_argument("--valid-data", type=str, required=True, help="Path to the binary np.uint16 validation data")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save model checkpoint files")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to .pt checkpoint to resume from")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=int, default=10_000)

    # Optimizer arguments
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--num-valid-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--valid-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
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

    os.makedirs(args.save_dir, exist_ok=True)

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
        lr = args.max_learning_rate,
        betas = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
    )

    if args.resume_from is not None:
        start_step = serialization.load_checkpoint(args.resume_from, transformer_lm, opt) + 1
    else:
        start_step = 1

    train_tokens = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(args.valid_data, dtype=np.uint16, mode="r")
    
    transformer_lm.train()

    for step in range(start_step, args.num_iterations + 1):

        t0 = time.perf_counter()
        lr = optimizer.get_lr_cosine(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = data.get_batch(train_tokens, batch_size=args.batch_size, context_length=args.context_length, device=device)
        opt.zero_grad()
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = transformer_lm(x)
                loss = nn_utils.cross_entropy(logits, y)
        else:
            logits = transformer_lm(x)
            loss = nn_utils.cross_entropy(logits, y)
        loss.backward()
        nn_utils.clip_gradients(transformer_lm.parameters(), args.max_l2_norm)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        tokens_processed = args.batch_size * args.context_length
        tokens_per_sec = tokens_processed / dt

        if step % args.log_every == 0:
            print(f"step {step:6d} | loss {loss.item():.6f} | dt {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f} | lr {lr:.3e}")
        
        if step % args.valid_every == 0 or step == args.num_iterations:
            transformer_lm.eval()

            with torch.no_grad():
                valid_loss = 0.0
                for _ in range(args.num_valid_batches):
                    x, y = data.get_batch(valid_tokens, batch_size=args.batch_size, context_length=args.context_length, device=device)
                    if device.type == "cuda" and torch.cuda.is_bf16_supported():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = transformer_lm(x)
                            batch_loss = nn_utils.cross_entropy(logits, y)
                    else:
                        logits = transformer_lm(x)
                        batch_loss = nn_utils.cross_entropy(logits, y)
                    valid_loss += batch_loss.item()
                valid_loss /= args.num_valid_batches
                print(f"step {step:4d} | validation loss: {valid_loss:.6f}")
            
            transformer_lm.train()
        
        if step % args.save_every == 0 or step == args.num_iterations:
            serialization.save_checkpoint(transformer_lm, opt, step, f"{args.save_dir}/checkpoint_step_{step:06d}.pt")

if __name__ == "__main__":
    main()