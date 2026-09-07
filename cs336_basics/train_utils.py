import os
import argparse
import time
import numpy as np
import torch
import wandb
from pathlib import Path
from cs336_basics import data, model, optimizer, nn_utils, serialization
from cs336_basics.modal_utils import DATA_PATH


def parse_args(arglist: tuple[str, ...] | list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Train a transformer language model")

    # Input/output parameters
    parser.add_argument("--train-data", type=str, required=True, help="Path to the binary np.uint16 training data")
    parser.add_argument("--valid-data", type=str, required=True, help="Path to the binary np.uint16 validation data")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to .pt checkpoint to resume from")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--position-encoding", type=str, choices=["rope", "none"], default="rope")
    parser.add_argument("--rope-theta", type=float, default=10_000)
    parser.add_argument("--layer-norm", type=str, choices=["pre", "post", "none"], default="pre")
    parser.add_argument("--activation", type=str, choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--weight-tying", type=str, choices=["yes", "no"], default="no")

    # Optimizer arguments
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-learning-rate", type=float, default=4e-3)
    parser.add_argument("--min-learning-rate", type=float, default=4e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10_000)
    parser.add_argument("--max-l2-norm", type=float, default=1.0)
    parser.add_argument("--muon", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--muon-max-learning-rate", type=float, default=2e-2)
    parser.add_argument("--muon-min-learning-rate", type=float, default=2e-3)
    parser.add_argument("--muon-beta", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.01)

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=10_000)
    parser.add_argument("--num-valid-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--valid-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=336)

    return parser.parse_args(args = arglist)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> None:
    run = wandb.init(
        entity = "jacosand-personal",
        project = "cs336-assignment1",
        config=vars(args)
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")

    seed_everything(args.seed)

    save_dir = DATA_PATH / "checkpoints" / run.id
    os.makedirs(save_dir, exist_ok=True)

    transformer_lm = model.TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        position_encoding = args.position_encoding,
        rope_theta = args.rope_theta,
        layer_norm = args.layer_norm,
        activation = args.activation,
        weight_tying = args.weight_tying,
        device = device,
    )

    if args.muon == 'yes': 
        adam_params = []
        muon_params = []
        qkv_params = []

        for name, p in transformer_lm.named_parameters():
            if 'qkv_proj' in name:
                qkv_params.append(p)
            elif 'ln' in name or 'token_embeddings' in name or 'lm_head' in name:
                adam_params.append(p)
            else:
                muon_params.append(p)

        adamw = optimizer.AdamW(
            adam_params,
            lr = args.max_learning_rate,
            betas = (args.beta1, args.beta2),
            weight_decay = args.weight_decay,
        )

        muon = optimizer.Muon(
            [
                {"params": muon_params, "split_qkv": False},
                {"params": qkv_params, "split_qkv": True},
            ],
            lr = args.muon_max_learning_rate,
            mu = args.muon_beta,
            weight_decay = args.muon_weight_decay,
        )

        optimizers = [adamw, muon]

    else:
        adamw = optimizer.AdamW(
            transformer_lm.parameters(),
            lr = args.max_learning_rate,
            betas = (args.beta1, args.beta2),
            weight_decay = args.weight_decay,
        )

        optimizers = [adamw]

    if args.resume_from is not None:
        resume_from = Path(args.resume_from)
        start_step = serialization.load_checkpoint(resume_from, transformer_lm, optimizers) + 1
    else:
        start_step = 1

    train_data = Path(args.train_data)
    valid_data = Path(args.valid_data)
    train_tokens = np.memmap(train_data, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(valid_data, dtype=np.uint16, mode="r")

    if device.type == "cuda":
        transformer_lm.compile()

    transformer_lm.train()

    running_steps = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    running_time = 0.0
    running_tokens_processed = 0

    best_valid_loss = float('inf')

    for step in range(start_step, args.num_iterations + 1):

        t0 = time.perf_counter()
        adamw_lr = optimizer.get_lr_cosine(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for group in adamw.param_groups:
            group["lr"] = adamw_lr
        
        if args.muon == 'yes':
            muon_lr = optimizer.get_lr_cosine(step, args.muon_max_learning_rate, args.muon_min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
            for group in muon.param_groups:
                group["lr"] = muon_lr

        x, y = data.get_batch(train_tokens, batch_size=args.batch_size, context_length=args.context_length, device=device)
        for opt in optimizers:
            opt.zero_grad()
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = transformer_lm(x)
                loss = nn_utils.cross_entropy(logits, y)
        else:
            logits = transformer_lm(x)
            loss = nn_utils.cross_entropy(logits, y)
        loss.backward()
        grad_norm = nn_utils.clip_gradients(transformer_lm.parameters(), args.max_l2_norm)
        for opt in optimizers:
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        dt = time.perf_counter() - t0

        tokens_processed = args.batch_size * args.context_length

        running_steps += 1
        running_loss += loss.item()
        running_time += dt
        running_tokens_processed += tokens_processed
        running_grad_norm += grad_norm

        metrics = {}

        if step % args.log_every == 0 or step == args.num_iterations:

            metrics.update({
                "train/loss": running_loss / running_steps,
                "train/learning_rate": adamw_lr,
                "train/grad_norm": running_grad_norm / running_steps,
                "train/tokens_processed": step * tokens_processed,
                "performance/step_time_sec": running_time / running_steps,
                "performance/tokens_per_sec": running_tokens_processed / running_time,
            })

            if args.muon == "yes":
                metrics.update({
                    "train/muon_learning_rate": muon_lr,
                })

            if args.muon == "yes":
                lr_string = (
                    f"adam_lr {adamw_lr:.3e} | "
                    f"muon_lr {muon_lr:.3e} | "
                )
            else:
                lr_string = f"adam_lr {adamw_lr:.3e} | "

            print(
                f"step {step:6d} | "
                f"loss {running_loss / running_steps:.6f} | "
                f"dt {running_time / running_steps * 1000:.2f}ms | "
                f"tok/sec {running_tokens_processed / running_time:.2f} | "
                f"{lr_string}"
                f"grad_norm {running_grad_norm / running_steps:.2f}"
            )

            running_steps = 0
            running_loss = 0.0
            running_grad_norm = 0.0
            running_time = 0.0
            running_tokens_processed = 0

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

                metrics["valid/loss"] = valid_loss

                print(
                    f"step {step:4d} | "
                    f"validation loss: {valid_loss:.6f}"
                )

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    serialization.save_checkpoint(transformer_lm, optimizers, step, save_dir / "checkpoint_best.pt")
            
            transformer_lm.train()
        
        if metrics:
            run.log(metrics, step=step)

        if step % args.save_every == 0 or step == args.num_iterations:
            serialization.save_checkpoint(transformer_lm, optimizers, step, save_dir / f"checkpoint_step_{step:06d}.pt")
            serialization.save_checkpoint(transformer_lm, optimizers, step, save_dir / "checkpoint_latest.pt")

    run.finish()