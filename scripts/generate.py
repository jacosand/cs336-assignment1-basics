import sys
import argparse
from cs336_basics import tokenizer, model, serialization
from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image, secrets
import wandb
import torch

DEFAULT_WANDB_RUN = "vwfd7ttv" # fiery-microwave-17
DEFAULT_CHECKPOINT_FILE = "checkpoint_best.pt"


def parse_args(arglist: tuple[str, ...] | list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Generate text from a transformer language model")

    # Prompt parameters
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use to start generating text")

    # Trained model parameters
    parser.add_argument("--wandb-id", type=str, default=DEFAULT_WANDB_RUN, help="WandB run ID to load")
    parser.add_argument("--checkpoint-file", type=str, default=DEFAULT_CHECKPOINT_FILE, help=".pt checkpoint to load")

    # Text generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)

    return parser.parse_args(args = arglist)


def load_wandb_run(
    wandb_id: str,
    checkpoint_file: str,
    device = None,
) -> dict[str, tokenizer.Tokenizer | model.TransformerLM]:
    api = wandb.Api()
    run = api.run(f"jacosand-personal/cs336-assignment1/{wandb_id}")

    transformer_run = model.TransformerLM(
        vocab_size = run.config['vocab_size'],
        context_length = run.config['context_length'],
        d_model = run.config['d_model'],
        num_layers = run.config['num_layers'],
        num_heads = run.config['num_heads'],
        d_ff = run.config['d_ff'],
        rope_theta = run.config['rope_theta'],
        device = device,
    )

    serialization.load_checkpoint(DATA_PATH / "checkpoints" / run.id / checkpoint_file, transformer_run, optimizer=None, device=device)

    if 'tinystories' in run.config['train_data'] and 'tinystories' in run.config['valid_data']:
        tokenizer_run = tokenizer.Tokenizer.from_files(
            vocab_filepath = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-vocab.pkl",
            merges_filepath = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-merges.pkl",
            special_tokens = ["<|endoftext|>"],
        )
    elif 'owt' in run.config['train_data'] and 'owt' in run.config['valid_data']:
        tokenizer_run = tokenizer.Tokenizer.from_files(
            vocab_filepath = DATA_PATH / "tokenizers" / "tokenizer-owt-32000-vocab.pkl",
            merges_filepath = DATA_PATH / "tokenizers" / "tokenizer-owt-32000-merges.pkl",
            special_tokens = ["<|endoftext|>"],
        )
    else:
        raise ValueError(f"No tokenizer available corresponding to training data {run.config['train_data']} and validation data {run.config['valid_data']}")

    return {
        "model": transformer_run,
        "tokenizer": tokenizer_run,
    }


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=10*60)
def generate(*arglist: str) -> str:

    args = parse_args(arglist)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")

    wandb_run = load_wandb_run(args.wandb_id, args.checkpoint_file, device=device)

    token_ids = wandb_run['tokenizer'].encode(args.prompt)
    tokens = torch.tensor(token_ids, device=device)
    eot_token = wandb_run['tokenizer'].encode("<|endoftext|>")[0]

    wandb_run['model'].eval()
    
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = wandb_run['model'].generate(
                tokens,
                args.max_new_tokens,
                eot_token=eot_token,
                temperature=args.temperature,
                top_p=args.top_p
            )
    else:
        tokens = wandb_run['model'].generate(
            tokens,
            args.max_new_tokens,
            eot_token=eot_token,
            temperature=args.temperature,
            top_p=args.top_p
        )

    response = wandb_run['tokenizer'].decode(tokens.tolist())

    return response


@app.local_entrypoint()
def modal_main(*arglist: str) -> None:
    print("Generating text on Modal")
    response = generate.remote(*arglist)
    print(response)


if __name__ == "__main__":
    print("Generating text locally")
    response = generate.local(*sys.argv[1:])
    print(response)