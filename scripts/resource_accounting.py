MODEL_CONFIGS = {
    "gpt-2-small": {
        'vocab_size': 50_257,
        'context_length': 1_024,
        'num_layers': 12,
        'd_model': 768,
        'num_heads': 12,
    },
    "gpt-2-medium": {
        'vocab_size': 50_257,
        'context_length': 1_024,
        'num_layers': 24,
        'd_model': 1_024,
        'num_heads': 16,
    },
    "gpt-2-large": {
        'vocab_size': 50_257,
        'context_length': 1_024,
        'num_layers': 36,
        'd_model': 1_280,
        'num_heads': 20,
    },
    "gpt-2-xl": {
        'vocab_size': 50_257,
        'context_length': 1_024,
        'num_layers': 48,
        'd_model': 1_600,
        'num_heads': 25,
    },
    "gpt-2-xl-longcontext": {
        'vocab_size': 50_257,
        'context_length': 16_384,
        'num_layers': 48,
        'd_model': 1_600,
        'num_heads': 25,
    }
}


def print_transformer_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int | None = None,
):
    if d_ff is None:
        d_ff = int(round((8./3 * d_model / 64), 0) * 64)

    trainable_parameters = 2 * context_length * d_model * (vocab_size + num_layers * (4 * d_model + 2 * context_length + 3 * d_ff))

    lm_head_flops = 2 * context_length * d_model * vocab_size
    attention_mechanism_flops = 4 * num_layers * (context_length ** 2) * d_model
    attention_projection_flops =  8 * num_layers * context_length * (d_model ** 2)
    feed_forward_flops = 6 * num_layers * context_length * d_model * d_ff

    total_flops = lm_head_flops + attention_mechanism_flops + attention_projection_flops + feed_forward_flops

    print(f"vocab_size={vocab_size}, context_length={context_length}, num_layers={num_layers}, d_model={d_model}, d_ff={d_ff}")
    print()
    print(f"Trainable Parameters: {trainable_parameters}")
    print()
    print(f"LM Head: {lm_head_flops} FLOPs ({100 * lm_head_flops / total_flops:.1f}%)")
    print(f"Attention Mechanism: {attention_mechanism_flops} FLOPs ({100 * attention_mechanism_flops / total_flops:.1f}%)")
    print(f"Attention Projections: {attention_projection_flops} FLOPs ({100 * attention_projection_flops / total_flops:.1f}%)")
    print(f"Position-wise Feed Forward: {feed_forward_flops} FLOPs ({100 * feed_forward_flops / total_flops:.1f}%)")
    print()
    print(f"Total: {total_flops} FLOPs")


def print_adamw_accounting(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int | None = None,
):
    if d_ff is None:
        d_ff = 8./3 * d_model

    n_parameters = d_model * (2 * vocab_size + 1) + num_layers * d_model * (4 * d_model + 3 * d_ff + 2)
    n_activations_per_example = num_layers * context_length * (8 * d_model + 4 * d_ff + 2 * num_heads * context_length) + context_length * (d_model + 2 * vocab_size)
    n_gradients = n_parameters
    n_optimizer_state_parameters = 2 * n_parameters

    # memory requirement with float32 tensors is a * batch_size + b
    a = 4 * n_activations_per_example
    b = 4 * (n_parameters + n_gradients + n_optimizer_state_parameters)

    print(f"vocab_size={vocab_size}, context_length={context_length}, num_layers={num_layers}, d_model={d_model}, num_heads={num_heads}")
    print()
    print(f"Memory in bytes: {a:.0f}*batch_size + {b:.0f}")


def main():
    for model, config in MODEL_CONFIGS.items():
        print(f"Model: {model}")
        print_transformer_accounting(**config)
        print()
        print()
    
    for model, config in MODEL_CONFIGS.items():
        print(f"Model: {model}")
        print_adamw_accounting(**config)
        print()
        print()


if __name__ == "__main__":
    main()