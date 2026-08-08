import math
from cs336_basics import train_utils
from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets

BATCH_SIZES_AND_TUNED_MAX_LEARNING_RATES = [
    (1, 4e-4),
    (8, 1e-3),
    (64, 1e-3),
    (128, 4e-3),
    (1024, 4e-3),
]

@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=2*60*60)
def run_batch_size_experiment(
    batch_size_and_max_learning_rate: tuple[int, float],
) -> None:
    batch_size, max_learning_rate = batch_size_and_max_learning_rate

    num_iterations = 256_000 // batch_size
    cosine_cycle_iters = num_iterations

    num_valid_batches = max(1, math.ceil((128 / batch_size) * 10))

    log_every = num_iterations // 100
    valid_every = num_iterations // 10
    save_every = num_iterations
    warmup_iters = num_iterations // 50

    args = train_utils.parse_args([
        "--train-data",
        "data/tokens/tokens-tinystories-train.bin",
        "--valid-data",
        "data/tokens/tokens-tinystories-valid.bin",
        "--batch-size",
        str(batch_size),
        "--max-learning-rate",
        str(max_learning_rate),
        "--min-learning-rate",
        str(0.1*max_learning_rate),
        "--num-iterations",
        str(num_iterations),
        "--cosine-cycle-iters",
        str(cosine_cycle_iters),
        "--log-every",
        str(log_every),
        "--valid-every",
        str(valid_every),
        "--save-every",
        str(save_every),
        "--warmup-iters",
        str(warmup_iters),
        "--num-valid-batches",
        str(num_valid_batches),
    ])

    train_utils.train(args)


@app.local_entrypoint()
def modal_main() -> None:
    list(run_batch_size_experiment.map(BATCH_SIZES_AND_TUNED_MAX_LEARNING_RATES))