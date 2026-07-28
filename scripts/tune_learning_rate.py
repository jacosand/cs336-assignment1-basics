from scripts import train_utils
from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets

LEARNING_RATES = [3e-4, 6e-4, 9e-4, 1.2e-3]


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=2*60*60)
def run_lr_experiment(
    lr: float,
) -> None:
    args = train_utils.parse_args([
        "--train-data",
        "data/tokens/tokens-tinystories-train.bin",
        "--valid-data",
        "data/tokens/tokens-tinystories-valid.bin",
        "--max-learning-rate",
        str(lr),
        "--min-learning-rate",
        str(0.1*lr),
    ])

    train_utils.train(args)


@app.local_entrypoint()
def modal_main() -> None:
    list(run_lr_experiment.map(LEARNING_RATES))
