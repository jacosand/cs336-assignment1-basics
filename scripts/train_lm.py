import sys
from cs336_basics.modal_utils import VOLUME_MOUNTS, app, build_image, secrets
from cs336_basics import train_utils


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def train_lm(*arglist: str) -> None:
    args = train_utils.parse_args(arglist)
    train_utils.train(args)


@app.local_entrypoint()
def modal_main(*arglist: str) -> None:
    print("Training LM on Modal")
    train_lm.remote(*arglist)


if __name__ == "__main__":
    print("Training LM locally")
    train_lm.local(*sys.argv[1:])