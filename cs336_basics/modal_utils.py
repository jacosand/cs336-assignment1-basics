from pathlib import Path, PurePosixPath

import modal

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"cs336-basics")
user_volume = modal.Volume.from_name(f"cs336-basics", create_if_missing=True, version=2)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim().apt_install("wget", "gzip").uv_sync()
    image = image.add_local_python_source("cs336_basics")
    image = image.add_local_file("AGENTS.md", "/root/AGENTS.md")
    image = image.add_local_file("CLAUDE.md", "/root/CLAUDE.md")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image


VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets() -> list[modal.Secret]:
    secrets = [modal.Secret.from_name("wandb-secret")]
    return secrets