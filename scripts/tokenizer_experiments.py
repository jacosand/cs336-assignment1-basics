from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import itertools
import numpy as np
import time

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image


def sample_documents(
    in_file: str | Path,
    delimiter: str,
    n: int = 10,
) -> list[str]:
    
    documents = []
    current_document = ""
    with open(in_file, "r") as f:
        for line in f:
            if delimiter in line:
                current_document += line.split(delimiter)[0]
                documents.append(current_document)
                current_document = ""
            else:
                current_document += line
            if len(documents) == n:
                break
    
    return documents


def compute_tokenizer_stats(
    tokenizer: Tokenizer,
    documents: list[str]
    ) -> tuple[float, float]:

    tokenizer_time = 0
    n_bytes = 0
    n_tokens = 0
    for doc in documents:
        start = time.perf_counter()
        tokens = tokenizer.encode(doc)
        tokenizer_time += time.perf_counter()-start
        n_bytes += len(doc.encode('utf-8'))
        n_tokens += len(tokens)

    compression_ratio = n_bytes / n_tokens
    throughput = n_bytes / tokenizer_time

    return compression_ratio, throughput


def tokenize_to_bin(
    tokenizer: Tokenizer,
    in_file: str | Path,
    out_file: str | Path,
    chunk_size: int = 1_000_000,
):
    
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r") as f, open(out_file, "wb") as g:
        token_iterable = tokenizer.encode_iterable(f)
        while True:
            chunk = np.fromiter(itertools.islice(token_iterable, chunk_size), dtype=np.uint16)
            if chunk.size == 0:
                break
            g.write(chunk.tobytes())


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60*60)
def tokenizer_experiments():

    tokenizer_tinystories = Tokenizer.from_files(
        vocab_filepath = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-vocab.pkl",
        merges_filepath = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    tokenizer_owt = Tokenizer.from_files(
        vocab_filepath = DATA_PATH / "tokenizers" / "tokenizer-owt-32000-vocab.pkl",
        merges_filepath = DATA_PATH / "tokenizers" / "tokenizer-owt-32000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    docs_tinystories = sample_documents(DATA_PATH / "raw_data" / "TinyStoriesV2-GPT4-train.txt", "<|endoftext|>")
    docs_owt = sample_documents(DATA_PATH / "raw_data" / "owt_train.txt", "<|endoftext|>")
    
    # Calculate statistics
    compression_ratio_tinystories, throughput_tinystories = compute_tokenizer_stats(tokenizer_tinystories, docs_tinystories)
    compression_ratio_owt, throughput_owt = compute_tokenizer_stats(tokenizer_owt, docs_owt)

    # Calculate cross statistics
    compression_ratio_cross, throughput_cross = compute_tokenizer_stats(tokenizer_tinystories, docs_owt)

    print(f"Compression Ratio TinyStories: {compression_ratio_tinystories:.3f} bytes/token")
    print(f"Compression Ratio OpenWebText: {compression_ratio_owt:.3f} bytes/token")

    print(f"Compression Ratio Crossed: {compression_ratio_cross:.3f} bytes/token")

    print(f"Throughput TinyStories: {throughput_tinystories:.3f} bytes/second")
    print(f"Throughput OpenWebText: {throughput_owt:.3f} bytes/second")

    # Tokenize TinyStories
    tokenize_to_bin(tokenizer_tinystories, DATA_PATH / "raw_data" / "TinyStoriesV2-GPT4-valid.txt", DATA_PATH / "tokens" / "tokens-tinystories-valid.bin")
    tokenize_to_bin(tokenizer_tinystories, DATA_PATH / "raw_data" / "TinyStoriesV2-GPT4-train.txt", DATA_PATH / "tokens" / "tokens-tinystories-train.bin")

    # Tokenize OpenWebText
    tokenize_to_bin(tokenizer_owt, DATA_PATH / "raw_data" / "owt_valid.txt", DATA_PATH / "tokens" / "tokens-owt-valid.bin")
    tokenize_to_bin(tokenizer_owt, DATA_PATH / "raw_data" / "owt_train.txt", DATA_PATH / "tokens" / "tokens-owt-train.bin")


@app.local_entrypoint()
def modal_main() -> None:
    print("Running tokenizer on Modal")
    tokenizer_experiments.remote()


if __name__ == "__main__":
    print("Running tokenizer locally")
    tokenizer_experiments.local()