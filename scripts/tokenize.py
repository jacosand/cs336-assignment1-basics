from cs336_basics.tokenizer import Tokenizer
import itertools
import numpy as np


def tokenize_to_bin(
    tokenizer: Tokenizer,
    in_file: str,
    out_file: str,
    chunk_size: int = 1_000_000,
):
    with open(in_file, "r") as f, open(out_file, "wb") as g:
        token_iterable = tokenizer.encode_iterable(f)
        while True:
            chunk = np.fromiter(itertools.islice(token_iterable, chunk_size), dtype=np.uint16)
            if chunk.size == 0:
                break
            g.write(chunk.tobytes())


def main():

    tokenizer_tinystories = Tokenizer.from_files(
        vocab_filepath = "artifacts/tokenizer-tinystories-10000-vocab.pkl",
        merges_filepath = "artifacts/tokenizer-tinystories-10000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    tokenizer_owt = Tokenizer.from_files(
        vocab_filepath = "artifacts/tokenizer-owt-32000-vocab.pkl",
        merges_filepath = "artifacts/tokenizer-owt-32000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    # Tokenize TinyStories
    tokenize_to_bin(tokenizer_tinystories, "data/TinyStoriesV2-GPT4-valid.txt", "artifacts/tokens-tinystories-valid.bin")
    tokenize_to_bin(tokenizer_tinystories, "data/TinyStoriesV2-GPT4-train.txt", "artifacts/tokens-tinystories-train.bin")

    # Tokenize OpenWebText
    tokenize_to_bin(tokenizer_owt, "data/owt_valid.txt", "artifacts/tokens-owt-valid.bin")
    tokenize_to_bin(tokenizer_owt, "data/owt_train.txt", "artifacts/tokens-owt-train.bin")


if __name__ == "__main__":
    main()