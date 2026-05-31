import numpy as np
from cs336_basics.tokenizer import Tokenizer

def main():

    tokenizer = Tokenizer.from_files(
        vocab_filepath = "artifacts/tokenizer-tinystories-10000-vocab.pkl",
        merges_filepath = "artifacts/tokenizer-tinystories-10000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    with open("data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save("artifacts/tokens-tinystories-valid.npy", token_ids)

    with open("data/TinyStoriesV2-GPT4-train.txt", "r") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save("artifacts/tokens-tinystories-train.npy", token_ids)

if __name__ == "__main__":
    main()