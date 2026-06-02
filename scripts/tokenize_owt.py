import numpy as np
from cs336_basics.tokenizer import Tokenizer

def main():

    tokenizer = Tokenizer.from_files(
        vocab_filepath = "artifacts/tokenizer-owt-32000-vocab.pkl",
        merges_filepath = "artifacts/tokenizer-owt-32000-merges.pkl",
        special_tokens = ["<|endoftext|>"],
    )

    with open("data/owt_valid.txt", "r") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save("artifacts/tokens-owt-valid.npy", token_ids)

    with open("data/owt_train.txt", "r") as f:
        token_ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)

    np.save("artifacts/tokens-owt-train.npy", token_ids)

if __name__ == "__main__":
    main()