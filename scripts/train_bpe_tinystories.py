from cs336_basics import train_bpe
import pickle
import time

def main():

    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    output_file = "artifacts/tinystories-tokenizer-10000.pkl"

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    start = time.perf_counter()
    vocab, merges = train_bpe.train_bpe(input_file, vocab_size, special_tokens)
    training_time = time.perf_counter()-start

    print(
        f"Training on TinyStories with a maximum vocabulary size of "
        f"{vocab_size} took {training_time:2f} seconds."
    )

    with open(output_file, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


if __name__ == "__main__":
    main()