from cs336_basics import train_bpe
import pickle
import time
import multiprocessing

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, cpu=8.0, timeout=60*60)
def train_bpe_tinystories():

    input_file = DATA_PATH / "raw_data" / "TinyStoriesV2-GPT4-train.txt"
    vocab_file = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-vocab.pkl"
    merges_file = DATA_PATH / "tokenizers" / "tokenizer-tinystories-10000-merges.pkl"

    vocab_file.parent.mkdir(exist_ok=True, parents=True)

    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    num_processes = min(multiprocessing.cpu_count(), 8)

    start = time.perf_counter()
    vocab, merges = train_bpe.train_bpe(input_file, vocab_size, special_tokens, num_processes)
    training_time = time.perf_counter()-start

    longest_token_id, longest_token_bytes = max(vocab.items(), key = lambda kv: len(kv[1]))
    longest_token_str = longest_token_bytes.decode('utf-8')

    print(f"Time: {training_time:.2f} s")
    print(f"Longest Token: {longest_token_str}")

    with open(vocab_file, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)


@app.local_entrypoint()
def modal_main() -> None:
    print("Training BPE on TinyStories on Modal")
    train_bpe_tinystories.remote()


if __name__ == "__main__":
    print("Training BPE on TinyStories locally")
    train_bpe_tinystories.local()