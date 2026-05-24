import os
from typing import BinaryIO
import regex as re
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def find_merge_pair(pretoken_freq, vocab):
    pair_freq = {}
    for pretoken, freq in pretoken_freq.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + freq

    return max(pair_freq, key = lambda pair: (pair_freq[pair], tuple(vocab[p] for p in pair)))


def merge(pretoken_freq, pair, idx):
    merged_pretoken_freq = {}

    for pretoken, freq in pretoken_freq.items():
        new_pretoken = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == pair:
                new_pretoken.append(idx)
                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1
        new_pretoken = tuple(new_pretoken)
        merged_pretoken_freq[new_pretoken] = merged_pretoken_freq.get(new_pretoken, 0) + freq

    return merged_pretoken_freq


def compute_pretoken_counts(chunk):

    input_path, start, end, special_tokens = chunk

    pretoken_freq = {}

    split_pattern = "|".join(re.escape(special_token) for special_token in special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        file_chunk = f.read(end - start).decode("utf-8", errors="ignore")

    docs = re.split(split_pattern, file_chunk)

    for doc in docs:
        for pretoken in re.finditer(PAT, doc):
            pretoken_tuple = tuple(pretoken.group().encode('utf-8'))
            pretoken_freq[pretoken_tuple] = pretoken_freq.get(pretoken_tuple, 0) + 1
    
    return pretoken_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    n_special_tokens = len(special_tokens)
    n_single_bytes = 256

    vocab = {}
    merges = []
    for i in range(n_single_bytes):
        vocab[i] = bytes([i])

    pretoken_freq = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunks.append((input_path, start, end, special_tokens))
    
    with multiprocessing.Pool(num_processes) as pool:
        pretoken_freqs = pool.map(compute_pretoken_counts, chunks)
        
    for pt_freq in pretoken_freqs:
        for k, v in pt_freq.items():
            pretoken_freq[k] = pretoken_freq.get(k, 0) + v
    
    for i in range(n_single_bytes, vocab_size - n_special_tokens):
        int1, int2 = find_merge_pair(pretoken_freq, vocab)
        byte1 = vocab[int1]
        byte2 = vocab[int2]
        vocab[i] = byte1 + byte2
        merges.append((byte1, byte2))
        pretoken_freq = merge(pretoken_freq, (int1, int2), i)

    for i in range(n_special_tokens):
        vocab[vocab_size - n_special_tokens + i] = special_tokens[i].encode('utf-8')

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 1000, ["<|endoftext|>"])
    print(vocab)
    print(merges)
