import os
from typing import BinaryIO
from collections import defaultdict
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


def get_pair_stats(
        pretoken_tuples: dict[int, tuple],
        pretoken_freq: dict[int, int],
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set]]:

    pair_freq = defaultdict(int)
    pair_locations = defaultdict(set)
    
    for pretoken_id, freq in pretoken_freq.items():
        pretoken = pretoken_tuples[pretoken_id]
        for pair in zip(pretoken, pretoken[1:]):
            pair_freq[pair] += freq
            pair_locations[pair].add(pretoken_id)
    
    return pair_freq, pair_locations


def merge_pretoken(
    pretoken: tuple,
    merge_pair: tuple[int, int],
    idx: int,
) -> tuple:

    new_pretoken = []

    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == merge_pair:
            new_pretoken.append(idx)
            i += 2
        else:
            new_pretoken.append(pretoken[i])
            i += 1
    
    return tuple(new_pretoken)


def merge(
        pair_freq: dict[tuple[int, int], int],
        pair_locations: dict[tuple[int, int], set],
        pretoken_tuples: dict[int, tuple],
        pretoken_freq: dict[int, int],
        merge_pair: tuple[int, int],
        idx: int,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set], dict[int, tuple]]:

    for pretoken_id in pair_locations[merge_pair]:
        
        pretoken = pretoken_tuples[pretoken_id]

        new_pretoken  = merge_pretoken(pretoken, merge_pair, idx)

        for pair in zip(pretoken[:-1], pretoken[1:]):
            pair_freq[pair] -= pretoken_freq[pretoken_id]
        
        for pair in zip(new_pretoken[:-1], new_pretoken[1:]):
            pair_freq[pair] += pretoken_freq[pretoken_id]
            pair_locations[pair].add(pretoken_id)

        pretoken_tuples[pretoken_id] = tuple(new_pretoken)

    return pair_freq, pair_locations, pretoken_tuples


def find_merge_pair(
        pair_freq: dict[tuple, int],
        vocab: dict[int, bytes],
    ) -> tuple[int, int]:
    return max(pair_freq, key = lambda pair: (pair_freq[pair], tuple(vocab[p] for p in pair)))


def compute_pretoken_counts(
        chunk: tuple[str | os.PathLike, int, int, list[str]]
    ) -> dict[tuple, int]:

    input_path, start, end, special_tokens = chunk

    pretoken_freq = defaultdict(int)

    split_pattern = "|".join(re.escape(special_token) for special_token in special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        file_chunk = f.read(end - start).decode("utf-8", errors="ignore")

    docs = re.split(split_pattern, file_chunk)

    for doc in docs:
        for pretoken in re.finditer(PAT, doc):
            pretoken_tuple = tuple(pretoken.group().encode('utf-8'))
            pretoken_freq[pretoken_tuple] += 1
    
    return pretoken_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    n_special_tokens = len(special_tokens)
    n_single_bytes = 256
    num_processes = 4

    vocab = {}
    merges = []
    for i in range(n_single_bytes):
        vocab[i] = bytes([i])

    pretoken_tuple_to_freq = defaultdict(int)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunks.append((input_path, start, end, special_tokens))
    
    with multiprocessing.Pool(num_processes) as pool:
        pretoken_freq_by_chunk = pool.map(compute_pretoken_counts, chunks)
        
    for pretoken_freq_chunk in pretoken_freq_by_chunk:
        for pretoken_tuple, freq in pretoken_freq_chunk.items():
            pretoken_tuple_to_freq[pretoken_tuple] += freq
    
    pretoken_freq = defaultdict(int)
    pretoken_tuples = {}

    for id, (pretoken, freq) in enumerate(pretoken_tuple_to_freq.items()):
        pretoken_freq[id] = freq
        pretoken_tuples[id] = pretoken

    pair_freq, pair_locations = get_pair_stats(pretoken_tuples, pretoken_freq)

    for i in range(n_single_bytes, vocab_size - n_special_tokens):
        int1, int2 = find_merge_pair(pair_freq, vocab)
        byte1 = vocab[int1]
        byte2 = vocab[int2]
        vocab[i] = byte1 + byte2
        merges.append((byte1, byte2))
        pair_freq, pair_locations, pretoken_tuples = merge(pair_freq, pair_locations, pretoken_tuples, pretoken_freq, (int1, int2), i)

    for i in range(n_special_tokens):
        vocab[vocab_size - n_special_tokens + i] = special_tokens[i].encode('utf-8')

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 1000, ["<|endoftext|>"])
    print(vocab)
    print(merges)
