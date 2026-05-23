import os
from pathlib import Path
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    path = Path(input_path)
    input_text = path.read_text()

    n_special_tokens = len(special_tokens)
    n_single_bytes = 256

    vocab = {}
    merges = []
    for i in range(n_single_bytes):
        vocab[i] = bytes([i])

    split_pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
    docs = re.split(split_pattern, input_text)

    pretoken_freq = {}
    for doc in docs:
        for pretoken in re.finditer(PAT, doc):
            pretoken_tuple = tuple(pretoken.group().encode('utf-8'))
            pretoken_freq[pretoken_tuple] = pretoken_freq.get(pretoken_tuple, 0) + 1
    
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
