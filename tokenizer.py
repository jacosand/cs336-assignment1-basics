import os

input_string = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


def find_merge_pair(pretoken_freq):
    pair_freq = {}
    for pretoken, freq in pretoken_freq.items():
        for pair in zip(pretoken, pretoken[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + freq

    return max(pair_freq, key = lambda pair: (pair_freq[pair], pair))


def merge(pretoken_freq, pair, idx):
    merged_pretoken_freq = {}

    for pretoken, freq in pretoken_freq.items():
        i = 0
        new_pretoken = pretoken
        while i < len(pretoken) - 1:
            if (pretoken[i], pretoken[i+1]) == pair:
                new_pretoken = (*new_pretoken[:i], idx, *new_pretoken[i+2:])
                i += 2
            else:
                i += 1
        merged_pretoken_freq[new_pretoken] = pretoken_freq[pretoken]

    return merged_pretoken_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    n_special_tokens = len(special_tokens)
    n_single_bytes = 256

    vocab = {}
    merges = []
    for i in range(n_special_tokens):
        vocab[i] = special_tokens[i].encode('utf-8')
    for i in range(n_single_bytes):
        vocab[n_special_tokens + i] = chr(i).encode('utf-8')

    pretokens = input_path.split()
    pretoken_freq = {}
    for pretoken in pretokens:
        pretoken_tuple = tuple(pretoken.encode('utf-8'))
        pretoken_freq[pretoken_tuple] = pretoken_freq.get(pretoken_tuple, 0) + 1
    
    for i in range(n_special_tokens + n_single_bytes, vocab_size):
        int1, int2 = find_merge_pair(pretoken_freq)
        byte1 = vocab[int1 + n_special_tokens]
        byte2 = vocab[int2 + n_special_tokens]
        merges.append((byte1, byte2))
        pretoken_freq = merge(pretoken_freq, (int1, int2), i)
        vocab[i + n_special_tokens] = byte1 + byte2

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe(input_string, 263, ["<|endoftext|>"])
    print(vocab)
    print(merges)
