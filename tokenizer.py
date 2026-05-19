input_string = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


def find_merge_pair(freq_table):
    pair_counts = {}
    for k, v in freq_table.items():
        for pair in zip(k, k[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + v

    return max(pair_counts, key = lambda pair: (pair_counts[pair], pair))

def merge(freq_table, pair, idx):
    new_freq_table = {}

    for k, v in freq_table.items():
        i = 0
        new_k = k
        while i < len(k) - 1:
            if (k[i], k[i+1]) == pair:
                new_k = (*new_k[:i], idx, *new_k[i+2:])
                i += 2
            else:
                i += 1
        new_freq_table[new_k] = freq_table[k]

    return new_freq_table

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
    ) -> (dict[int, bytes], list[tuple[bytes, bytes]]):

    pre_tokens = input_path.split()

    vocab = {}
    for i in range(256):
        vocab[i] = chr(i).encode('utf-8')

    freq_table = {}
    for pt in pre_tokens:
        pt_bytes = tuple(pt.encode('utf-8'))
        freq_table[pt_bytes] = freq_table.get(pt_bytes, 0) + 1
    
    merge_pair = find_merge_pair(freq_table)

    for i in range(256, vocab_size):
        merge_pair = find_merge_pair(freq_table)
        freq_table = merge(freq_table, merge_pair, i)

    #vocab[i] = b"".join(chr(c).encode('utf-8') for c in merge_pair)
    #print(vocab)

    merges = []


    return vocab, merges

if __name__ == "__main__":
    train_bpe(input_string, 256+6, [])
