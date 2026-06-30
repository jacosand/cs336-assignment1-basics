# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Answers to questions

### `unicode1`

#### (a) What Unicode character does `chr(0)` return?

It returns the null character `'\x00'`.

#### (b) How does this character’s string representation (`__repr__()`) differ from its printed representation?

The string representation of the character by itself is `'\\x00'`.

#### (c) What happens when this character occurs in text?

In text, the character is stored as `\x00`, but in the text's string representation the character is not shown at all.  However, the character is still present as is demonstrated by measuring the length of the string.

### `unicode2`

#### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

UTF-16 and UTF-32 encodings are much longer, so less text will fit into a transformer's context window.  In addition, because UTF-32 is fixed-length, it contains a lot of zeros that contain no semantic meaning but take up the transformer's context window.

#### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

The function is incorrect because often multiple bytes correspond to a single Unicode character.  An example is the character `'牛'`, whose byte string is `b'\xe7\x89\x9b'`.  The first byte `b'\xe7'` cannot be decoded on its own.

#### (c) Give a two-byte sequence that does not decode to any Unicode character(s).

The two-byte sequence `b'\xe7\x89` does not decode to any Unicode character(s), because those are the first two bytes of, for example, the character `'牛'`, whose byte string is `b'\xe7\x89\x9b'`.

### `train_bpe_tinystories`

#### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

The training took 102.25 seconds and had a peak memory use of 446 MB.  The longest token in the vocabular is ` accomplishment`, which makes sense since it is a relatively common 15-byte string.

#### (b) Profile your code. What part of the tokenizer training process takes the most time?

The pretokenization takes the most time: 94.52 seconds.

### `train_bpe_owt`

#### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?

The training took 19.49 minutes and had a peak memory use of 7.86 GB.  The longest token is `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`, which makes sense since OpenWebText is a much more realistic dataset of text on the web than the curated TinyStories dataset.

#### (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

The vocabulary in the TinyStories tokenizer is highly curated, since the underlying dataset is highly curated.  The words generally correspond to real words.  In contrast, the OpenWebText tokenizer yields a vocabulary that is much less curated, consisting of any conglomerations of characters frequently found on the internet.

### `tokenizer_experiments`

#### (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer's compression ratio (bytes/token)?
The compression ratio for TinyStories is 4.126 bytes/token, while the compression ratio for OpenWebText is 4.678 bytes/token.

#### (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?  Compare the compression ratio and/or qualitatively describe what happens.
The compression ratio goes down to 3.187 bytes/token, because the TinyStories tokenizer was not specifically created for the OpenWebText dataset, so it is less efficient.

#### (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
The throughput for the TinyStories tokenizer is 719235 bytes/second, and the throughput for the OpenWebText tokenizer is 548926 bytes/second.  Tokenizing the Pile dataset would take `825 * 10**9 bytes / (719235 bytes/second) / (60 seconds/minute) / (60 minutes/hour) / (24 hours/day) = 13.28 days` with the TinyStories tokenizer and  `825 * 10**9 bytes / (548926 bytes/second) / (60 seconds/minute) / (60 minutes/hour) / (24 hours/day) = 17.40 days` with the OpenWebText tokenizer.

#### (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We'll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?

`uint16` is an appropriate choice because `2**16 = 65536`, so having available integers from `0` to `65535` for tokenization accommodates vocabulary sizes of 10000 (TinyStories) and 32000 (OpenWebText).  Each token then requires just 2 bytes to store.
