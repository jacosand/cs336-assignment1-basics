from typing import Iterable, Iterator
import pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key = len, reverse = True)
            current_token_id = len(self.vocab)
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in self.vocab.values():
                    self.vocab[current_token_id] = special_token.encode("utf-8")
                    current_token_id += 1
        
        self.token_ids = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        
        tokens = []

        if self.special_tokens:
            split_pattern = "|".join(re.escape(special_token) for special_token in self.special_tokens)
            docs = re.split(f"({split_pattern})", text)
        else:
            docs = [text]

        for doc in docs:
            if self.special_tokens and doc in self.special_tokens:
                tokens.append(self.token_ids[doc.encode("utf-8")])
            else:
                for pretoken in re.finditer(PAT, doc):
                    pretoken_bytes = [bytes([b]) for b in pretoken.group().encode("utf-8")]
                    new_pretoken_bytes = []

                    for merge in self.merges:
                        pretoken_index = 0
                        while pretoken_index <= len(pretoken_bytes) - 1:
                            if pretoken_index < len(pretoken_bytes) - 1 and tuple(pretoken_bytes[pretoken_index:pretoken_index+2]) == merge:
                                new_pretoken_bytes.append(pretoken_bytes[pretoken_index] + pretoken_bytes[pretoken_index+1])
                                pretoken_index += 2
                            else:
                                new_pretoken_bytes.append(pretoken_bytes[pretoken_index])
                                pretoken_index += 1
                        pretoken_bytes = new_pretoken_bytes
                        new_pretoken_bytes = []
            
                    tokens += [self.token_ids[b] for b in pretoken_bytes]

        return tokens


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def main():
    tokenizer = Tokenizer.from_files(
        vocab_filepath = "artifacts/tokenizer-tinystories-10000-vocab.pkl",
        merges_filepath = "artifacts/tokenizer-tinystories-10000-merges.pkl",
        special_tokens = None,
    )
    
    encoding = tokenizer.encode("hello")
    print(encoding)
    decoding = tokenizer.decode(encoding)
    print(decoding)

if __name__ == "__main__":
    main()