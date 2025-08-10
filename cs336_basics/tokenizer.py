import os
import regex as re
from collections import defaultdict, Counter
from typing import Iterator, BinaryIO, DefaultDict, Iterable
from functools import cache
from multiprocessing import Pool


PAT_STR: str = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@cache
def word_to_id_tuple(word: bytes, len_special: int) -> tuple[int, ...]:
    return tuple(len_special + byte for byte in word)


def chunkate(input_text: str, special_tokens: list[str]) -> list[str]:
    return re.split('|'.join([re.escape(token) for token in special_tokens]), 
                        input_text)

def pretokenize(input_text: str, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    chunks = chunkate(input_text, special_tokens)
    pretoken_count = defaultdict(int)
    len_special = len(special_tokens)

    for chunk in chunks:
        for match in re.finditer(PAT_STR, chunk):
            word = match.group().encode('utf-8')
            pretoken_count[word_to_id_tuple(word, len_special)] += 1
    return Counter(pretoken_count)


def pretokenize_for_encoding(text: str, special_tokens: list[str]) -> list[tuple[int, ...]]:
    chunks = chunkate(text, special_tokens)
    pretokens = []
    len_special = len(special_tokens)

    for chunk in chunks:
        for match in re.finditer(PAT_STR, chunk):
            word = match.group().encode('utf-8')
            pretokens.append(word_to_id_tuple(word, len_special))

    return pretokens


def pretokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter[tuple[int, ...]]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize(chunk, special_tokens)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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


def pretokenize_from_path(input_path: str, special_tokens: list[str], num_processes: int) -> Counter[tuple[int, ...]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, split_special_token="<|endoftext|>".encode("utf-8"))
            
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_processes) as pool:
        # Process each chunk in parallel
        pretoken_counts = pool.starmap(pretokenize_chunk, chunk_args)
        
    pretoken_count = sum(pretoken_counts, Counter())    
    return pretoken_count


def update_pairs(pairs, pretoken: tuple[int, ...], count: int):
    for i in range(len(pretoken) - 1):
        pairs[pretoken[i:i + 2]] += count
    return pairs


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], 
              num_processes=4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {}
    merges = []

    vocab = {i: token.encode('utf-8') for i, token in enumerate(special_tokens)}
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])



    pretoken_count = pretokenize_from_path(input_path, special_tokens, num_processes)

    # Initialize pairs with counts from pretoken_count
    pairs = defaultdict(int)
    pair_to_pretokens = defaultdict(set)

    for pretoken, count in pretoken_count.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pairs[pair] += count
            pair_to_pretokens[pair].add(pretoken)

    while len(vocab) < vocab_size:
    
        if not pairs:
            breakpoint()
            break

        freq_pair = max(pairs, key=lambda x: (pairs[x], (vocab[x[0]], vocab[x[1]])))

        affected_pretokens = pair_to_pretokens[freq_pair].copy()

        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[freq_pair[0]] + vocab[freq_pair[1]]


        for old_pretoken in affected_pretokens:
            if old_pretoken not in pretoken_count:
                continue
            new_pretoken = replace_pair_in_tuple(freq_pair, new_token_id, old_pretoken)
            count = pretoken_count.pop(old_pretoken)
            pretoken_count[new_pretoken] = count

            for i in range(len(old_pretoken) - 1):
                pair = old_pretoken[i:i + 2]
                pairs[pair] -= count
                if pairs[pair] == 0:
                    del pairs[pair]
                    del pair_to_pretokens[pair]

            for i in range(len(new_pretoken) - 1):
                pair = new_pretoken[i:i + 2]
                pairs[pair] += count
                pair_to_pretokens[pair].add(new_pretoken)

        merges.append((vocab[freq_pair[0]], vocab[freq_pair[1]]))

    return vocab, merges


def replace_pair_in_tuple(
        pair: tuple[int, int],
        new_token_id: int,
        pretoken: tuple[int, ...]) -> tuple[int, ...]:
    pretoken_list = []
    i = 0
    while i < len(pretoken) :
        if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
            pretoken_list.append(new_token_id)
            i += 2
        else:
            pretoken_list.append(pretoken[i])
            i += 1
    return tuple(pretoken_list)


def save_vocab(vocab: dict[int, bytes], filepath: str) -> None:
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({str(k): v.decode('latin-1') for k, v in vocab.items()}, f, ensure_ascii=False)


def save_merges(merges: list[tuple[bytes, bytes]], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for token1, token2 in merges:
            f.write(f"{token1.decode('latin-1')} {token2.decode('latin-1')}\n")


def load_vocab(filepath: str) -> dict[int, bytes]:
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return {int(k): v.encode('latin-1') for k, v in vocab.items()}


def load_merges(filepath: str) -> list[tuple[bytes, bytes]]:
    merges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Split on exactly one space to handle tokens that contain spaces
                parts = line.rstrip('\n').split(' ', 1)  # Split on first space only
                token1, token2 = parts
                merges.append((token1.encode('latin-1'), token2.encode('latin-1')))
    return merges

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}
        

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str) -> list[int]:
        pretokens = pretokenize_for_encoding(text, self.special_tokens)

        tokens = []
        for pretoken in pretokens:
            if len(pretoken) == 1 and pretoken[0] < len(self.special_tokens):
                tokens.append(pretoken[0])
            else:
                encoded = self.apply_merges(pretoken)
                tokens.extend(encoded)
        return tokens
        
    def apply_merges(self, pretoken: tuple[int, ...]) -> tuple[int, ...]:
        new_pretoken = pretoken
        for token1, token2 in self.merges:
            id1, id2 = self.bytes_to_id[token1], self.bytes_to_id[token2]
            new_token_id = self.bytes_to_id[token1 + token2]
            new_pretoken = replace_pair_in_tuple((id1, id2), new_token_id, new_pretoken)
        return new_pretoken

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            for token in tokens:
                yield token

    def decode(self, ids: list[int]) -> str:
        res = []
        for id in ids:
            if id < len(self.special_tokens):
                res.append(self.special_tokens[id])
            else:
                res.append(self.vocab[id].decode('utf-8', errors='replace'))
        return ''.join(res)





if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1024
    num_processes = 4

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=4
    )

    vocab_path = "data/vocab.json"
    merges_path = "data/merges.txt"

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    # tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)





