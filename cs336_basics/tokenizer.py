import os
import regex as re
from collections import defaultdict, Counter
from typing import Iterator, BinaryIO, DefaultDict
from functools import cache


class BPETokenizer:

    pat_str: str = \
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pairs: DefaultDict[bytes, int] = defaultdict(int)

    def __init__(self):
        pass

    def pretokenize(self, input_text: str, special_tokens: list[str]) -> dict[bytes, int]:
        chunks = re.split('|'.join([re.escape(token) for token in special_tokens]), 
                          input_text)
        pretoken_count = defaultdict(int)

        @cache
        def word_to_id_tuple(word: bytes) -> tuple[int, ...]:
            return tuple(len(special_tokens) + byte for byte in word)

        for chunk in chunks:
            for match in re.finditer(self.pat_str, chunk):
                word = match.group().encode('utf-8')
                pretoken_count[word_to_id_tuple(word)] += 1
        return pretoken_count


    def find_chunk_boundaries(
        self,
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
    
    def update_pairs(self, pairs, pretoken: tuple[int, ...], count: int):
        for i in range(len(pretoken) - 1):
            pairs[pretoken[i:i + 2]] += count
        return pairs

    def merge_pairs(self):
        pass

    def tokenize(self, input_path: str, vocab_size: int, special_tokens: list[str], 
                 num_processes=4) -> tuple[dict[int, bytes], list[tuple[int, int]]]:

        vocab = {}
        merges = []

        vocab = {i: token.encode('utf-8') for i, token in enumerate(special_tokens)}
        for i in range(256):
            vocab[len(special_tokens) + i] = bytes([i])

        pretoken_count = Counter()

        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, num_processes, split_special_token="<|endoftext|>".encode("utf-8"))
                
            # The following is a serial implementation, but you can parallelize this 
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                pretoken_count += Counter(self.pretokenize(chunk, special_tokens))


        while len(vocab) < vocab_size:
            pairs = defaultdict(int)

            for pretoken, count in pretoken_count.items():
                pairs = self.update_pairs(pairs, pretoken, count)
        
            freq_pair = max(pairs, key=pairs.get)

            new_token_id = len(vocab)
            vocab[new_token_id] = vocab[freq_pair[0]] + vocab[freq_pair[1]]

            new_pretoken_count = Counter()
            for pretoken, count in pretoken_count.items():
                new_pretoken = self.replace_pair_in_tuple(freq_pair, new_token_id, pretoken)
                new_pretoken_count[new_pretoken] += count
            pretoken_count = new_pretoken_count
            merges.append(freq_pair)
            
        return vocab, merges


    def replace_pair_in_tuple(
            self, 
            pair: tuple[int, int],
            new_token_id: int,
            pretoken: tuple[int, ...]) -> tuple[int, ...]:
        pretoken_list = list(pretoken)
        i = 0
        while i < len(pretoken) - 1:
            if pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
                pretoken_list[i:i + 2] = [new_token_id]
            i += 1
        return tuple(pretoken_list)



if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    input_path = "data/TinyStoriesV2-GPT4-valid.txt"

    tokenizer = BPETokenizer()
    self = tokenizer
    vocab, merges = tokenizer.tokenize(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=special_tokens,
        num_processes=4
    )

