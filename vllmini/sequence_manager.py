from typing import Dict, List, Tuple
import torch
from .kv_cache import KVCache

class Block:
    def __init__(self, block_size: int, head_size: int, num_heads: int):
        self.key = torch.zeros((num_heads, head_size, block_size))
        self.value = torch.zeros((num_heads, head_size, block_size))

class Sequence:
    def __init__(self, seq_id: int, prompt: List[int], max_length: int):
        self.id = seq_id
        self.tokens = prompt
        self.max_length = max_length
        self.logical_blocks: List[int] = []

class SequenceManager:
    def __init__(self, block_size: int, kv_cache: KVCache):
        self.block_size = block_size
        self.sequences: Dict[int, Sequence] = {}
        self.kv_cache = kv_cache
        self.active_sequences = set()

    def add_sequence(self, seq_id: int, prompt: List[int], max_length: int) -> Sequence:
        seq = Sequence(seq_id, prompt, max_length)
        self.sequences[seq_id] = seq
        self.active_sequences.add(seq_id)
        return seq

    def remove_sequence(self, seq_id: int):
        if seq_id in self.sequences:
            self.kv_cache.free(seq_id)
            del self.sequences[seq_id]
            self.active_sequences.remove(seq_id)