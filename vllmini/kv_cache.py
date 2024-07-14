import torch
from typing import List, Optional, Tuple
from paged_attention_cuda import cache_ops

class KVCache:
    def __init__(self, num_blocks: int, num_heads: int, head_size: int, block_size: int):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        # GPU caches
        self.key_cache = torch.zeros((num_blocks, num_heads, head_size // 8, block_size, 8), 
                                     dtype=torch.uint8, device='cuda')
        self.value_cache = torch.zeros((num_blocks, num_heads, head_size, block_size), 
                                       dtype=torch.uint8, device='cuda')

        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("Not enough free blocks")

        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.allocated_blocks[seq_id] = self.allocated_blocks.get(seq_id, []) + allocated
        return allocated

    def free(self, seq_id: int):
        if seq_id in self.allocated_blocks:
            self.free_blocks.extend(self.allocated_blocks[seq_id])
            del self.allocated_blocks[seq_id]

    def reshape_and_cache(self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor):
        cache_ops.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping, "auto", 1.0)

    def get_kv_cache(self, seq_id: int):
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"No blocks allocated for sequence {seq_id}")
        blocks = self.allocated_blocks[seq_id]
        return self.key_cache[blocks], self.value_cache[blocks]

    def copy_blocks(self, block_mapping: torch.Tensor):
        cache_ops.copy_blocks([self.key_cache], [self.value_cache], block_mapping)