from typing import List
import torch
import cache_ops  # Assuming this module exposes the CUDA kernels

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

        # CPU caches
        self.cpu_key_cache = torch.zeros_like(self.key_cache, device='cpu')
        self.cpu_value_cache = torch.zeros_like(self.value_cache, device='cpu')

        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = {}
        self.swapped_blocks = {}

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            self._swap_out()
        
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("Not enough free blocks even after swapping")

        allocated = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.allocated_blocks[seq_id] = self.allocated_blocks.get(seq_id, []) + allocated
        return allocated

    def free(self, seq_id: int):
        if seq_id in self.allocated_blocks:
            self.free_blocks.extend(self.allocated_blocks[seq_id])
            del self.allocated_blocks[seq_id]
        if seq_id in self.swapped_blocks:
            del self.swapped_blocks[seq_id]

    def reshape_and_cache(self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor):
        cache_ops.reshape_and_cache(key, value, self.key_cache, self.value_cache, slot_mapping, "fp8", 1.0)

    def get_kv_cache(self, seq_id: int):
        if seq_id in self.swapped_blocks:
            self._swap_in(seq_id)
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"No blocks allocated for sequence {seq_id}")
        blocks = self.allocated_blocks[seq_id]
        return self.key_cache[blocks], self.value_cache[blocks]

    def _swap_out(self):
        # Select a sequence to swap out (FCFS policy)
        seq_to_swap = min(self.allocated_blocks.keys())
        blocks_to_swap = self.allocated_blocks[seq_to_swap]

        # Prepare block mapping
        block_mapping = torch.tensor([(b, b) for b in blocks_to_swap], dtype=torch.long)

        # Use swap_blocks kernel to move data to CPU
        cache_ops.swap_blocks(self.key_cache, self.cpu_key_cache, block_mapping)
        cache_ops.swap_blocks(self.value_cache, self.cpu_value_cache, block_mapping)

        # Update bookkeeping
        self.free_blocks.extend(blocks_to_swap)
        self.swapped_blocks[seq_to_swap] = blocks_to_swap
        del self.allocated_blocks[seq_to_swap]

    def _swap_in(self, seq_id: int):
        blocks_to_swap = self.swapped_blocks[seq_id]
        new_blocks = self.allocate(seq_id, len(blocks_to_swap))

        # Prepare block mapping
        block_mapping = torch.tensor(list(zip(blocks_to_swap, new_blocks)), dtype=torch.long)

        # Use swap_blocks kernel to move data back to GPU
        cache_ops.swap_blocks(self.cpu_key_cache, self.key_cache, block_mapping)
        cache_ops.swap_blocks(self.cpu_value_cache, self.value_cache, block_mapping)

        del self.swapped_blocks[seq_id]

    def copy_blocks(self, block_mapping: torch.Tensor):
        cache_ops.copy_blocks([self.key_cache], [self.value_cache], block_mapping)