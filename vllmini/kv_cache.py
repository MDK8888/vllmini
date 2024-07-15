import torch
from typing import List, Optional, Tuple, Dict
from paged_attention_cuda import cache_ops

class KVCache:
    def __init__(self, num_layers: int, num_blocks: int, num_heads: int, head_size: int, block_size: int):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size

        # GPU caches for each layer
        self.key_caches = [torch.zeros((num_blocks, num_heads, head_size // 8, block_size, 8), 
                                       dtype=torch.uint8, device='cuda') for _ in range(num_layers)]
        self.value_caches = [torch.zeros((num_blocks, num_heads, head_size, block_size), 
                                         dtype=torch.uint8, device='cuda') for _ in range(num_layers)]

        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks: Dict[int, List[int]] = {}
        
        # CPU swap space
        self.cpu_key_caches = [torch.zeros_like(cache, device='cpu') for cache in self.key_caches]
        self.cpu_value_caches = [torch.zeros_like(cache, device='cpu') for cache in self.value_caches]
        self.swapped_blocks: Dict[int, List[int]] = {}

    def allocate(self, seq_id: int, num_blocks: int) -> List[int]:
        if len(self.free_blocks) < num_blocks:
            self._swap_out()  # Attempt to free up space by swapping out

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

    def reshape_and_cache(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor):
        cache_ops.reshape_and_cache(key, value, self.key_caches[layer_idx], self.value_caches[layer_idx], slot_mapping, "auto", 1.0)

    def get_kv_cache(self, seq_id: int):
        if seq_id in self.swapped_blocks:
            self._swap_in(seq_id)
        
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"No blocks allocated for sequence {seq_id}")
        
        blocks = self.allocated_blocks[seq_id]
        return [cache[blocks] for cache in self.key_caches], [cache[blocks] for cache in self.value_caches]

    def copy_blocks(self, block_mapping: torch.Tensor):
        cache_ops.copy_blocks(self.key_caches, self.value_caches, block_mapping)

    def _swap_out(self):
        # Simple FCFS policy: swap out the earliest allocated sequence
        if not self.allocated_blocks:
            return
        
        seq_id_to_swap = min(self.allocated_blocks.keys())
        blocks_to_swap = self.allocated_blocks[seq_id_to_swap]
        
        for layer_idx in range(self.num_layers):
            self.cpu_key_caches[layer_idx][blocks_to_swap] = self.key_caches[layer_idx][blocks_to_swap].cpu()
            self.cpu_value_caches[layer_idx][blocks_to_swap] = self.value_caches[layer_idx][blocks_to_swap].cpu()
        
        self.swapped_blocks[seq_id_to_swap] = blocks_to_swap
        self.free_blocks.extend(blocks_to_swap)
        del self.allocated_blocks[seq_id_to_swap]

    def _swap_in(self, seq_id: int):
        if seq_id not in self.swapped_blocks:
            return
        
        blocks_to_swap = self.swapped_blocks[seq_id]
        
        # Ensure we have enough free blocks
        if len(self.free_blocks) < len(blocks_to_swap):
            self._swap_out()  # This might recursively call _swap_out multiple times
        
        for layer_idx in range(self.num_layers):
            self.key_caches[layer_idx][blocks_to_swap] = self.cpu_key_caches[layer_idx][blocks_to_swap].cuda()
            self.value_caches[layer_idx][blocks_to_swap] = self.cpu_value_caches[layer_idx][blocks_to_swap].cuda()
        
        self.allocated_blocks[seq_id] = blocks_to_swap
        self.free_blocks = [block for block in self.free_blocks if block not in blocks_to_swap]
        del self.swapped_blocks[seq_id]