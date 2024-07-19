import torch
import time
from typing import Dict, List, Tuple

class KVCache:
    def __init__(self, num_blocks: int, num_heads: int, head_size: int, block_size: int, max_blocks_per_seq:int):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.max_blocks_per_seq = max_blocks_per_seq

        self.key_cache = torch.zeros(num_blocks, num_heads, head_size // 8, block_size, 8, dtype=torch.float16, device='cuda')
        self.value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device='cuda')

        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks: Dict[int, List[int]] = {}
        self.block_tables: Dict[int, List[Tuple[int, int]]] = {}  # seq_id: [(block, num_filled), ...]
        self.paged_attention_block_tables: Dict[int, List[List[int]]] = {}  # seq_id: [[layer0_blocks], [layer1_blocks], ...]

    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[List[int], List[torch.Tensor], List[List[int]]]:
        if len(self.free_blocks) < num_layers:
            raise RuntimeError("Not enough free blocks for prefill allocation")

        allocated = self.free_blocks[:num_layers]
        self.free_blocks = self.free_blocks[num_layers:]
        self.allocated_blocks[seq_id] = allocated
        
        # Initialize both block tables
        self.block_tables[seq_id] = [(block, min(seq_len, self.block_size)) for block in allocated]
        self.paged_attention_block_tables[seq_id] = [torch.tensor([[block] + [-1] * (self.max_blocks_per_seq - 1)], device="cuda", dtype=torch.int32) \
                                                     for block in allocated]

        # Calculate slot mappings based on allocated blocks
        slot_mappings = [torch.arange(seq_len, dtype=torch.long, device='cuda') + block * self.block_size for block in allocated]

        return allocated, slot_mappings, self.paged_attention_block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> List[Tuple[int, int]]:
        if seq_id not in self.block_tables:
            raise ValueError(f"No block table for sequence {seq_id}")
        return self.block_tables[seq_id]

    def get_paged_attention_block_table(self, seq_id: int) -> List[List[int]]:
        if seq_id not in self.paged_attention_block_tables:
            raise ValueError(f"No paged attention block table for sequence {seq_id}")
        return self.paged_attention_block_tables[seq_id]

    def get_kv_cache(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"No KV cache allocated for sequence {seq_id}")

        blocks = self.allocated_blocks[seq_id]
        return self.key_cache[blocks], self.value_cache[blocks]

    def append_block(self, seq_id: int, layer_idx: int):
        if len(self.free_blocks) == 0:
            raise RuntimeError("No free blocks available")
        
        new_block = self.free_blocks.pop(0)
        self.allocated_blocks[seq_id].append(new_block)
        
        # Update both block tables
        self.block_tables[seq_id].append((new_block, 0))
        new_block_index = 0
        for i in range(self.max_blocks_per_seq):
            if self.paged_attention_block_tables[seq_id][layer_idx][0][i] == -1:
                new_block_index = i
                break

        self.paged_attention_block_tables[seq_id][layer_idx][0][new_block_index] = new_block

        return new_block

    def update_block_table(self, seq_id: int, block: int, num_filled: int):
        for i, (b, _) in enumerate(self.block_tables[seq_id]):
            if b == block:
                self.block_tables[seq_id][i] = (block, num_filled)
                break

    def free(self, seq_id: int):
        if seq_id in self.allocated_blocks:
            self.free_blocks.extend(self.allocated_blocks[seq_id])
            del self.allocated_blocks[seq_id]
            del self.block_tables[seq_id]
            del self.paged_attention_block_tables[seq_id]