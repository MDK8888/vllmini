from typing import Dict, Tuple, List
import torch
from .kv_cache import KVCache

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_size: int, max_blocks_per_seq:int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size

        self.kv_cache = KVCache(num_blocks, num_heads, head_size, block_size, max_blocks_per_seq)
        self.cpu_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[int, List[int], List[torch.Tensor], List[List[int]]]:
        allocated, slot_mappings, paged_attention_block_table = self.kv_cache.allocate_for_prefill(seq_id, num_layers, seq_len)
        return seq_id, allocated, slot_mappings, paged_attention_block_table

    def get_block_table(self, seq_id: int) -> List[Tuple[int, int]]:
        return self.kv_cache.get_block_table(seq_id)

    def get_paged_attention_block_table(self, seq_id: int) -> List[List[int]]:
        return self.kv_cache.get_paged_attention_block_table(seq_id)

    def get_kv_cache(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache.get_kv_cache(seq_id)

    def decode_step(self, seq_id: int, input_len: int) -> Tuple[List[List[int]], torch.Tensor]:
        # This function remains unchanged as per your request
        block_table = self.kv_cache.get_block_table(seq_id)
        paged_attention_block_table = self.kv_cache.get_paged_attention_block_table(seq_id)

        new_slot_mapping = []
        for layer_idx, layer_blocks in enumerate(paged_attention_block_table):
            last_block = -1
            for i in range(1, len(layer_blocks[0])):
                if layer_blocks[0][i] == -1:
                    last_block = layer_blocks[0][i-1]
                    break
            
            for (block, filled) in block_table:
                if block == last_block:
                    last_block_info = (block, filled)
                    break
            
            _, num_filled = last_block_info

            if num_filled == self.block_size:
                print("in BlockManager.decode_step, block is full, we need to search for new block.")
                # Current block is full, append a new one
                new_block = self.kv_cache.append_block(seq_id, layer_idx)
                last_block = new_block
                num_filled = 0

            new_slot = last_block * self.block_size + num_filled
            new_slot_mapping.append(torch.tensor([new_slot], dtype=torch.long, device="cuda"))
            
            # Update the block table with the new number of filled slots
            self.kv_cache.update_block_table(seq_id, last_block, num_filled + input_len)

        paged_attention_block_table = self.kv_cache.get_paged_attention_block_table(seq_id)        

        return paged_attention_block_table, new_slot_mapping

    def free(self, seq_id: int):
        self.kv_cache.free(seq_id)
        if seq_id in self.cpu_cache:
            del self.cpu_cache[seq_id]

    def swap_to_cpu(self, seq_id: int):
        key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)
        self.cpu_cache[seq_id] = (key_cache.cpu(), value_cache.cpu())
        self.kv_cache.free(seq_id)

    def swap_from_cpu(self, seq_id: int) -> bool:
        if seq_id not in self.cpu_cache:
            return False

        cpu_key_cache, cpu_value_cache = self.cpu_cache[seq_id]
        try:
            allocated = self.kv_cache.allocate(seq_id, cpu_key_cache.size(0))
            key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)
            key_cache.copy_(cpu_key_cache.cuda())
            value_cache.copy_(cpu_value_cache.cuda())
            del self.cpu_cache[seq_id]
            return True
        except RuntimeError:
            return False