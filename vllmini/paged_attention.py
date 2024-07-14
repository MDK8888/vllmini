from typing import List, Tuple, Optional
import torch
from .kv_cache import KVCache
from .sequence_manager import Sequence, SequenceManager

class PagedAttention:
    def __init__(self, model, kv_cache: KVCache):
        self.model = model
        self.kv_cache = kv_cache

    def allocate_kv_blocks(self, seq_id: int, seq_len: int):
        num_blocks_needed = (seq_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        return self.kv_cache.allocate(seq_id, num_blocks_needed)

    def forward(self, 
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = True,
                is_prefill: bool = True,
                seq_id: Optional[int] = None,
                key_caches: Optional[List[torch.Tensor]] = None,
                value_caches: Optional[List[torch.Tensor]] = None,
                block_tables: Optional[torch.Tensor] = None,
                seq_lens: Optional[torch.Tensor] = None,
                max_seq_len: Optional[int] = None):
        
        batch_size, seq_len = input_ids.shape

        if is_prefill:
            # Allocate new blocks for prefill
            if seq_id is not None:
                self.allocate_kv_blocks(seq_id, seq_len)
            
            # Use vanilla attention for prefilling
            output, presents = self.model(input_ids, position_ids, attention_mask, use_cache, is_prefill)
            
            # Cache the key-value pairs
            if seq_id is not None:
                for layer_idx, (key, value) in enumerate(presents):
                    slot_mapping = torch.arange(seq_len, device='cuda')
                    self.kv_cache.reshape_and_cache(key, value, slot_mapping)
        else:
            # Decoding (auto-regressive generation)
            if seq_id is None:
                raise ValueError("seq_id must be provided for decoding")

            # Get the cached key-value pairs
            key_caches, value_caches = self.kv_cache.get_kv_cache(seq_id)
            
            # Prepare block tables and sequence lengths
            if block_tables is None:
                block_tables = torch.tensor([self.kv_cache.allocated_blocks[seq_id]], dtype=torch.long, device='cuda')
            if seq_lens is None:
                seq_lens = torch.tensor([seq_len], dtype=torch.long, device='cuda')
            if max_seq_len is None:
                max_seq_len = seq_len

            # Use paged attention for decoding
            output, presents = self.model(
                input_ids,
                position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_caches=key_caches,
                value_caches=value_caches,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len
            )

            # Cache the new key-value pair
            if use_cache:
                for layer_idx, (key, value) in enumerate(presents):
                    slot_mapping = torch.tensor([seq_len - 1], dtype=torch.long, device='cuda')
                    self.kv_cache.reshape_and_cache(key, value, slot_mapping)

        return output

    def swap_blocks(self, seq_id: int, new_block_mapping: List[Tuple[int, int]]):
        block_mapping = torch.tensor(new_block_mapping, dtype=torch.long, device='cuda')
        self.kv_cache.copy_blocks(block_mapping)

        # Update the sequence's logical blocks
        old_blocks = self.kv_cache.allocated_blocks[seq_id]
        new_blocks = [new_block for _, new_block in new_block_mapping]
        self.kv_cache.allocated_blocks[seq_id] = new_blocks

        # Update free blocks
        self.kv_cache.free_blocks.extend(old_blocks)
        self.kv_cache.free_blocks = list(set(self.kv_cache.free_blocks) - set(new_blocks))