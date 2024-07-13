from typing import List, Tuple
import torch
from .kv_cache import KVCache
from .sequence_manager import Sequence, SequenceManager

class PagedAttention:
    def __init__(self, model, kv_cache: KVCache, seq_manager: SequenceManager):
        self.model = model
        self.kv_cache = kv_cache
        self.seq_manager = seq_manager

    def allocate_kv_blocks(self, seq: Sequence):
        num_blocks_needed = (len(seq.tokens) + self.seq_manager.block_size - 1) // self.seq_manager.block_size
        new_blocks = self.kv_cache.allocate(seq.id, num_blocks_needed)
        seq.logical_blocks.extend(new_blocks)

    def forward(self, seq_id: int):
        seq = self.seq_manager.sequences[seq_id]
        if not seq.logical_blocks:
            self.allocate_kv_blocks(seq)

        input_ids = torch.tensor(seq.tokens, dtype=torch.long, device='cuda').unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        # Get the current KV cache
        key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)

        # Perform attention using the cached KV
        output = self.model(input_ids=input_ids[-1:],
                            attention_mask=attention_mask,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            kv_cache_blocks=seq.logical_blocks)

        # Generate new key and value for the current token
        new_key, new_value = self.model.generate_kv(output.hidden_states[:, -1:, :])

        # Create slot mapping for the new token
        new_slot = len(seq.tokens)
        slot_mapping = torch.tensor([seq.logical_blocks[-1] * self.kv_cache.block_size + new_slot % self.kv_cache.block_size],
                                    dtype=torch.long, device='cuda')

        # Append new key and value to the cache
        self.kv_cache.reshape_and_cache(new_key, new_value, slot_mapping)

        next_token = output.logits[:, -1, :].argmax(dim=-1)
        seq.tokens.append(next_token.item())

        if len(seq.tokens) % self.kv_cache.block_size == 0:
            self.allocate_kv_blocks(seq)

        return next_token.item()

    def swap_blocks(self, seq_id: int, new_block_mapping: List[Tuple[int, int]]):
        block_mapping = torch.tensor(new_block_mapping, dtype=torch.long, device='cpu')
        self.kv_cache.copy_blocks(block_mapping)

        # Update the sequence's logical blocks
        seq = self.seq_manager.sequences[seq_id]
        for old_block, new_block in new_block_mapping:
            index = seq.logical_blocks.index(old_block)
            seq.logical_blocks[index] = new_block