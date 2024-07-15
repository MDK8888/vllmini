from typing import List, Tuple, Optional
import torch
from transformers import GPT2Config
from vllmini.model.gpt2 import GPT2LMHeadModel
from .kv_cache import KVCache

class PagedAttention:
    def __init__(self, model_name: str, kv_cache: KVCache, device: str):
        self.device = device
        config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2LMHeadModel(config)
        self.model.load_huggingface_weights(model_name)
        self.model = self.model.to(self.device)
        self.kv_cache = kv_cache

    def forward(self, 
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                use_cache: bool = True,
                is_prefill: bool = True,
                seq_id:int = -1,
                slot_mapping: Optional[torch.Tensor] = None):
        
        batch_size, seq_len = input_ids.shape

        if is_prefill:
            # Allocate new blocks for prefill
            num_blocks_needed = (seq_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
            self.kv_cache.allocate(seq_id, num_blocks_needed)
            
            # Use vanilla attention for prefilling
            output, presents = self.model(input_ids, position_ids, attention_mask, use_cache, is_prefill)
            
            # Cache the key-value pairs
            if slot_mapping is not None:
                for layer_idx, (key, value) in enumerate(presents):
                    # initially, key.shape, value.shape will be (batch_size, num_heads, seq_len, head_dim).
                    # we want them to be (seq_len, num_heads, head_dim)
                    key, value = key.squeeze(0).permute(1, 0, 2), value.squeeze(0).permute(1, 0, 2)
                    self.kv_cache.reshape_and_cache(key, value, slot_mapping)
        else:
            # Decoding (auto-regressive generation)
            if slot_mapping is None:
                raise ValueError("slot_mapping must be provided for decoding")

            # Get the cached key-value pairs for this sequence
            key_cache, value_cache = self.kv_cache.get_kv_cache(seq_id)

            # Reshape key_cache and value_cache to match GPT2LMHeadModel expectations
            num_blocks, num_heads, head_size_over_x, block_size, x = key_cache.shape
            head_size = head_size_over_x * x
            
            key_cache = key_cache.permute(0, 3, 1, 2, 4).reshape(num_blocks * block_size, num_heads, head_size)
            value_cache = value_cache.permute(0, 3, 1, 2).reshape(num_blocks * block_size, num_heads, head_size)
            key_cache, value_cache = key_cache.unsqueeze(0), value_cache.unsqueeze(0)

            key_cache, value_cache = key_cache.permute(0, 2, 1, 3), value_cache.permute(0, 2, 1, 3)
            print("key_cache.shape:", key_cache.shape)
            print("value_cache.shape:", value_cache.shape)


            # Prepare inputs for paged attention
            block_tables = torch.tensor([self.kv_cache.allocated_blocks[seq_id]], dtype=torch.long, device=self.device)
            seq_lens = torch.tensor([seq_len], dtype=torch.long, device=self.device)
            max_seq_len = self.kv_cache.num_blocks * self.kv_cache.block_size

            # Use paged attention for decoding
            output, presents = self.model(
                input_ids,
                position_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_caches=key_cache,
                value_caches=value_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len
            )

            # Cache the new key-value pair
            if use_cache:
                for layer_idx, (key, value) in enumerate(presents):
                    self.kv_cache.reshape_and_cache(key, value, slot_mapping)

        return output

    def copy_blocks(self, seq_id: int, new_block_mapping: List[Tuple[int, int]]):
        block_mapping = torch.tensor(new_block_mapping, dtype=torch.long, device=self.device)
        self.kv_cache.copy_blocks(block_mapping)

        # Update the sequence's logical blocks
        old_blocks = self.kv_cache.allocated_blocks[seq_id]
        new_blocks = [new_block for _, new_block in new_block_mapping]
        self.kv_cache.allocated_blocks[seq_id] = new_blocks

        # Update free blocks
        self.kv_cache.free_blocks.extend(old_blocks)
        self.kv_cache.free_blocks = list(set(self.kv_cache.free_blocks) - set(new_blocks))

    def free_memory(self, seq_id: int):
        self.kv_cache.free(seq_id)