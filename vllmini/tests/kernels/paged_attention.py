import torch
import unittest
from typing import Tuple, List
from paged_attention_cuda import cache_ops, paged_attention_v1

class TestPagedAttention(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 1000
        self.num_heads = 12
        self.head_size = 64
        self.block_size = 16
        self.batch_size = 1
        self.seq_len = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Allocate key and value caches
        self.key_cache = torch.zeros(
            self.num_blocks, self.num_heads, self.head_size // 8, self.block_size, 8, 
            dtype=torch.float16, device=self.device
        )
        self.value_cache = torch.zeros(
            self.num_blocks, self.num_heads, self.head_size, self.block_size, 
            dtype=torch.float16, device=self.device
        )

    def generate_random_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        key = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, 
            dtype=torch.float16, device=self.device
        )
        value = torch.randn(
            self.batch_size, self.seq_len, self.num_heads, self.head_size, 
            dtype=torch.float16, device=self.device
        )
        return key, value

    def cache_and_reshape_kv(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        # Generate slot_mapping
        slot_mapping = torch.arange(self.seq_len, dtype=torch.long, device=self.device)
        slot_mapping = slot_mapping.repeat(self.batch_size)

        # Reshape key and value for cache_ops.reshape_and_cache
        key = key.reshape(-1, self.num_heads, self.head_size)
        value = value.reshape(-1, self.num_heads, self.head_size)

        print("slot_mapping:", slot_mapping)
        print("k.shape:", key.shape)
        print("v.shape:", value.shape)
        print("key_cache.shape:", self.key_cache.shape)
        print("value_cache.shape:", self.value_cache.shape)

        # Call cache_ops.reshape_and_cache
        cache_ops.reshape_and_cache(
            key, value, self.key_cache, self.value_cache, slot_mapping, 
            "auto", 1.0  # kv_cache_dtype and kv_scale
        )

        # Generate block_table
        block_table = [[0, -1, -1, -1]]

        return block_table, slot_mapping

    def verify_cache_and_reshape(self, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor):
        print("slot_mapping:", slot_mapping)
        for b in range(self.batch_size):
            for h in range(self.num_heads):
                for i in range(self.seq_len):
                    slot = slot_mapping[b * self.seq_len + i].item()
                    block_idx = slot // self.block_size
                    block_offset = slot % self.block_size

                    # Verify key
                    cached_key = self.key_cache[block_idx, h, :, block_offset, :].reshape(self.head_size)
                    original_key = key[b, i, h]
                    self.assertTrue(torch.allclose(cached_key, original_key, atol=1e-3), 
                                    f"Mismatch in key at batch {b}, head {h}, seq {i}")

                    # Verify value
                    cached_value = self.value_cache[block_idx, h, :, block_offset]
                    original_value = value[b, i, h]
                    self.assertTrue(torch.allclose(cached_value, original_value, atol=1e-3), 
                                    f"Mismatch in value at batch {b}, head {h}, seq {i}")

    def test_paged_attention_correctness(self):
        # Generate random keys and values
        key, value = self.generate_random_kv()

        # Cache and reshape keys and values
        block_table, slot_mapping = self.cache_and_reshape_kv(key, value)

        # Verify that cache_and_reshape worked correctly
        self.verify_cache_and_reshape(key, value, slot_mapping)
        print("verify_cache_and_reshape passed...")


        # Generate query
        query = torch.randn(
            self.batch_size, 1, self.num_heads, self.head_size, 
            dtype=torch.float16, device=self.device
        )

        # Compute vanilla attention
        scale = 1.0 / (self.head_size ** 0.5)
        key_for_vanilla = key.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        value_for_vanilla = value.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        query_for_vanilla = query.transpose(1, 2)  # [batch_size, num_heads, 1, head_size]
        
        attn_weights = torch.matmul(query_for_vanilla, key_for_vanilla.transpose(-1, -2)) * scale
        attn_probs = torch.softmax(attn_weights, dim=-1)
        vanilla_output = torch.matmul(attn_probs, value_for_vanilla)

        # Compute paged attention
        seq_lens = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=self.device)
        paged_output = torch.empty_like(vanilla_output)
        
        paged_attention_v1(
            paged_output,
            query.reshape(-1, self.num_heads, self.head_size),
            self.key_cache,
            self.value_cache,
            self.num_heads,
            scale,
            torch.tensor(block_table, dtype=torch.int32, device=self.device),
            seq_lens,
            self.block_size,
            self.seq_len,
            None,  # alibi_slopes
            "auto",
            1.0,  # kv_scale
            0,  # tp_rank
            0,  # blocksparse_local_blocks
            1,  # blocksparse_vert_stride
            1,  # blocksparse_block_size
            0,  # blocksparse_head_sliding_step
        )

        # Compare outputs
        self.assertTrue(torch.allclose(vanilla_output, paged_output, atol=1e-2), 
                        "Mismatch between vanilla attention and paged attention outputs")

if __name__ == "__main__":
    unittest.main()