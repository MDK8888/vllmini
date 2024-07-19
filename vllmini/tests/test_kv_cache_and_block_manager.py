import unittest
import torch
from vllmini.kv_cache import KVCache
from vllmini.block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config

class TestKVCacheAndBlockManager(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 100
        self.num_heads = 12
        self.head_size = 64
        self.block_size = 16
        self.max_blocks_per_seq = 4
        self.block_manager = BlockManager(self.num_blocks, self.block_size, self.num_heads, self.head_size, self.max_blocks_per_seq)
        
        # Set up GPT2 model and tokenizer
        config = GPT2Config.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel(config)
        self.model.load_huggingface_weights("gpt2")
        self.model = self.model.to("cuda")
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def test_allocate_for_prefill(self):
        group_id = 1
        num_layers = 12
        seq_len = 10
        seq_id, allocated, slot_mappings, paged_attention_block_table = self.block_manager.allocate_for_prefill(group_id, num_layers, seq_len)
        
        self.assertEqual(len(allocated), num_layers)
        self.assertEqual(len(slot_mappings), num_layers)
        self.assertEqual(len(paged_attention_block_table), num_layers)

    def test_decoding_with_gpt2(self):
        prompt = "Hello, how"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        
        group_id = 1
        num_layers = len(self.model.transformer.h)
        seq_len = input_ids.size(1)
        
        # Prefill phase
        key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache
        seq_id, allocated, slot_mappings, paged_attention_block_table = self.block_manager.allocate_for_prefill(group_id, num_layers, seq_len)
        
        print(f"Initial allocation: {allocated}")
        print(f"Initial paged_attention_block_table: {paged_attention_block_table}")
        
        # Prefill
        logits, kv_cache = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(seq_len, device="cuda"),
            attention_mask=None,
            use_cache=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=None,
        )
        
        generated = input_ids
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        
        print("Prefill completed. Starting decoding phase.")
        
        # Get the first token for decoding phase from the last logits of prefill
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=1)
        
        # Decoding phase
        for i in range(19):  # 19 because we've already generated one token
            paged_attention_block_table, new_slot_mappings = self.block_manager.decode_step(seq_id, 1)

            logits, kv_cache = self.model(
                input_ids=next_token,
                position_ids=torch.tensor([[generated.size(1) - 1]], device="cuda"),
                attention_mask=None,
                use_cache=True,
                is_prefill=False,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=new_slot_mappings,
                block_tables=paged_attention_block_table,
                seq_lens=torch.tensor([generated.size(1)], dtype=torch.int32, device="cuda"),
                max_seq_len=self.max_blocks_per_seq
            )

            key_cache, value_cache = kv_cache[0], kv_cache[1]
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
            
            print(f"Step {i+1}:")
            print(f"Generated token: {self.tokenizer.decode(next_token.item())}")
            print(f"logits: {logits}")
            print("---")
        
        generated_text = self.tokenizer.decode(generated[0])
        print(f"Full generated text: {generated_text}")
        
        self.block_manager.free(group_id)
        print("Blocks freed")

if __name__ == "__main__":
    unittest.main()