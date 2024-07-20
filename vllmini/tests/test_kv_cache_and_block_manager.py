import unittest
import torch
from vllmini.kv_cache import KVCache
from vllmini.block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config

def generate_triangular_mask(batch_size, num_heads, seq_len):
    # Create an upper triangular matrix with -inf, including the diagonal
    upper_triangular = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float16), diagonal=1)
    
    # Expand the upper triangular matrix to match the desired shape
    mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # shape (batch_size, num_heads, seq_len, seq_len)
    mask = mask.to("cuda", dtype=torch.float16)
    return mask

class TestKVCacheAndBlockManager(unittest.TestCase):
    def setUp(self):
        self.num_blocks = 1000
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
        
        attention_mask = generate_triangular_mask(1, 12, seq_len)

        self.model = self.model.to(torch.float16)

        # Prefill
        logits, presents = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(seq_len, device="cuda").unsqueeze(0),
            attention_mask=attention_mask,
            use_cache=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=None,
        )
        generated = input_ids
        print("logits after prefill:", logits)

        #make sure that the keys and values have been properly allocated.        
        for i in range(len(presents)):
            k, v = presents[i] #they each have shape (batch_size, num_heads, seq_len, head_dim)
            layer_slot_mapping = slot_mappings[i]
            for b in range(1):
                for h in range(self.num_heads):
                    for s in range(seq_len):
                        block_idx = layer_slot_mapping[s] // self.block_size
                        block_offset = layer_slot_mapping[s] % self.block_size

                        key_vector_from_global_cache = key_cache[block_idx, h, :, block_offset, :].reshape(self.head_size)
                        value_vector_from_global_cache = value_cache[block_idx, h, :, block_offset]

                        key = k[b, h, s]
                        value = v[b, h, s]

                        self.assertTrue(torch.allclose(key_vector_from_global_cache, key.half(), atol=1e-3), 
                                    f"Mismatch in key at batch {b}, head {h}, seq {i}")

                        self.assertTrue(torch.allclose(value_vector_from_global_cache, value.half(), atol=1e-3), 
                                    f"Mismatch in key at batch {b}, head {h}, seq {i}")

        print("Prefill looks good. Starting decoding phase.")

        # Get the first token for decoding phase from the last logits of prefill
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=-1)
        # Decoding phase
        paged_attention_block_table, new_slot_mappings = self.block_manager.decode_step(seq_id, 1)
        
        for i in range(19):

            paged_attention_block_table, new_slot_mappings = self.block_manager.decode_step(seq_id, 1)

            logits, _ = self.model(
                input_ids=next_token,
                position_ids=torch.tensor([generated.size(1) - 1], device="cuda"),
                attention_mask=None,
                use_cache=True,
                is_prefill=False,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=new_slot_mappings,
                block_tables=paged_attention_block_table,
                seq_lens=torch.tensor([generated.size(1) - 1], dtype=torch.int32, device="cuda"),
                max_seq_len=self.max_blocks_per_seq
            )
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
        
        print("length of generated:", generated.size(1))
        generated_text = self.tokenizer.decode(generated[0])
        print(f"Full generated text: {generated_text}")
        
        self.block_manager.free(group_id)
        print("Blocks freed")

if __name__ == "__main__":
    unittest.main()