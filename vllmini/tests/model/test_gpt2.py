import torch
import unittest
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config

class TestGPT2WithPagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = GPT2Config.from_pretrained('gpt2')
        cls.model = GPT2LMHeadModel(cls.config)
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

    def test_prefill_stage(self):
        input_text = "Hello, how are you?"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True
            )

        self.assertIsNotNone(outputs[0])  # Check if logits are produced
        self.assertIsNotNone(outputs[1])  # Check if key-value cache is produced
        self.assertEqual(len(outputs[1]), self.config.num_hidden_layers)  # Check if cache is produced for all layers

    def test_decoding_with_paged_attention(self):
        # Prefill stage
        input_text = "Hello, how are"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True
            )
        #input_text has 4 tokens, we have 12 layers, so we need to track 48 key and value vectors. Given that our block_size is 16, we only need three blocks. 
        #the keys and values for the first 4 layers should occupy the first block, layers 5 through 8 the second block, and 9 through 12 the third block. 

        # Prepare for decoding
        seq_len = input_ids.size(1)
        block_size = self.model.transformer.h[0].attn.block_size
        max_seq_len = block_size
        num_blocks = 1024  # As specified
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        x = 8  # As specified for key_cache

        # Preallocate key_cache and value_cache as single tensors
        key_cache = torch.zeros(num_blocks, num_heads, head_size // x, block_size, x, dtype=torch.float16, device=self.device)
        value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)

        # Create slot_mappings and block_tables for every layer
        slot_mappings = [
            torch.arange(seq_len * layer, seq_len * (layer + 1), dtype=torch.long).unsqueeze(0).to(self.device)
            for layer in range(num_layers)
        ]
        block_tables = []
        max_num_blocks_per_seq = 1
        for i in range(num_layers):
            layer_block = torch.zeros(max_num_blocks_per_seq, dtype=torch.int32, device=self.device)
            layer_block[0] = i
            block_tables.append(layer_block.unsqueeze(0))

        #block_tables = [torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0).to(self.device) for _ in range(num_layers)]
        seq_lens = torch.tensor([seq_len], dtype=torch.int32).to(self.device)

        # Sample next token from the last logits
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).unsqueeze(0)
        next_position_id = position_ids[:, -1:] + 1

        # Decoding stage
        with torch.no_grad():
            new_logits, _ = self.model(
                input_ids=next_token_id,
                position_ids=next_position_id,
                attention_mask=None,
                use_cache=True,
                is_prefill=False,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mappings,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len
            )
        self.assertIsNotNone(new_logits)  # Check if logits are produced
        self.assertEqual(new_logits.size(), (1, 1, self.config.vocab_size))  # Check if logits have the correct shape
        
        # Check if the caches are updated
        self.assertFalse(torch.all(key_cache == 0))
        self.assertFalse(torch.all(value_cache == 0))

        # Decode the next token
        next_token = self.tokenizer.decode(next_token_id[0])
        print(f"Input: '{input_text}', Next token: '{next_token}'")

        # Additional checks
        self.assertEqual(key_cache.shape, (num_blocks, num_heads, head_size // x, block_size, x))
        self.assertEqual(value_cache.shape, (num_blocks, num_heads, head_size, block_size))
        
        # Check slot_mappings and block_tables
        self.assertEqual(len(slot_mappings), num_layers)
        self.assertEqual(len(block_tables), num_layers)
        for layer in range(num_layers):
            self.assertEqual(slot_mappings[layer].shape, (1, seq_len))
            self.assertEqual(block_tables[layer].shape, (1, max_num_blocks_per_seq))

if __name__ == '__main__':
    unittest.main()