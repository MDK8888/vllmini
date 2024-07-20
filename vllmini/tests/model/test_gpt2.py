import torch
import unittest
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config

def generate_triangular_mask(batch_size, num_heads, seq_len):
    # Create an upper triangular matrix with -inf, including the diagonal
    upper_triangular = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    
    # Expand the upper triangular matrix to match the desired shape
    mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # shape (batch_size, num_heads, seq_len, seq_len)
    
    return mask

class TestGPT2WithPagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = GPT2Config.from_pretrained('gpt2')
        cls.model = GPT2LMHeadModel(cls.config)
        cls.model.load_huggingface_weights("gpt2")
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model.to(cls.device)

    def test_prefill_stage(self):
        input_text = "Hello, how are you?"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        # Prepare cache tensors for prefill
        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        block_size = self.model.transformer.h[0].attn.block_size
        num_blocks = 1024
        #max_num_blocks_per_seq = num_blocks * num_layers

        key_cache = torch.zeros(num_blocks, num_heads, head_size // 8, block_size, 8, dtype=torch.float16, device=self.device)
        value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping
            )

        self.assertIsNotNone(outputs[0])  # Check if logits are produced
        self.assertIsNotNone(outputs[1])  # Check if key-value cache is produced
        self.assertEqual(len(outputs[1]), self.config.num_hidden_layers)  # Check if cache is produced for all layers
        
        # Check if caches are populated
        self.assertFalse(torch.all(key_cache == 0))
        self.assertFalse(torch.all(value_cache == 0))
    
    def test_prefill_and_decode_one_token(self):
        input_text = "Hello, how"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        # Prepare cache tensors
        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        block_size = self.model.transformer.h[0].attn.block_size
        num_blocks = 1024

        attention_mask = generate_triangular_mask(1, num_heads, seq_len)
        attention_mask = attention_mask.to(self.device)

        key_cache = torch.zeros(num_blocks, num_heads, head_size // 8, block_size, 8, dtype=torch.float16, device=self.device)
        value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)
        
        # Prefill phase
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        with torch.no_grad():
            prefill_outputs, kv_cache = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                is_prefill=True,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping
            )

        self.assertIsNotNone(prefill_outputs[0])  # Check if logits are produced
        self.assertIsNotNone(prefill_outputs[1])  # Check if key-value cache is produced
        self.assertEqual(len(prefill_outputs[1]), self.config.num_hidden_layers)  # Check if cache is produced for all layers
        
        # Check if caches are populated
        self.assertFalse(torch.all(key_cache == 0))
        self.assertFalse(torch.all(value_cache == 0))

        # Sample from the last logits to get the next token
        last_token_logits = prefill_outputs[0][0, -1, :]
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        # Decoding phase (one token)
        current_length = input_ids.size(1)
        position_ids = torch.tensor([current_length], dtype=torch.long, device=self.device)
        
        # Prepare slot mapping for decoding
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.tensor([seq_len], dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        # Prepare block tables
        block_tables = []
        for i in range(num_layers):
            block_table = torch.tensor([i, -1, -1, -1]).unsqueeze(0).to(dtype=torch.int32, device=self.device)
            block_tables.append(block_table)

        # Prepare sequence lengths
        seq_lens = torch.tensor([current_length], dtype=torch.int32, device=self.device)
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        
        with torch.no_grad():
            decode_outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                position_ids=position_ids,
                attention_mask=None,
                use_cache=True,
                is_prefill=False,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mappings=slot_mapping,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=block_size
            )

        last_token_logits = decode_outputs[0][0]
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)

        # Check if decoding produced one new token
        self.assertEqual(generated_ids.size(1), input_ids.size(1) + 2)

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_ids[0])
        print(f"Input text: {input_text}")
        print(f"Generated text: {generated_text}")

        # Additional checks
        self.assertNotEqual(input_text, generated_text)  # Ensure something was generated
        self.assertTrue(generated_text.startswith(input_text))

if __name__ == '__main__':
    unittest.main()