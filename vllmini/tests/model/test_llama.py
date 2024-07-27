import torch
import unittest
from vllmini.model.llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")

class TestLlamaWithPagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        cls.config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir)
        
        # Load tokenizer
        cls.tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir)
        
        print("Initializing custom model...")
        cls.model = LlamaForCausalLM(cls.config)
        
        print("Loading weights into custom model...")
        cls.model.load_huggingface_weights('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir)
        
        print("Moving model to GPU and converting to half precision...")
        cls.model = cls.model.half().to(cls.device)
        
        torch.cuda.empty_cache()
        print("Model setup complete.")

    def test_prefill_stage(self):
        input_text = "Hello, how are you?"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        block_size = 16  # Assuming block size is 16
        num_blocks = 1024

        key_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)
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
        # self.assertFalse(torch.all(key_cache == 0))
        # self.assertFalse(torch.all(value_cache == 0))

    def test_prefill_and_decode_one_token(self):
        input_text = "Hello, how"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(self.device)

        seq_len = input_ids.size(1)
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        head_size = self.config.hidden_size // num_heads
        block_size = 16  # Assuming block size is 16
        num_blocks = 1024

        key_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)
        value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device=self.device)
        
        slot_mapping = []
        for i in range(num_layers):
            layer_slot_mapping = torch.arange(seq_len, dtype=torch.long, device=self.device) + i * block_size
            slot_mapping.append(layer_slot_mapping)

        with torch.no_grad():
            prefill_outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
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
        # self.assertFalse(torch.all(key_cache == 0))
        # self.assertFalse(torch.all(value_cache == 0))

        # Sample from the last logits to get the next token
        last_token_logits = prefill_outputs[0][0, -1, :]
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
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