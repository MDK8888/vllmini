import unittest
import torch
from vllmini.scheduler import Scheduler
from vllmini.block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from transformers import GPT2Config, GPT2Tokenizer

class TestScheduler(unittest.TestCase):
    def setUp(self):
        # Initialize the necessary components
        config = GPT2Config.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel(config)
        self.model.load_huggingface_weights("gpt2")
        self.model = self.model.to("cuda")
        self.model.eval()

        num_blocks = 1000
        num_heads = 12
        head_size = 64
        block_size = 16
        max_blocks_per_seq = 4
        self.block_manager = BlockManager(num_blocks, block_size, num_heads, head_size, max_blocks_per_seq)

        max_length = 20
        self.scheduler = Scheduler(self.model, self.block_manager, max_length)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def test_add_sequence(self):
        prompt = "Hello, how are you?"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones_like(input_ids)

        initial_queue_size = self.scheduler.queue.qsize()
        self.scheduler.add_sequence(input_ids)

        self.assertEqual(self.scheduler.queue.qsize(), initial_queue_size + 1)
        self.assertEqual(len(self.scheduler.active_sequences), 1)
        self.assertEqual(len(self.scheduler.last_logits), 1)
        self.assertEqual(len(self.scheduler.sequence_lengths), 1)

    def test_run_single_sequence(self):
        prompt = "Hello, how are you?"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        self.scheduler.add_sequence(input_ids)
        self.scheduler.run()

        self.assertEqual(self.scheduler.queue.qsize(), 0)
        self.assertEqual(len(self.scheduler.active_sequences), 0)


    def test_run_multiple_sequences(self):
        prompts = ["Hello, how are you?", "What's the weather like today?", "Tell me a joke."]
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            self.scheduler.add_sequence(input_ids)

        self.scheduler.run()

        self.assertEqual(self.scheduler.queue.qsize(), 0)
        self.assertEqual(len(self.scheduler.active_sequences), 0)

    def test_max_length(self):
        prompt = "This is a very short prompt."
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones_like(input_ids)

        self.scheduler.add_sequence(input_ids)
        self.scheduler.run()

        for seq_len in self.scheduler.sequence_lengths.values():
            self.assertLessEqual(seq_len, self.scheduler.max_length)

    def test_remove_sequence(self):
        prompt = "Test prompt"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones_like(input_ids)

        self.scheduler.add_sequence(input_ids)
        seq_id = list(self.scheduler.active_sequences.keys())[0]

        self.scheduler.remove_sequence(seq_id)

        self.assertNotIn(seq_id, self.scheduler.active_sequences)
        self.assertNotIn(seq_id, self.scheduler.last_logits)
        self.assertNotIn(seq_id, self.scheduler.sequence_lengths)

    def test_sample_next_token(self):
        prompt = "The quick brown fox"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")

        self.scheduler.add_sequence(input_ids)
        seq_id = list(self.scheduler.active_sequences.keys())[0]

        next_token = self.scheduler.sample_next_token(seq_id)

        self.assertIsInstance(next_token, torch.Tensor)
        self.assertEqual(next_token.shape, torch.Size([1]))

if __name__ == '__main__':
    unittest.main()