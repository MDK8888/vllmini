import time
from queue import PriorityQueue
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from .block_manager import BlockManager
from vllmini.model.gpt2 import GPT2LMHeadModel
from vllmini.model.helpers.generate_triangular_mask import generate_triangular_mask

class Scheduler:
    def __init__(self, model:GPT2LMHeadModel, block_manager:BlockManager, max_length: int):
        self.model = model
        self.model = self.model.to(torch.float16)
        self.block_manager = block_manager
        self.max_length = max_length
        self.queue = PriorityQueue()
        self.active_sequences: Dict[int, float] = {}  # seq_id: arrival_time
        self.last_logits: Dict[int, torch.Tensor] = {}
        self.sequence_lengths: Dict[int, int] = {}

    def add_sequence(self, input_ids: torch.Tensor):
        arrival_time = time.time()
        seq_id = self._generate_seq_id()
        self.queue.put((arrival_time, seq_id))
        self.active_sequences[seq_id] = arrival_time
        
        seq_len = input_ids.size(1)
        num_layers = len(self.model.transformer.h)
        
        # Allocate blocks and perform initial prefill
        seq_id, _, slot_mappings, paged_attention_block_table = self.block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
        
        key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache
        
        attention_mask = generate_triangular_mask(1, self.block_manager.num_heads, seq_len)

        logits, _ = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(seq_len, device=input_ids.device),
            attention_mask=attention_mask,
            use_cache=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=paged_attention_block_table,
        )
        
        self.last_logits[seq_id] = logits[:, -1, :]
        self.sequence_lengths[seq_id] = seq_len
        print(f"Prefill complete for sequence {seq_id}, length: {seq_len}")

    def run(self):
        while not self.queue.empty():
            print(f"Active sequences: {self.active_sequences}")
            print(f"Queue size: {self.queue.qsize()}")
            
            _, seq_id = self.queue.get()
            print(f"Processing sequence {seq_id}")

            if seq_id not in self.active_sequences:
                print(f"Sequence {seq_id} is no longer active, skipping")
                continue

            try:
                print("current sequence_lengths in run:", self.sequence_lengths)
                print(f"Processing sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                
                if self.sequence_lengths[seq_id] >= self.max_length:
                    print(f"Sequence {seq_id} has reached max_length, ending generation")
                    self.remove_sequence(seq_id)
                    continue

                next_token = self.sample_next_token(seq_id)
                
                input_ids = next_token.unsqueeze(0)
                position_ids = torch.tensor([self.sequence_lengths[seq_id]], device=input_ids.device)
                
                paged_attention_block_table, new_slot_mappings = self.block_manager.decode_step(seq_id, 1)
                key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache

                logits, _ = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                    use_cache=True,
                    is_prefill=False,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mappings=new_slot_mappings,
                    block_tables=paged_attention_block_table,
                    seq_lens=torch.tensor([self.sequence_lengths[seq_id]], dtype=torch.int32, device=input_ids.device),
                    max_seq_len=self.block_manager.kv_cache.max_blocks_per_seq * self.block_manager.block_size
                )

                self.last_logits[seq_id] = logits[:, -1, :]
                self.sequence_lengths[seq_id] += 1

                if next_token.item() != self.model.config.eos_token_id and self.sequence_lengths[seq_id] < self.max_length:
                    self.queue.put((self.active_sequences[seq_id], seq_id))
                    print(f"Re-queued sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                else:
                    print(f"Sequence {seq_id} completed or reached max_length, final length: {self.sequence_lengths[seq_id]}")
                    self.remove_sequence(seq_id)

            except RuntimeError as e:
                print(f"Error processing sequence {seq_id}: {str(e)}")
                if "CUDA out of memory" in str(e):
                    self.handle_out_of_memory([seq_id])
                else:
                    raise e

    def handle_out_of_memory(self, batch_seq_ids: List[int]):
        print("Handling out of memory")
        if self.active_sequences:
            seq_to_remove = max(
                (seq for seq in self.active_sequences if seq not in batch_seq_ids),
                key=self.active_sequences.get,
                default=None
            )
            if seq_to_remove is None:
                seq_to_remove = max(self.active_sequences, key=self.active_sequences.get)
            print(f"Removing sequence {seq_to_remove} due to memory constraints")
            self.remove_sequence(seq_to_remove)
        else:
            print("No active sequences to remove")

    def remove_sequence(self, seq_id: int):
        self.block_manager.free(seq_id)
        del self.active_sequences[seq_id]
        if seq_id in self.last_logits:
            del self.last_logits[seq_id]
        if seq_id in self.sequence_lengths:
            del self.sequence_lengths[seq_id]

    def sample_next_token(self, seq_id: int) -> torch.Tensor:
        logits = self.last_logits[seq_id]
        temperature = 1.0
        logits = logits / temperature
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[0, next_token_index[0]]
        return next_token

    def _generate_seq_id(self) -> int:
        return int(time.time() * 1000000)