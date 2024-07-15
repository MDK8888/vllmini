import time
from queue import PriorityQueue
from typing import List, Dict
import torch
from .kv_cache import KVCache
from .paged_attention import PagedAttention

class Scheduler:
    def __init__(self, paged_attention: PagedAttention):
        self.paged_attention = paged_attention
        self.kv_cache = paged_attention.kv_cache
        self.queue = PriorityQueue()  # Use PriorityQueue for FCFS ordering
        self.active_sequences: Dict[int, float] = {}  # Dict to store active sequences and their arrival times
        self.preempted_sequences = PriorityQueue()  # Store preempted sequences
        self.block_tables: Dict[int, List[int]] = {}  # Store block tables for each sequence

    def add_sequence(self, seq_id: int, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor):
        arrival_time = time.time()
        self.queue.put((arrival_time, seq_id))
        self.active_sequences[seq_id] = arrival_time
        
        # Allocate blocks and create block table for the new sequence
        seq_len = input_ids.size(1)
        num_blocks_needed = (seq_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        allocated_blocks = self.kv_cache.allocate(seq_id, num_blocks_needed)
        self.block_tables[seq_id] = allocated_blocks

        # Perform initial prefill
        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=self.paged_attention.device)
        self.paged_attention.forward(
            input_ids,
            position_ids,
            attention_mask,
            use_cache=True,
            is_prefill=True,
            seq_id=seq_id,
            slot_mapping=slot_mapping
        )

    def remove_sequence(self, seq_id: int):
        del self.active_sequences[seq_id]
        del self.block_tables[seq_id]
        self.kv_cache.free(seq_id)

    def run(self):
        while not self.queue.empty() or not self.preempted_sequences.empty():
            if not self.preempted_sequences.empty():
                _, seq_id = self.preempted_sequences.get()
            elif not self.queue.empty():
                _, seq_id = self.queue.get()
            else:
                break

            if seq_id in self.active_sequences:
                self.process_next_token(seq_id)

            print("Finished processing all sequences")
    
    def process_next_token(self, seq_id):
        global sequences
        seq = sequences[seq_id]
        
        if len(seq["tokens"]) >= seq["max_length"]:
            seq["completed"] = True
            self.remove_sequence(seq_id)
            return

        input_ids = torch.tensor([[seq["tokens"][-1]]], dtype=torch.int64, device=self.paged_attention.device)
        position_ids = torch.tensor([[len(seq["tokens"]) - 1]], dtype=torch.int64, device=self.paged_attention.device)
        attention_mask = torch.ones((1, 1, 1, len(seq["tokens"])), dtype=torch.float32, device=self.paged_attention.device)
        slot_mapping = torch.tensor([len(seq["tokens"])], dtype=torch.int64, device=self.paged_attention.device)

        output = self.paged_attention.forward(
            input_ids,
            position_ids,
            attention_mask,
            use_cache=True,
            is_prefill=False,
            seq_id=seq_id,
            slot_mapping=slot_mapping
        )

        next_token = output.argmax(dim=-1).item()
        seq["tokens"].append(next_token)

        global tokenizer
        if next_token == tokenizer.eos_token_id:
            seq["completed"] = True
            self.remove_sequence(seq_id)
        else:
            self.add_sequence(seq_id, input_ids, position_ids, attention_mask, slot_mapping)

    def handle_out_of_memory(self, batch_seq_ids: List[int]):
        print(f"Handling out of memory. Free blocks: {len(self.kv_cache.free_blocks)}")
        max_attempts = len(self.active_sequences)  # Limit the number of preemption attempts
        attempts = 0
        while len(self.kv_cache.free_blocks) < 1 and attempts < max_attempts:  # We need at least one free block
            if self.active_sequences:
                # Preempt the oldest sequence that's not in the current batch
                seq_to_preempt = max(
                    (seq for seq in self.active_sequences if seq not in batch_seq_ids),
                    key=self.active_sequences.get,
                    default=None
                )
                if seq_to_preempt is None:
                    # If all sequences in active_sequences are in the batch, preempt the oldest one
                    seq_to_preempt = max(self.active_sequences, key=self.active_sequences.get)
                print(f"Attempting to preempt sequence {seq_to_preempt}")
                self.preempt_sequence(seq_to_preempt)
            else:
                print("No active sequences to preempt")
                break
            attempts += 1
            print(f"After attempt {attempts}, free blocks: {len(self.kv_cache.free_blocks)}")
        
        if len(self.kv_cache.free_blocks) < 1:
            print(f"Failed to free blocks after {max_attempts} attempts")
        else:
            print(f"Successfully freed blocks. Current free blocks: {len(self.kv_cache.free_blocks)}")

    def preempt_sequence(self, seq_id: int):
        print(f"Preempting sequence {seq_id}")
        # Get the blocks associated with this sequence
        blocks_to_free = self.block_tables.get(seq_id, [])
        # Swap out all blocks of the sequence
        self.kv_cache._swap_out()  # No seq_id parameter
        # Move the sequence to the preempted queue
        if seq_id in self.active_sequences:
            self.preempted_sequences.put((self.active_sequences[seq_id], seq_id))
            # Remove from active sequences, but keep the block table
            del self.active_sequences[seq_id]
        # Ensure the blocks are actually freed
        self.kv_cache.free_blocks.extend(blocks_to_free)
        print(f"Finished preempting sequence {seq_id}. Freed {len(blocks_to_free)} blocks.")