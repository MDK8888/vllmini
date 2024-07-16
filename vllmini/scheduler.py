import time
from queue import PriorityQueue
from typing import List, Dict
import torch
import torch.nn.functional as F
from .kv_cache import KVCache
from .paged_attention import PagedAttention

class Scheduler:
    def __init__(self, paged_attention: PagedAttention, max_length: int):
        self.paged_attention = paged_attention
        self.kv_cache = paged_attention.kv_cache
        self.max_length = max_length
        self.queue = PriorityQueue()  # Use PriorityQueue for FCFS ordering
        self.active_sequences: Dict[int, float] = {}  # Dict to store active sequences and their arrival times
        self.preempted_sequences = PriorityQueue()  # Store preempted sequences
        self.block_tables: Dict[int, List[int]] = {}  # Store block tables for each sequence
        self.last_logits: Dict[int, torch.Tensor] = {}  # Store last logits for each sequence
        self.sequence_lengths: Dict[int, int] = {}  # Store current length of each sequence

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
        output = self.paged_attention.forward(
            input_ids,
            position_ids,
            attention_mask,
            use_cache=True,
            is_prefill=True,
            seq_id=seq_id,
            slot_mapping=slot_mapping
        )
        self.last_logits[seq_id] = output[:, -1, :]  # Store only the last token's logits
        self.sequence_lengths[seq_id] = seq_len  # Initialize sequence length
        print(f"prefill complete for sequence {seq_id}, length: {seq_len}")

    def remove_sequence(self, seq_id: int):
        del self.active_sequences[seq_id]
        del self.block_tables[seq_id]
        if seq_id in self.last_logits:
            del self.last_logits[seq_id]
        if seq_id in self.sequence_lengths:
            del self.sequence_lengths[seq_id]
        self.kv_cache.free(seq_id)

    def run(self):
        while not self.queue.empty() or not self.preempted_sequences.empty():
            print(f"Current free blocks: {len(self.kv_cache.free_blocks)}")
            print(f"Active sequences: {self.active_sequences}")
            print(f"Queue size: {self.queue.qsize()}")
            print(f"Preempted sequences: {self.preempted_sequences.qsize()}")
            
            # Get the next sequence to process
            if not self.preempted_sequences.empty():
                _, seq_id = self.preempted_sequences.get()
                print(f"Processing preempted sequence {seq_id}")
            elif not self.queue.empty():
                _, seq_id = self.queue.get()
                print(f"Processing queued sequence {seq_id}")
            else:
                print("No sequences to process")
                break

            if seq_id not in self.active_sequences:
                print(f"Sequence {seq_id} is no longer active, skipping")
                continue

            try:
                print(f"Preparing sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                
                # Check if we've reached max_length
                if self.sequence_lengths[seq_id] >= self.max_length:
                    print(f"Sequence {seq_id} has reached max_length, ending generation")
                    self.remove_sequence(seq_id)
                    continue

                # Sample the next token using the last logits
                next_token = self.sample_next_token(seq_id)
                
                # Prepare input for this sequence (single token)
                input_ids = next_token.unsqueeze(0)
                position_ids = torch.tensor([[self.sequence_lengths[seq_id]]], dtype=torch.int64, device=self.paged_attention.device)
                attention_mask = None  # Remove attention mask for single-token input
                slot_mapping = torch.tensor([self.sequence_lengths[seq_id]], dtype=torch.int64, device=self.paged_attention.device)
                print("all variables initialized...")

                # Check if we have enough free blocks
                if len(self.kv_cache.free_blocks) == 0:
                    print(f"No free blocks available, attempting to free some")
                    self.handle_out_of_memory([seq_id])
                    if len(self.kv_cache.free_blocks) == 0:
                        print(f"Failed to free any blocks, skipping sequence {seq_id}")
                        continue

                output = self.paged_attention.forward(
                    input_ids,
                    position_ids,
                    attention_mask,
                    use_cache=True,
                    is_prefill=False,
                    seq_id=seq_id,
                    slot_mapping=slot_mapping
                )

                print("forward pass finished...")
                # Store the new logits
                self.last_logits[seq_id] = output[:, -1, :]

                # Increment sequence length
                self.sequence_lengths[seq_id] += 1

                # Check if the sequence is finished
                if next_token.item() != self.paged_attention.model.config.eos_token_id and self.sequence_lengths[seq_id] < self.max_length:
                    # Continue sequence
                    if seq_id in self.active_sequences:
                        self.queue.put((self.active_sequences[seq_id], seq_id))
                        print(f"Re-queued sequence {seq_id}, current length: {self.sequence_lengths[seq_id]}")
                        # Update block table if necessary
                        if len(self.block_tables[seq_id]) * self.kv_cache.block_size <= self.sequence_lengths[seq_id]:
                            new_block = self.kv_cache.allocate(seq_id, 1)[0]
                            self.block_tables[seq_id].append(new_block)
                            print(f"Allocated new block for sequence {seq_id}")
                    else:
                        print(f"Sequence {seq_id} was preempted during processing, not re-queueing")
                else:
                    # End of sequence
                    self.remove_sequence(seq_id)
                    print(f"Sequence {seq_id} completed or reached max_length, final length: {self.sequence_lengths[seq_id]}")

            except RuntimeError as e:
                print(f"Error processing sequence {seq_id}: {str(e)}")
                if "Not enough free blocks" in str(e):
                    self.handle_out_of_memory([seq_id])
                else:
                    raise e
                
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
        self.kv_cache._swap_out()
        # Move the sequence to the preempted queue
        if seq_id in self.active_sequences:
            self.preempted_sequences.put((self.active_sequences[seq_id], seq_id))
            # Remove from active sequences, but keep the block table
            del self.active_sequences[seq_id]
        # Ensure the blocks are actually freed
        self.kv_cache.free_blocks.extend(blocks_to_free)
        print(f"Finished preempting sequence {seq_id}. Freed {len(blocks_to_free)} blocks.")

    def sample_next_token(self, seq_id: int) -> torch.Tensor:
        logits = self.last_logits[seq_id]
        # Apply temperature (optional)
        temperature = 1.0
        logits = logits / temperature
        # Apply top-k sampling (optional)
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        # Sample from the distribution
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[0, next_token_index[0]]
        return next_token