import time
from queue import Queue, PriorityQueue
from .kv_cache import KVCache
from .paged_attention import PagedAttention
from .sequence_manager import SequenceManager

class Scheduler:
    def __init__(self, paged_attention: PagedAttention, sequence_manager: SequenceManager):
        self.paged_attention = paged_attention
        self.paged_attention.seq_manager = sequence_manager
        self.queue = PriorityQueue()  # Use PriorityQueue for FCFS ordering
        self.active_sequences = {}  # Dict to store active sequences and their arrival times
        self.preempted_sequences = PriorityQueue()  # Store preempted sequences

    def add_sequence(self, seq_id: int):
        arrival_time = time.time()
        self.queue.put((arrival_time, seq_id))
        self.active_sequences[seq_id] = arrival_time

    def remove_sequence(self, seq_id: int):
        del self.active_sequences[seq_id]
        self.paged_attention.seq_manager.remove_sequence(seq_id)
        self.paged_attention.kv_cache.free(seq_id)

    def run(self):
        while True:
            # First, try to resume any preempted sequences
            while not self.preempted_sequences.empty():
                _, seq_id = self.preempted_sequences.get()
                if self.process_sequence(seq_id):
                    break
            else:
                # If no preempted sequences, get the next sequence from the main queue
                if not self.queue.empty():
                    _, seq_id = self.queue.get()
                    self.process_sequence(seq_id)

            self.queue.task_done()

    def process_sequence(self, seq_id: int):
        if seq_id in self.active_sequences:
            try:
                next_token = self.paged_attention.forward(seq_id)
                if next_token != self.paged_attention.model.config.eos_token_id:
                    self.queue.put((self.active_sequences[seq_id], seq_id))
                    return True
                else:
                    self.remove_sequence(seq_id)
                    return False
            except RuntimeError as e:
                if "Not enough free blocks" in str(e):
                    self.handle_out_of_memory()
                    # Preempt this sequence
                    self.preempted_sequences.put((self.active_sequences[seq_id], seq_id))
                    return False
                else:
                    raise e
        return False

    def handle_out_of_memory(self):
        # Implement the preemption logic
        while True:
            if not self.paged_attention.kv_cache.free_blocks:
                # No free blocks, need to preempt
                seq_to_preempt = max(self.active_sequences, key=self.active_sequences.get)
                self.preempt_sequence(seq_to_preempt)
            else:
                # We have free blocks now, can continue processing
                break

    def preempt_sequence(self, seq_id: int):
        seq = self.paged_attention.seq_manager.sequences[seq_id]
        # Swap out all blocks of the sequence
        self.paged_attention.kv_cache._swap_out(seq_id)
        # Move the sequence to the preempted queue
        self.preempted_sequences.put((self.active_sequences[seq_id], seq_id))
        # Remove from active sequences, but don't remove from sequence manager
        del self.active_sequences[seq_id]