from queue import PriorityQueue
import torch
from transformers import GPT2Config
from vllmini.paged_attention import PagedAttention
from vllmini.kv_cache import KVCache
from vllmini.scheduler import Scheduler

def test_scheduler():
    print("Initializing test environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "gpt2"
    num_layers = 12
    config = GPT2Config.from_pretrained(model_name)
    num_blocks = 20
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads
    block_size = 16
    kv_cache = KVCache(num_layers, num_blocks, num_heads, head_size, block_size)
    paged_attention = PagedAttention(model_name, kv_cache, device)
    scheduler = Scheduler(paged_attention)

    def generate_input(batch_size, seq_len):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)
        return input_ids, position_ids, attention_mask

    print("\nTesting sequence addition...")
    seq_id_1 = 1
    input_ids, position_ids, attention_mask = generate_input(1, 64)
    scheduler.add_sequence(seq_id_1, input_ids, position_ids, attention_mask)
    assert seq_id_1 in scheduler.active_sequences, "Sequence 1 not added to active sequences"
    assert seq_id_1 in scheduler.block_tables, "Block table not created for Sequence 1"
    print("Sequence addition test passed.")

    print("\nTesting sequence removal...")
    scheduler.remove_sequence(seq_id_1)
    assert seq_id_1 not in scheduler.active_sequences, "Sequence 1 not removed from active sequences"
    assert seq_id_1 not in scheduler.block_tables, "Block table not removed for Sequence 1"
    print("Sequence removal test passed.")

    print("\nTesting multiple sequence handling...")
    seq_id_2 = 2
    seq_id_3 = 3
    input_ids_2, position_ids_2, attention_mask_2 = generate_input(1, 32)
    input_ids_3, position_ids_3, attention_mask_3 = generate_input(1, 48)
    scheduler.add_sequence(seq_id_2, input_ids_2, position_ids_2, attention_mask_2)
    scheduler.add_sequence(seq_id_3, input_ids_3, position_ids_3, attention_mask_3)
    assert seq_id_2 in scheduler.active_sequences and seq_id_3 in scheduler.active_sequences, "Not all sequences added"
    print("Multiple sequence handling test passed.")

    print("\nTesting out-of-memory handling...")
    # Reduce the number of blocks to force out-of-memory situation
    kv_cache.num_blocks = 5
    scheduler.kv_cache = kv_cache

    # Clear any existing sequences
    scheduler.active_sequences.clear()
    scheduler.block_tables.clear()
    scheduler.queue = PriorityQueue()
    scheduler.preempted_sequences = PriorityQueue()

    print(f"Initial free blocks: {len(kv_cache.free_blocks)}")

    # Fill up the memory
    for i in range(1, 6):  # Add 5 sequences to fill all blocks
        input_ids, position_ids, attention_mask = generate_input(1, block_size)
        scheduler.add_sequence(i, input_ids, position_ids, attention_mask)
        print(f"After adding sequence {i}, free blocks: {len(kv_cache.free_blocks)}")

    # Verify that memory is full
    assert len(kv_cache.free_blocks) == 0, "Memory should be full"

    # Try to add one more sequence
    extra_seq_id = 6
    input_ids, position_ids, attention_mask = generate_input(1, block_size)
    try:
        scheduler.add_sequence(extra_seq_id, input_ids, position_ids, attention_mask)
    except RuntimeError as e:
        if "Not enough free blocks" not in str(e):
            raise e
        print("Expected 'Not enough free blocks' error raised.")

    # Mock the PagedAttention forward method to return a predetermined output
    def mock_forward(*args, **kwargs):
        return torch.ones(1, 1, config.vocab_size, device=device)

    paged_attention.forward = mock_forward

    # Run the scheduler and check for preemption
    print("Running scheduler...")
    preemption_occurred = False
    max_iterations = 1000  # Set a reasonable maximum number of iterations
    for i in range(max_iterations):
        initial_active_sequences = set(scheduler.active_sequences.keys())
        scheduler.run()
        final_active_sequences = set(scheduler.active_sequences.keys())
        
        if initial_active_sequences != final_active_sequences:
            preempted_sequences = initial_active_sequences - final_active_sequences
            if preempted_sequences:
                print(f"Preemption occurred at iteration {i}")
                print(f"Preempted sequences: {preempted_sequences}")
                preemption_occurred = True
                break
        
        if not scheduler.queue.empty() or not scheduler.preempted_sequences.empty():
            continue
        else:
            print("All sequences processed")
            break

    assert preemption_occurred, "No sequences were preempted during out-of-memory handling"

    print(f"\nFinal state:")
    print(f"Active sequences: {scheduler.active_sequences}")
    print(f"Preempted sequences: {scheduler.preempted_sequences.qsize()}")
    print(f"Queue size: {scheduler.queue.qsize()}")
    print(f"Free blocks: {len(kv_cache.free_blocks)}")

    assert len(kv_cache.free_blocks) > 0, "No blocks freed after preemption"
    print("Out-of-memory handling test passed.")

    print("\nAll tests completed.")

if __name__ == '__main__':
    test_scheduler()