import torch
from transformers import GPT2Config
from vllmini.paged_attention import PagedAttention
from vllmini.kv_cache import KVCache

def test_paged_attention():
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

    def generate_input(batch_size, seq_len):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        upper_triangular = torch.triu(torch.full((seq_len, seq_len), float('inf')), diagonal=1)
        # Expand the upper triangular matrix to match the desired shape
        attention_mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
        attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len).to(device)
        return input_ids, position_ids, attention_mask

    print("\nTesting prefill...")
    batch_size, seq_len = 1, 64
    input_ids, position_ids, attention_mask = generate_input(batch_size, seq_len)
    seq_id = 1

    # Create slot_mapping for prefill
    slot_mapping = torch.arange(seq_len, dtype=torch.long, device=paged_attention.device)

    output = paged_attention.forward(
        input_ids, 
        position_ids, 
        attention_mask, 
        use_cache=True, 
        is_prefill=True, 
        seq_id=seq_id,
        slot_mapping=slot_mapping
    )

    assert output.shape == (batch_size, seq_len, config.vocab_size), "Incorrect output shape for prefill"
    assert len(kv_cache.allocated_blocks[seq_id]) == (seq_len + block_size - 1) // block_size, "Incorrect number of blocks allocated for prefill"
    print("Prefill test passed.")

    print("\nTesting decoding (single token generation)...")
    input_ids, position_ids, attention_mask = generate_input(batch_size, 1)

    # Create slot_mapping for decoding
    slot_mapping = torch.tensor([seq_len], dtype=torch.int64, device=paged_attention.device)

    output = paged_attention.forward(
        input_ids, 
        position_ids, 
        attention_mask, 
        use_cache=True, 
        is_prefill=False, 
        seq_id=seq_id,
        slot_mapping=slot_mapping
    )

    assert output.shape == (batch_size, 1, config.vocab_size), "Incorrect output shape for decoding"
    print("Decoding test passed.")

    print("\nTesting multiple sequence handling...")
    seq_id_2 = 2
    input_ids, position_ids, attention_mask = generate_input(1, 32)
    paged_attention.forward(input_ids, position_ids, attention_mask, use_cache=True, is_prefill=True, seq_id=seq_id_2)
    assert seq_id_2 in kv_cache.allocated_blocks, "Second sequence not allocated in KV cache"
    print("Multiple sequence handling test passed.")

    print("\nTesting block allocation...")
    initial_free_blocks = len(kv_cache.free_blocks)
    seq_id_3 = 3
    input_ids, position_ids, attention_mask = generate_input(1, 128)
    paged_attention.forward(input_ids, position_ids, attention_mask, use_cache=True, is_prefill=True, seq_id=seq_id_3)
    assert len(kv_cache.free_blocks) < initial_free_blocks, "Blocks not allocated for new sequence"
    print("Block allocation test passed.")

    print("\nTesting sequence continuation...")
    seq_id_3 = 3
    initial_seq_len = 128  # This was the length of the sequence we prefilled for seq_id_3

    continuation_input_ids, continuation_position_ids, continuation_attention_mask = generate_input(1, 1)

    # Calculate the correct slot_mapping for continuation
    continuation_slot_mapping = torch.tensor([initial_seq_len], dtype=torch.int64, device=paged_attention.device)

    continuation_output = paged_attention.forward(
        continuation_input_ids, 
        continuation_position_ids, 
        continuation_attention_mask, 
        use_cache=True, 
        is_prefill=False, 
        seq_id=seq_id_3,
        slot_mapping=continuation_slot_mapping
    )

    assert continuation_output.shape == (1, 1, config.vocab_size), "Incorrect output shape for sequence continuation"
    print("Sequence continuation test passed.")

    print("\nTesting error handling...")
    try:
        paged_attention.forward(input_ids, position_ids, attention_mask, use_cache=True, is_prefill=False, seq_id=None)
        print("Failed: Missing seq_id for decoding did not raise an error")
    except ValueError:
        print("Error handling for missing seq_id passed.")

    try:
        non_existent_seq_id = 999
        paged_attention.forward(input_ids, position_ids, attention_mask, use_cache=True, is_prefill=False, seq_id=non_existent_seq_id)
        print("Failed: Non-existent seq_id did not raise an error")
    except ValueError:
        print("Error handling for non-existent seq_id passed.")

    print("\nTesting long sequence handling...")
    long_seq_len = 2000  # Longer than the model's n_positions
    input_ids, position_ids, attention_mask = generate_input(1, long_seq_len)
    try:
        paged_attention.forward(input_ids, position_ids, attention_mask, use_cache=True, is_prefill=True, seq_id=6)
        print("Long sequence handling test passed.")
    except RuntimeError as e:
        print(f"Failed: Long sequence handling raised an error: {e}")

    print("\nAll tests completed.")

if __name__ == '__main__':
    test_paged_attention()