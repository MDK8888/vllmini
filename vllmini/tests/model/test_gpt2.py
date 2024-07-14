import torch
from transformers import GPT2Config
from vllmini.model.gpt2 import GPT2LMHeadModel  # Replace 'your_module' with the actual module name

def test_decoding():
    # Initialize model and test inputs
    config = GPT2Config(n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, n_inner=3072)
    model = GPT2LMHeadModel(config)
    model.eval()

    batch_size = 2
    prefill_length = 10
    vocab_size = config.vocab_size
    block_size = 16  # Make sure this matches the block_size in your GPT2Attention class

    # Prefill step
    input_ids = torch.randint(0, vocab_size, (batch_size, prefill_length))
    position_ids = torch.arange(0, prefill_length).unsqueeze(0).expand(batch_size, -1)
    attention_mask = torch.ones(batch_size, prefill_length)

    with torch.no_grad():
        _, presents = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            use_cache=True,
            is_prefill=True
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare inputs for decoding step
    next_token = torch.randint(0, vocab_size, (batch_size, 1)).to(device)
    next_position = torch.tensor([[prefill_length]] * batch_size).to(device)
    
    # Convert presents to key_caches and value_caches
    key_caches = [present[0].to(device) for present in presents]
    value_caches = [present[1].to(device) for present in presents]

    # Prepare block tables and sequence lengths
    block_tables = torch.tensor([[i for i in range((prefill_length + 1) // block_size + 1)] for _ in range(batch_size)]).to(torch.int32).to(device)
    seq_lens = torch.tensor([prefill_length + 1] * batch_size).to(torch.int32).to(device)  # +1 for the new token
    max_seq_len = 1024

    model = model.to(device)

    # Decoding step
    with torch.no_grad():
        output, new_presents = model(
            input_ids=next_token,
            position_ids=next_position,
            attention_mask=None,  # Not needed for decoding phase
            use_cache=True,
            is_prefill=False,
            key_caches=key_caches,
            value_caches=value_caches,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len
        )

    # Check output shapes
    assert output.shape == (batch_size, 1, vocab_size), "Output shape mismatch"
    assert len(new_presents) == config.n_layer, "Incorrect number of present states"
    for present in new_presents:
        assert present[0].shape == (batch_size, config.num_attention_heads, 1, config.hidden_size // config.num_attention_heads), f"Key shape mismatch, current key shape: {present[0].shape}"
        assert present[1].shape == (batch_size, config.num_attention_heads, 1, config.hidden_size // config.num_attention_heads), f"Value shape mismatch, current key shape: {present[1].shape}"

    print("Decoding test passed!")

# Run the test
if __name__ == "__main__":
    test_decoding()