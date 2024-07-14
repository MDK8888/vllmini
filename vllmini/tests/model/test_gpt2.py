import torch
from transformers import GPT2Config
from vllmini.model.gpt2 import GPT2LMHeadModel

def test_basic_forward_pass():
    # Initialize configuration
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    
    # Initialize model
    model = GPT2LMHeadModel(config)
    
    # Prepare inputs
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    kv_caches = [torch.zeros((batch_size, config.num_attention_heads, seq_length, config.hidden_size // config.num_attention_heads)) 
                 for _ in range(config.num_hidden_layers)]
    block_tables = torch.zeros((batch_size, seq_length), dtype=torch.long)
    seq_lens = torch.full((batch_size,), seq_length, dtype=torch.long)
    max_seq_len = seq_length
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids, position_ids, kv_caches, block_tables, seq_lens, max_seq_len)
    
    # Check output shape
    expected_shape = (batch_size, seq_length, config.vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    
    print("Basic forward pass test passed!")

if __name__ == "__main__":
    test_basic_forward_pass()