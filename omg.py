import torch
from paged_attention_cuda import paged_attention_v1

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dimensions
batch_size = 2
num_heads = 4
head_size = 64
seq_len = 128
num_kv_heads = 2
block_size = 16
max_seq_len = 256

# Create dummy tensors
out = torch.zeros(batch_size, num_heads, head_size, device=device)
query = torch.randn(batch_size, num_heads, head_size, device=device)
key_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_size, device=device)
value_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_size, device=device)
block_tables = torch.randint(0, seq_len // block_size, (batch_size, seq_len // block_size), dtype=torch.int32, device=device)
seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

# Other parameters
scale = 1.0 / (head_size ** 0.5)
alibi_slopes = None
kv_cache_dtype = "auto"
kv_scale = 1.0
tp_rank = 0
blocksparse_local_blocks = 0
blocksparse_vert_stride = 1
blocksparse_block_size = 16
blocksparse_head_sliding_step = 0

try:
    paged_attention_v1(
        out, query, key_cache, value_cache, 
        num_kv_heads,  # Make sure this is an integer, not a tensor
        scale,
        block_tables, 
        seq_lens, 
        block_size,  # Make sure this is an integer, not a tensor
        max_seq_len,  # Make sure this is an integer, not a tensor
        alibi_slopes,
        kv_cache_dtype, 
        kv_scale, 
        tp_rank,  # Make sure this is an integer, not a tensor
        blocksparse_local_blocks,  # Make sure this is an integer, not a tensor
        blocksparse_vert_stride,  # Make sure this is an integer, not a tensor
        blocksparse_block_size,  # Make sure this is an integer, not a tensor
        blocksparse_head_sliding_step  # Make sure this is an integer, not a tensor
    )
    print("Paged attention kernel executed successfully!")
    print("Output:", out)
    print("Output shape:", out.shape)
    print("Output sum:", out.sum().item())
except Exception as e:
    print("Error occurred while running paged attention kernel:", str(e))