import torch
from paged_attention_cuda import paged_attention_v1

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dimensions
num_seqs = 2
num_heads = 4
num_kv_heads = 2  # Can be different from num_heads for multi-query attention
head_size = 64
max_seq_len = 256
block_size = 16
num_blocks = 32  # Total number of blocks in the KV cache

# Create dummy tensors
out = torch.zeros(num_seqs, num_heads, head_size, device=device)
query = torch.randn(num_seqs, num_heads, head_size, device=device)

# Key and value caches now use the paged structure
key_cache = torch.randn(num_blocks, num_kv_heads, head_size//8, block_size, 8, device=device)
value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, device=device)

# Block tables map sequence positions to block numbers
# Each sequence can use a different number of blocks
block_tables = torch.randint(0, num_blocks, (num_seqs, max_seq_len // block_size), dtype=torch.int32, device=device)

# Sequence lengths (can be different for each sequence in the batch)
seq_lens = torch.tensor([200, 150], dtype=torch.int32, device=device)

# Other parameters
scale = 1.0 / (head_size ** 0.5)
max_num_blocks_per_seq = max_seq_len // block_size
kv_cache_dtype = "auto"
kv_scale = 1.0
tp_rank = 0

# Set these to default values for vanilla attention
blocksparse_local_blocks = 0
blocksparse_vert_stride = 1
blocksparse_block_size = 16
blocksparse_head_sliding_step = 0

try:
    paged_attention_v1(
        out, 
        query, 
        key_cache, 
        value_cache, 
        num_kv_heads, 
        scale,
        block_tables, 
        seq_lens, 
        block_size, 
        max_seq_len, 
        None,  # alibi_slopes set to None for vanilla attention
        kv_cache_dtype, 
        kv_scale, 
        tp_rank, 
        blocksparse_local_blocks, 
        blocksparse_vert_stride, 
        blocksparse_block_size, 
        blocksparse_head_sliding_step
    )
    print("Paged attention kernel executed successfully!")
    print("Output shape:", out.shape)
    print("Output sum:", out.sum().item())
except Exception as e:
    print("Error occurred while running paged attention kernel:", str(e))