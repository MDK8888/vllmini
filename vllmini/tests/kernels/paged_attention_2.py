import torch
from paged_attention_cuda import paged_attention_v1

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensions as specified
batch_size = 2
num_heads = 12
head_size = 64
seq_len = 11
num_kv_heads = 12
block_size = 16
max_seq_len = 1024

# Create tensors with specified shapes
out = torch.zeros(batch_size, num_heads, head_size, device=device)
query = torch.randn(batch_size, num_heads, head_size, device=device)
key_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_size, device=device)
value_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_size, device=device)
block_tables = torch.randint(0, seq_len // block_size + 1, (batch_size, 1), dtype=torch.int32, device=device)
seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

# Other parameters
scale = 0.125
alibi_slopes = None
kv_cache_dtype = "auto"
kv_scale = 1.0
tp_rank = 0
blocksparse_local_blocks = 0
blocksparse_vert_stride = 1
blocksparse_block_size = 1
blocksparse_head_sliding_step = 0

try:
    paged_attention_v1(
        out, query, key_cache, value_cache, 
        num_kv_heads,
        scale,
        block_tables, 
        seq_lens, 
        block_size,
        max_seq_len,
        alibi_slopes,
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

# Print shapes of input tensors for verification
print("\nInput tensor shapes:")
print("query shape:", query.shape)
print("key_cache shape:", key_cache.shape)
print("value_cache shape:", value_cache.shape)
print("block_tables:", block_tables)
print("seq_lens:", seq_lens)
print("block_size:", block_size)
print("max_seq_len:", max_seq_len)