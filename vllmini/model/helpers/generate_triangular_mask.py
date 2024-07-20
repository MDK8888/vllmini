import torch 

def generate_triangular_mask(batch_size, num_heads, seq_len):
    # Create an upper triangular matrix with -inf, including the diagonal
    upper_triangular = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float16), diagonal=1)
    
    # Expand the upper triangular matrix to match the desired shape
    mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # shape (batch_size, num_heads, seq_len, seq_len)
    mask = mask.to("cuda", dtype=torch.float16)
    return mask