import torch
from vllmini.kv_cache import KVCache

def test_kv_cache():
    num_blocks = 10
    num_heads = 12
    head_size = 64
    block_size = 16
    
    print("Initializing KVCache...")
    kv_cache = KVCache(num_blocks, num_heads, head_size, block_size)

    print("\nTesting initial state...")
    assert len(kv_cache.free_blocks) == num_blocks, "Initial free blocks count is incorrect"
    assert len(kv_cache.allocated_blocks) == 0, "Initial allocated blocks should be empty"
    print("Initial state test passed.")

    print("\nTesting allocation...")
    seq_id, num_blocks_to_allocate = 1, 3
    allocated = kv_cache.allocate(seq_id, num_blocks_to_allocate)
    assert len(allocated) == num_blocks_to_allocate, "Incorrect number of blocks allocated"
    assert len(kv_cache.free_blocks) == num_blocks - num_blocks_to_allocate, "Incorrect number of free blocks after allocation"
    assert len(kv_cache.allocated_blocks[seq_id]) == num_blocks_to_allocate, "Incorrect number of allocated blocks for sequence"
    print("Allocation test passed.")

    print("\nTesting allocation for multiple sequences...")
    kv_cache.allocate(2, 2)
    kv_cache.allocate(3, 1)
    assert len(kv_cache.free_blocks) == 4, "Incorrect number of free blocks after multiple allocations"
    assert len(kv_cache.allocated_blocks) == 3, "Incorrect number of sequences with allocated blocks"
    print("Multiple sequence allocation test passed.")

    print("\nTesting freeing...")
    kv_cache.free(1)
    assert len(kv_cache.free_blocks) == 7, "Incorrect number of free blocks after freeing"
    assert len(kv_cache.allocated_blocks) == 2, "Incorrect number of sequences with allocated blocks after freeing"
    assert 1 not in kv_cache.allocated_blocks, "Freed sequence should not be in allocated blocks"
    print("Freeing test passed.")

    print("\nTesting allocation after freeing...")
    new_allocated = kv_cache.allocate(4, 2)
    assert len(new_allocated) == 2, "Incorrect number of blocks allocated after freeing"
    assert len(kv_cache.free_blocks) == 5, "Incorrect number of free blocks after new allocation"
    assert len(kv_cache.allocated_blocks[4]) == 2, "Incorrect number of allocated blocks for new sequence"
    print("Allocation after freeing test passed.")

    print("\nTesting out-of-blocks allocation...")
    try:
        kv_cache.allocate(5, 6)  # This should raise an exception
        print("Failed: Out-of-blocks allocation did not raise an exception")
    except RuntimeError:
        print("Out-of-blocks allocation test passed.")

    print("\nTesting get_kv_cache...")
    seq_id = 2
    key_cache, value_cache = kv_cache.get_kv_cache(seq_id)
    assert key_cache.shape == (2, num_heads, head_size // 8, block_size, 8), "Incorrect key cache shape"
    assert value_cache.shape == (2, num_heads, head_size, block_size), "Incorrect value cache shape"
    print("Get KV cache test passed.")

    print("\nTesting get_kv_cache for invalid sequence...")
    try:
        kv_cache.get_kv_cache(999)  # This should raise an exception
        print("Failed: Invalid sequence get_kv_cache did not raise an exception")
    except ValueError:
        print("Invalid sequence get_kv_cache test passed.")

    print("\nTesting reshape_and_cache...")
    seq_id = 2
    key = torch.randn(1, num_heads, head_size, device='cuda')
    value = torch.randn(1, num_heads, head_size, device='cuda')
    slot_mapping = torch.tensor([0], dtype=torch.long, device='cuda')
    try:
        kv_cache.reshape_and_cache(key, value, slot_mapping)
        print("Reshape and cache test passed.")
    except Exception as e:
        print(f"Failed: Reshape and cache raised an exception: {e}")

    print("\nTesting copy_blocks...")
    block_mapping = torch.tensor([[0, 2], [1, 3]], dtype=torch.long, device='cuda')
    try:
        kv_cache.copy_blocks(block_mapping)
        print("Copy blocks test passed.")
    except Exception as e:
        print(f"Failed: Copy blocks raised an exception: {e}")

    print("\nTesting complex scenario...")
    kv_cache = KVCache(num_blocks, num_heads, head_size, block_size)  # Reset the cache
    
    kv_cache.allocate(1, 3)
    kv_cache.allocate(2, 2)
    kv_cache.allocate(3, 1)
    assert len(kv_cache.free_blocks) == 4, "Incorrect number of free blocks after multiple allocations"
    assert len(kv_cache.allocated_blocks) == 3, "Incorrect number of sequences with allocated blocks"

    kv_cache.free(2)
    assert len(kv_cache.free_blocks) == 6, "Incorrect number of free blocks after freeing"
    assert len(kv_cache.allocated_blocks) == 2, "Incorrect number of sequences with allocated blocks after freeing"

    kv_cache.allocate(4, 2)
    assert len(kv_cache.free_blocks) == 4, "Incorrect number of free blocks after new allocation"
    assert len(kv_cache.allocated_blocks) == 3, "Incorrect number of sequences with allocated blocks"

    block_mapping = torch.tensor([[0, 5], [1, 6]], dtype=torch.long, device='cuda')
    kv_cache.copy_blocks(block_mapping)

    kv_cache.allocate(5, 4)
    assert len(kv_cache.free_blocks) == 0, "All blocks should be allocated"
    assert len(kv_cache.allocated_blocks) == 4, "Incorrect number of sequences with allocated blocks"
    assert sum(len(blocks) for blocks in kv_cache.allocated_blocks.values()) == num_blocks, "Total allocated blocks should equal total number of blocks"

    print("Complex scenario test passed.")

    print("\nAll tests completed.")

if __name__ == '__main__':
    test_kv_cache()