#include "cache_kernels.h"
#include <torch/extension.h>

#define TORCH_EXTENSION_NAME paged_attention_cuda

// Declare the CUDA functions for paged attention
void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double kv_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& seq_lens,
    int64_t block_size,
    int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double kv_scale,
    const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride,
    const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1", &paged_attention_v1, "Paged Attention V1");
    m.def("paged_attention_v2", &paged_attention_v2, "Paged Attention V2");

    // Create a submodule for cache operations
    py::module cache_ops = m.def_submodule("cache_ops", "Cache operations");
    cache_ops.def("swap_blocks", &swap_blocks, "Swap blocks in cache");
    cache_ops.def("copy_blocks", &copy_blocks, "Copy blocks in cache");
    cache_ops.def("reshape_and_cache", &reshape_and_cache, "Reshape and cache key/value tensors");
    cache_ops.def("reshape_and_cache_flash", &reshape_and_cache_flash, "Reshape and cache key/value tensors (flash attention version)");
    cache_ops.def("convert_fp8", &convert_fp8, "Convert cache to/from FP8");
}
